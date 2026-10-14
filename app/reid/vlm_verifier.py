"""
Local VLM Second-Stage Corroborator (Ollama).

Uses a local vision-language model (e.g. qwen3.5:2b-q4_K_M via Ollama) strictly as
a non-blocking, asynchronous second-stage tie-breaker for ambiguous Re-ID matches.
Hard business rules (gate crossing, co-presence, spatio-temporal continuity) ALWAYS
take precedence and can NEVER be overridden by the VLM.
"""

import os
import base64
import json
import logging
import threading
import urllib.request
import urllib.error
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, List, Any, Tuple

logger = logging.getLogger(__name__)


class VLMVerifier:
    """
    Second-stage identity verifier using a local Ollama Vision-Language Model.
    Runs strictly locally (e.g. http://localhost:11434) with zero cloud dependencies.
    Provides non-blocking background evaluation and verdict caching so video tracking
    is never frozen or stalled.
    """
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3.5:2b-q4_K_M",
        enabled: bool = True,
        timeout: float = 6.0,
        ambiguity_low: float = 0.68,
        ambiguity_high: float = 0.78,
        margin_threshold: float = 0.08,
        persist_dir: str = "outputs/reference_crops",
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.enabled = enabled
        self.timeout = timeout
        self.ambiguity_low = ambiguity_low
        self.ambiguity_high = ambiguity_high
        self.margin_threshold = margin_threshold
        self.persist_dir = persist_dir

        self._lock = threading.Lock()
        # gid -> list of base64 jpgs
        self.reference_crops: Dict[str, List[str]] = {}
        # (track_id, gid) -> verdict dict
        self.results: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.in_flight: Dict[Tuple[str, str], bool] = {}

        # Asynchronous background executor: max 1 worker to not overload local GPU
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="VLMWorker")

        self._load_persisted_crops()

    def _load_persisted_crops(self):
        """Load any previously saved reference crops from disk."""
        if not os.path.exists(self.persist_dir):
            return
        try:
            for gid in os.listdir(self.persist_dir):
                gid_dir = os.path.join(self.persist_dir, gid)
                if not os.path.isdir(gid_dir):
                    continue
                crops_b64 = []
                for fname in sorted(os.listdir(gid_dir)):
                    if fname.endswith((".jpg", ".png")):
                        fpath = os.path.join(gid_dir, fname)
                        img = cv2.imread(fpath)
                        if img is not None and img.size > 0:
                            success, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                            if success:
                                crops_b64.append(base64.b64encode(buf.tobytes()).decode("utf-8"))
                if crops_b64:
                    with self._lock:
                        self.reference_crops[gid] = crops_b64[:3]
        except Exception as e:
            logger.debug("Failed loading persisted crops: %s", e)

    def is_available(self) -> bool:
        """Check if local Ollama service is reachable."""
        if not self.enabled:
            return False
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                return resp.status == 200
        except Exception:
            return False

    def store_reference_crop(self, gid: str, crop: np.ndarray):
        """Store up to 3 high-quality reference crops in memory and persist to disk."""
        if crop is None or crop.size == 0:
            return
        with self._lock:
            if gid not in self.reference_crops:
                self.reference_crops[gid] = []
            if len(self.reference_crops[gid]) >= 3:
                return

            success, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if success:
                b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
                self.reference_crops[gid].append(b64)
                crop_idx = len(self.reference_crops[gid]) - 1
            else:
                return

        # Persist to disk asynchronously / safely
        try:
            gid_dir = os.path.join(self.persist_dir, gid)
            os.makedirs(gid_dir, exist_ok=True)
            fpath = os.path.join(gid_dir, f"crop_{crop_idx}.jpg")
            cv2.imwrite(fpath, crop)
        except Exception as e:
            logger.debug("Could not persist reference crop: %s", e)

    def should_verify(self, top_sim: float, second_sim: Optional[float], margin: float) -> bool:
        """Check if a match falls into the ambiguous zone requiring VLM corroboration."""
        if not self.enabled:
            return False
        if self.ambiguity_low <= top_sim <= self.ambiguity_high:
            if second_sim is not None and margin < self.margin_threshold:
                return True
        return False

    def get_cached_verdict(self, track_id: str, gid: str) -> Optional[Dict[str, Any]]:
        """Return cached verdict if available, else None."""
        key = (track_id, gid)
        with self._lock:
            return self.results.get(key)

    def request_verification_async(self, query_crop: np.ndarray, gid: str, track_id: str):
        """
        Submit a background verification task without blocking the video frame processing loop.
        """
        if not self.enabled or query_crop is None or query_crop.size == 0:
            return
        key = (track_id, gid)
        with self._lock:
            if key in self.results or self.in_flight.get(key, False):
                return
            self.in_flight[key] = True

        crop_copy = query_crop.copy()
        self._executor.submit(self._async_worker, crop_copy, gid, track_id)

    def _async_worker(self, query_crop: np.ndarray, gid: str, track_id: str):
        try:
            verdict = self.verify_candidate(query_crop, gid)
            with self._lock:
                self.results[(track_id, gid)] = verdict
        except Exception as e:
            logger.debug("VLM async worker exception: %s", e)
        finally:
            with self._lock:
                self.in_flight[(track_id, gid)] = False

    def verify_candidate(self, query_crop: np.ndarray, gid: str) -> Dict[str, Any]:
        """
        Verify if query_crop matches the reference crops for gid using local Ollama.
        Returns dict with keys: match (bool), confidence (float), reason (str), status (str).
        """
        if not self.enabled:
            return {"match": False, "confidence": 0.0, "reason": "VLM disabled", "status": "SKIPPED"}

        with self._lock:
            ref_crops = list(self.reference_crops.get(gid, []))

        if not ref_crops:
            return {"match": False, "confidence": 0.0, "reason": "No reference crop stored", "status": "NO_REFERENCE"}

        if query_crop is None or query_crop.size == 0:
            return {"match": False, "confidence": 0.0, "reason": "Empty query crop", "status": "INVALID_CROP"}

        success, buf = cv2.imencode(".jpg", query_crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            return {"match": False, "confidence": 0.0, "reason": "Crop encoding failed", "status": "ENCODE_ERROR"}

        query_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        
        # Use primary entrance crop and secondary view crop if available
        images = [ref_crops[0]]
        if len(ref_crops) > 1:
            images.append(ref_crops[1])
        images.append(query_b64)

        if len(images) == 3:
            prompt = (
                "You are an expert CCTV visual identity corroborator. "
                "Image 1 and Image 2 are reference views of a customer recorded at the store entrance. "
                "Image 3 is a query observation of a customer inside the retail store. "
                "Carefully compare: 1) Clothing patterns and colors (upper garment and pants/skirt), "
                "2) Footwear, 3) Handbags or backpacks carried. "
                "Are Image 3 and the reference images (Image 1/2) depicting the EXACT same person? "
                "Respond in valid JSON only with keys: "
                '{"match": true/false, "confidence": float between 0.0 and 1.0, "reason": "brief rationale"}'
            )
        else:
            prompt = (
                "You are an expert CCTV visual identity corroborator. "
                "Image 1 is a reference view of a customer recorded at the store entrance. "
                "Image 2 is a query observation of a customer inside the retail store. "
                "Carefully compare: 1) Upper garment color/pattern, 2) Lower garment color/pattern, "
                "3) Footwear, 4) Any carried items. "
                "Are Image 2 and Image 1 depicting the EXACT same person? "
                "Respond in valid JSON only with keys: "
                '{"match": true/false, "confidence": float between 0.0 and 1.0, "reason": "brief rationale"}'
            )

        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": images,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.05,
            }
        }

        try:
            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{self.base_url}/api/generate",
                data=req_data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status == 200:
                    resp_body = json.loads(resp.read().decode("utf-8"))
                    res_text = resp_body.get("response", "{}")
                    parsed = json.loads(res_text)
                    return {
                        "match": bool(parsed.get("match", False)),
                        "confidence": float(parsed.get("confidence", 0.0)),
                        "reason": str(parsed.get("reason", "")),
                        "status": "SUCCESS"
                    }
        except urllib.error.URLError as e:
            logger.debug("VLM offline or connection refused: %s", e)
            return {"match": False, "confidence": 0.0, "reason": f"VLM unavailable ({e})", "status": "UNAVAILABLE"}
        except Exception as e:
            logger.debug("VLM verification failed or timed out: %s", e)
            return {"match": False, "confidence": 0.0, "reason": str(e), "status": "TIMEOUT"}

        return {"match": False, "confidence": 0.0, "reason": "Unknown response", "status": "FAILED"}
