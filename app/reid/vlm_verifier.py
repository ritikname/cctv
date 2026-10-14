"""
Local VLM Second-Stage Corroborator (Ollama).

Uses a local vision-language model (e.g. qwen3.5:2b-q4_K_M via Ollama) strictly as
a second-stage tie-breaker for ambiguous Re-ID matches (e.g. margin < 0.08).
Hard business rules (gate crossing, co-presence, spatio-temporal continuity) ALWAYS
take precedence and can NEVER be overridden by the VLM.
"""

import base64
import json
import logging
import urllib.request
import urllib.error
import cv2
import numpy as np
from typing import Optional, Dict, List, Any

logger = logging.getLogger(__name__)


class VLMVerifier:
    """
    Second-stage identity verifier using a local Ollama Vision-Language Model.
    Runs strictly locally (e.g. http://localhost:11434) with zero cloud dependencies.
    """
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3.5:2b-q4_K_M",
        enabled: bool = False,
        timeout: float = 3.0,
        ambiguity_low: float = 0.68,
        ambiguity_high: float = 0.78,
        margin_threshold: float = 0.08,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.enabled = enabled
        self.timeout = timeout
        self.ambiguity_low = ambiguity_low
        self.ambiguity_high = ambiguity_high
        self.margin_threshold = margin_threshold
        # Cache up to 3 reference crops per visitor: gid -> list of base64 jpgs
        self.reference_crops: Dict[str, List[str]] = {}

    def is_available(self) -> bool:
        """Check if local Ollama service is reachable."""
        if not self.enabled:
            return False
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                return resp.status == 200
        except Exception:
            return False

    def store_reference_crop(self, gid: str, crop: np.ndarray):
        """Store up to 3 high-quality reference crops for a confirmed visitor."""
        if crop is None or crop.size == 0:
            return
        if gid not in self.reference_crops:
            self.reference_crops[gid] = []
        if len(self.reference_crops[gid]) >= 3:
            return

        success, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if success:
            b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
            self.reference_crops[gid].append(b64)

    def should_verify(self, top_sim: float, second_sim: Optional[float], margin: float) -> bool:
        """Check if a match falls into the ambiguous zone requiring VLM corroboration."""
        if not self.enabled:
            return False
        if self.ambiguity_low <= top_sim <= self.ambiguity_high:
            if second_sim is not None and margin < self.margin_threshold:
                return True
        return False

    def verify_candidate(self, query_crop: np.ndarray, gid: str) -> Dict[str, Any]:
        """
        Verify if query_crop matches the reference crops for gid.
        Returns dict with keys: match (bool), confidence (float), reason (str), status (str).
        """
        if not self.enabled:
            return {"match": False, "confidence": 0.0, "reason": "VLM disabled", "status": "SKIPPED"}

        ref_crops = self.reference_crops.get(gid, [])
        if not ref_crops:
            return {"match": False, "confidence": 0.0, "reason": "No reference crop stored", "status": "NO_REFERENCE"}

        if query_crop is None or query_crop.size == 0:
            return {"match": False, "confidence": 0.0, "reason": "Empty query crop", "status": "INVALID_CROP"}

        success, buf = cv2.imencode(".jpg", query_crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            return {"match": False, "confidence": 0.0, "reason": "Crop encoding failed", "status": "ENCODE_ERROR"}

        query_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        ref_b64 = ref_crops[0]  # Primary reference crop from gate entrance

        prompt = (
            "You are a computer vision assistant verifying person re-identification. "
            "Compare Image 1 (reference crop from store entrance) and Image 2 (query crop from store interior). "
            "Analyze: 1) Upper clothing color, 2) Lower clothing color, 3) Footwear, 4) Carrying items / bags. "
            "Respond in valid JSON only with keys: "
            '{"match": true/false, "confidence": float between 0.0 and 1.0, "reason": "brief explanation"}'
        )

        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [ref_b64, query_b64],
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.1,
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
            logger.info("VLM offline or unavailable (%s), skipping VLM corroboration", e)
            return {"match": False, "confidence": 0.0, "reason": f"VLM unavailable ({e})", "status": "UNAVAILABLE"}
        except Exception as e:
            logger.warning("VLM verification failed or timed out: %s", e)
            return {"match": False, "confidence": 0.0, "reason": str(e), "status": "ERROR"}

        return {"match": False, "confidence": 0.0, "reason": "Unknown response", "status": "FAILED"}
