import cv2
import torch
import numpy as np
from ultralytics import YOLO
from app.config import Config


class LocalTracker:
    def __init__(self, camera_id):
        self.camera_id = camera_id
        self.model = YOLO(Config.YOLO_MODEL)
        if torch.cuda.is_available():
            self.model.to('cuda')

    def _is_valid_person_box(self, x1, y1, x2, y2, frame_h, frame_w):
        """
        Reject bounding boxes that cannot physically belong to a single person.
        This kills merged-person blobs and random object detections.
        """
        bw = x2 - x1
        bh = y2 - y1

        # Too small → noise
        if bw < Config.MIN_PERSON_WIDTH_PX or bh < Config.MIN_PERSON_HEIGHT_PX:
            return False

        # Too wide or too tall relative to frame → likely a merged blob
        if bw / frame_w > Config.MAX_PERSON_WIDTH_RATIO:
            return False
        if bh / frame_h > Config.MAX_PERSON_HEIGHT_RATIO:
            return False

        # Aspect ratio check: people are taller than wide
        aspect = bh / max(bw, 1)
        if aspect < Config.MIN_ASPECT_RATIO or aspect > Config.MAX_ASPECT_RATIO:
            return False

        return True

    def process_frame(self, frame):
        """
        Run YOLO + BoT-SORT and return validated per-person detections.
        Returns a list of dicts:
            camera_id, track_id, bbox (x1,y1,x2,y2), conf, crop
        """
        h, w = frame.shape[:2]

        results = self.model.track(
            frame,
            persist=True,
            classes=[0],            # person only
            tracker=Config.TRACKER_TYPE,
            conf=Config.DETECTION_CONFIDENCE,
            iou=Config.NMS_IOU,     # LOW → keeps overlapping people separate
            imgsz=Config.DETECTION_IMGSZ,
            device=Config.DEVICE,   # RESTORED for NVIDIA GPU
            verbose=False
        )

        detections = []
        if not results or results[0].boxes is None or results[0].boxes.id is None:
            return detections

        res = results[0]
        boxes     = res.boxes.xyxy.cpu().numpy()
        track_ids = res.boxes.id.int().cpu().numpy()
        confs     = res.boxes.conf.cpu().numpy()

        for box, track_id, conf in zip(boxes, track_ids, confs):
            x1, y1, x2, y2 = map(int, box)

            # Clamp to frame
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # ── Sanity filter: drop merged blobs & object false-positives ──
            if not self._is_valid_person_box(x1, y1, x2, y2, h, w):
                continue

            crop = frame[y1:y2, x1:x2]
            detections.append({
                'camera_id': self.camera_id,
                'track_id':  f"{self.camera_id}-T{track_id}",
                'bbox':      (x1, y1, x2, y2),
                'conf':      float(conf),
                'crop':      crop,
            })

        return detections
