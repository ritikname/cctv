import torch

class Config:
    # ─── YOLO Detection ───────────────────────────────────────────────────────
    YOLO_MODEL              = "yolo11x.pt"          # Standard object detection model
    DETECTION_CONFIDENCE    = 0.50          
    NMS_IOU                 = 0.75          
    DETECTION_IMGSZ         = 736                   # Restored max resolution

    # ─── Bounding Box Sanity Filters ─────────────────────────────────────────
    MAX_PERSON_WIDTH_RATIO  = 0.80          
    MAX_PERSON_HEIGHT_RATIO = 0.95
    MIN_PERSON_WIDTH_PX     = 20            # Lowered: Pose model naturally rejects shelves, allowing us to track tiny background people
    MIN_PERSON_HEIGHT_PX    = 50            # Lowered
    MIN_ASPECT_RATIO        = 0.30          # Further relaxed for heavily occluded folks
    MAX_ASPECT_RATIO        = 6.0

    # ─── Tracking ────────────────────────────────────────────────────────────
    TRACKER_TYPE            = "botsort.yaml"
    TRACK_BUFFER            = 120           # Increased from 45 to 120 (4 seconds). Helps BoT-SORT hold onto IDs during long occlusions.

    # ─── Rolling Frame Gallery ────────────────────────────────────────────────
    # Each known person keeps a sliding window of their N most-recent embeddings.
    # Old ones drop out automatically as new frames come in.
    GALLERY_SIZE            = 8             # rolling window size per person
    GALLERY_UPDATE_INTERVAL = 10            # update gallery every N frames for active tracks
    MIN_CROP_WIDTH          = 20            # Minimum width for feature extraction
    MIN_CROP_HEIGHT         = 40            # Minimum height for feature extraction

    # ─── Re-ID Matching ───────────────────────────────────────────────────────
    # We compare the NEW track's embedding against every embedding in the gallery.
    # The highest single match is used (best-of-gallery strategy).
    REID_MODEL                  = "osnet_x1_0"
    REID_MATCH_THRESHOLD        = 0.75      
    REID_STRONG_MATCH_THRESHOLD = 0.75      
    ACTIVE_VISITOR_REID_THRESHOLD = 0.75    # Minimum appearance similarity to recover an active visitor
    MAX_REID_TIME_GAP           = 1800      # seconds – ignore identities older than 30 min
    VISITOR_RETENTION_TIME      = 3600      # seconds (1 hour) – retain departed visitor gallery for re-entry recognition

    # ─── Local VLM Verifier (Ollama) ──────────────────────────────────────────
    VLM_ENABLED                 = False
    VLM_BASE_URL                = "http://localhost:11434"
    VLM_MODEL                   = "qwen3.5:2b-q4_K_M"
    VLM_TIMEOUT                 = 3.0
    VLM_AMBIGUITY_LOW           = 0.68
    VLM_AMBIGUITY_HIGH          = 0.78
    VLM_MARGIN_THRESHOLD        = 0.08

    # ─── Camera Topology ─────────────────────────────────────────────────────
    MAX_CAMERA_TRANSITION_TIME  = 120

    # ─── Runtime ─────────────────────────────────────────────────────────────
    PROCESSING_FPS  = None
    DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

    # ─── Output ───────────────────────────────────────────────────────────────
    OUTPUT_DIR  = "outputs"
    MODELS_DIR  = "models"
