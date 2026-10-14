# Multi-Camera CCTV Analytics & Person Re-Identification

A retail and commercial CCTV analytics system engineered for tracking, counting, and re-identifying visitors across multiple cameras. Combines YOLO object/pose detection, BoT-SORT tracking, Torchreid OSNet / ResNet-50 appearance embeddings, invariant clothing color histograms, tripwire gate crossing validation, and group detection.

---

## Key Features

- **Accurate Gate Entry & Exit Counting:** Directional line-crossing validation with hysteresis prevents duplicate counts and ignores people lingering outside or walking behind glass.
- **Persistent Global Visitor IDs:** Distinguishes temporary camera tracker IDs (`cam1-T12`) from persistent visitor identities (`P-001`, `P-002`).
- **Front-to-Back Viewpoint Turn Recovery:** Fuses deep appearance embeddings with upper/lower body HSV clothing histograms. Visitors who turn away from the camera or temporarily leave frame recover their identities upon reappearance.
- **Statue & Mannequin Rejection:** Spatial immobility and temporal motion verification filters prevent static mannequins, posters, and statues from receiving visitor IDs while retaining the ability to track real people who pause and resume moving.
- **Short-Visit Exclusion:** Configurable visit duration filtering (e.g. 5 seconds) to ignore transient passersby from official gate analytics.
- **Group Detection & Dynamics:** Real-time spatial clustering tracks multi-person groups (`G1`, `G2`) based on proximity and co-movement duration.
- **Interactive Desktop UI:** Built with PySide6, featuring interactive gate line drawing, ROI adjustment, real-time metrics, and live video overlays.

---

## Model Specifications & References

> 📖 **Full Model Documentation:** See [MODELS.md](MODELS.md) for detailed architecture descriptions, input/output tensor shapes, weights download sources, and offline deployment instructions.

| Component | Default Model | Framework | Output |
| :--- | :--- | :--- | :--- |
| **Object Detection & Pose** | YOLO11x (`yolo11x.pt`) | Ultralytics PyTorch | Bounding boxes & keypoint foot coordinates |
| **Local Tracking** | BoT-SORT (`botsort.yaml`) | Ultralytics Tracker | Trajectories & track IDs |
| **Person Re-ID (Primary)** | OSNet (`osnet_x1_0`) | Torchreid | 512-dim L2-normalized visual embedding |
| **Person Re-ID (Fallback)** | ResNet-50 | Torchvision | 2048-dim feature representation |
| **Clothing Profile** | Multi-Region HSV | Pure Python / OpenCV | 34-dim upper/lower color histogram |

*Note: Large model checkpoint files (`.pt`, `.pth`) are excluded from Git repository to keep code downloads fast. Ultralytics and Torchreid download them automatically on first run, or you can place them into `models/`.*

---

## Repository Structure

```
.
├── app/
│   ├── analytics/       # Tripwire counter, gate crossing & group detector
│   ├── cameras/         # Multi-camera topology & spatial transitions
│   ├── config.py        # Centralized system & model configuration
│   ├── events/          # Structured JSON/CSV logging
│   ├── identity/        # Global Identity Manager, gallery & Re-ID fusion
│   ├── reid/            # OSNet/ResNet feature extractors & clothing histograms
│   ├── tracking/        # BoT-SORT local tracking & foot-point extraction
│   └── video/           # Multi-threaded video stream processor
├── frontend/
│   ├── desktop_app.py   # PySide6 desktop GUI with live video & metrics
│   ├── interactive_canvas.py # Click-and-drag gate line and ROI configuration
│   └── streamlit_app.py # Web-based analytics viewer
├── models/
│   └── README.md        # Checkpoint directory guide (weights omitted from Git)
├── tests/
│   └── test_scenarios.py# Unit tests (occlusion, Re-ID recovery, gate crossings)
├── counthum.py          # Standalone analytics pipeline script
├── main.py              # CLI entry point
├── MODELS.md            # Comprehensive model specifications for developers & AI
├── requirements.txt     # Python dependencies
└── start_gpu.bat        # Windows launch shortcut
```

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ritikname/cctv.git
   cd cctv
   ```

2. **Install core dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Torchreid (Recommended for optimal Re-ID):**
   ```bash
   pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
   ```
   *(If omitted, the system seamlessly falls back to Torchvision ResNet-50).*

---

## Running the Application

### 1. Interactive Desktop GUI (Recommended)
```bash
python frontend/desktop_app.py
```
Or on Windows:
```bash
start_gpu.bat
```
- Click **"Browse Video"** to load camera feeds.
- Adjust the **Gate Line** by dragging endpoints on the live canvas.
- Toggle **Inside Direction** (`below` / `above`).
- Observe live counters for Active Visitors, Gate In, Gate Out, Statues, and Groups.

### 2. Command Line Interface (CLI)
```bash
python main.py --videos path/to/cam1.mp4 path/to/cam2.mp4
```

### 3. Streamlit Dashboard
```bash
streamlit run frontend/streamlit_app.py
```

---

## Running Automated Tests

Run the comprehensive test suite (39 scenario tests covering tracking, tripwire gating, occlusion recovery, statue filtering, and group detection):

```bash
python -m unittest discover tests
```

---

## License

This project is licensed under the MIT License.
