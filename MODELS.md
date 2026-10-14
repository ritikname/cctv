# Model Reference Guide for CCTV Tracking & Re-ID

This document serves as the technical specification and reference manual for all deep learning models, visual feature extractors, and tracking algorithms used in this CCTV Video Analytics system. It enables developers and automated AI models to understand model architectures, input/output tensor dimensions, pretrained weights origins, and runtime deployment details.

---

## 1. System Model Architecture Overview

The pipeline integrates three visual analysis layers:

```
[ CCTV Video Stream ]
          │
          ▼
┌────────────────────────────────────────┐
│ 1. Detection & Pose Estimation         │ ───► Bounding boxes & Foot keypoints
│    (Ultralytics YOLO11x / YOLO11x-Pose)│
└────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────┐
│ 2. Multi-Object Local Tracking         │ ───► Trajectories & Frame-to-frame association
│    (BoT-SORT + Kalman Filter)          │
└────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────┐
│ 3. Global Person Re-Identification     │ ───► 512-dim / 2048-dim embeddings +
│    - Primary: OSNet (osnet_x1_0)       │      34-dim HSV Clothing profiles
│    - Fallback: ResNet-50 (Torchvision) │
│    - Clothing: Torso/Leg Color Hists   │
└────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────┐
│ 4. Analytics & Identity Management     │ ───► Persistent Visitor IDs (P-001...),
│    - Global Identity Manager Gallery   │      Gate Line Crossing (IN/OUT),
│    - Tripwire Counter                  │      Statue/Mannequin Filter,
│    - Multi-Person Group Detection      │      Likely Group Clustering
└────────────────────────────────────────┘
```

---

## 2. Models Specification Table

| Model Component | Architecture / Model Identifier | Weights File | Input Dimensions | Output Dimensions | Framework | Source / Download Link |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Object Detector** *(Default)* | YOLO11x (`yolo11x`) | `yolo11x.pt` | `(3, 736, 736)` or `(3, 640, 640)` | BBoxes: `(N, 6)` `[x1, y1, x2, y2, conf, cls]` | Ultralytics PyTorch | [Ultralytics YOLO11x Release](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt) |
| **Pose & Keypoint Detector** *(Alternative)* | YOLO11x-Pose (`yolo11x-pose`) | `yolo11x-pose.pt` | `(3, 736, 736)` or `(3, 640, 640)` | Keypoints: `(N, 17, 3)` `[x, y, conf]` + BBoxes | Ultralytics PyTorch | [Ultralytics YOLO11x-Pose Release](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-pose.pt) |
| **Lightweight Detector** *(Low-VRAM)* | YOLOv8m / YOLO11m | `yolov8m.pt` / `yolo11m.pt` | `(3, 640, 640)` | BBoxes: `(N, 6)` | Ultralytics PyTorch | [Ultralytics YOLOv8m Release](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt) |
| **Person Re-ID (Primary)** | Omni-Scale Network (`osnet_x1_0`) | `osnet_x1_0_msmt17...pth` | `(3, 256, 128)` | `(512,)` Normalized Vector | Torchreid (PyTorch) | [Torchreid Model Zoo](https://github.com/KaiyangZhou/deep-person-reid) |
| **Person Re-ID (Fallback)** | ResNet-50 (`resnet50`) | Standard Torchvision weights | `(3, 256, 128)` | `(2048,)` Projected Vector | Torchvision | PyTorch Hub (`torchvision.models.resnet50`) |
| **Clothing Profile Extractor** | Multi-Region HSV Histograms | Pure algorithmic (No weights required) | BGR Person Crop `(H, W, 3)` | Upper Torso + Lower Legs `(34,)` L2 Normalized | OpenCV / NumPy | Integrated in `app/reid/clothing.py` |
| **Multi-Object Tracker** | BoT-SORT (Kalman Filter + GMC) | `botsort.yaml` config | BBoxes & Detections | Track IDs & Bounding Boxes | Ultralytics Tracker | Embedded in Ultralytics |

---

## 3. Deep Learning Model Details

### 3.1 Detection & Tracking: YOLO11x & YOLO11x-Pose
- **Implementation**: [`app/tracking/local_tracker.py`](file:///c:/Users/ritik/Downloads/my%20projects/mine/multi_cam_tracking/app/tracking/local_tracker.py) & [`app/config.py`](file:///c:/Users/ritik/Downloads/my%20projects/mine/multi_cam_tracking/app/config.py).
- **Class Filtering**: Only class `0` (`person`) is tracked; all other classes (vehicles, chairs, bags) are filtered out.
- **Foot Point Calculation**:
  - In bounding-box mode: Midpoint of bottom edge: `( (x1 + x2)/2, y2 )`.
  - In pose mode: Ankle keypoints (`left_ankle`, `right_ankle`) are averaged to obtain accurate ground-plane contact coordinates, critical for tripwire entry/exit crossing evaluation.
- **Configuration**:
  - Image size: `736` (high resolution to capture distant/background persons).
  - Confidence threshold: `0.50`.
  - NMS IoU threshold: `0.75`.

### 3.2 Person Re-Identification: OSNet (`osnet_x1_0`)
- **Paper**: *Omni-Scale Feature Learning for Person Re-Identification* (Zhou et al., ICCV 2019).
- **Architecture**: Residual blocks with multiple convolutional streams capturing features at various scales (omni-scale).
- **Training Dataset**: Trained on MSMT17 (Multi-Scene Multi-Time person re-identification benchmark) with combineall, label smoothing, and AMSGrad optimizer.
- **Embedding Properties**:
  - Output dimension: 512-dimensional vector.
  - L2-normalized: $\|f\|_2 = 1.0$.
  - Distance metric: Cosine similarity $\cos(\theta) = f_1 \cdot f_2 \in [-1, 1]$.
- **Auto-Download**: When `torchreid` is installed, calling `FeatureExtractor(model_name="osnet_x1_0")` automatically checks `~/.cache/torch/checkpoints/` and downloads the weights if missing.

### 3.3 Fallback Person Re-ID: ResNet-50
- **Implementation**: [`app/reid/model.py`](file:///c:/Users/ritik/Downloads/my%20projects/mine/multi_cam_tracking/app/reid/model.py).
- **Architecture**: Standard ResNet-50 backbone initialized with PyTorch `IMAGENET1K_V2` weights, with classification head (`fc`) replaced by `nn.Identity()`.
- **Embedding Dimension**: 2048-dimensional feature representation concatenated with multi-layer features and normalised color histogram.

### 3.4 Clothing Color Profile Extractor
- **Implementation**: [`app/reid/clothing.py`](file:///c:/Users/ritik/Downloads/my%20projects/mine/multi_cam_tracking/app/reid/clothing.py).
- **Rationale**: When a visitor enters facing the camera and later turns their back to the camera, facial and frontal features vanish, causing CNN feature similarity to drop to $0.40 - 0.55$. However, clothing colors remain invariant.
- **Feature Extraction**:
  - **Upper Body (Torso)**: $15\% - 55\%$ height, central $64\%$ width.
  - **Lower Body (Legs/Pants)**: $55\% - 88\%$ height, central $60\%$ width.
  - Channels: 18 Hue bins (primary color), 8 Saturation bins (purity), 8 Value bins (shade).
  - Normalization: Independent L2-normalization for upper and lower vectors, combined via weighted cosine similarity ($0.60 \times \text{upper} + 0.40 \times \text{lower}$).

---

## 4. Setup and Weights Placement

### Automatic Download (Recommended)
The Ultralytics and PyTorch models download weights automatically upon first invocation:
```python
from ultralytics import YOLO
# Automatically downloads yolo11x.pt into current directory if not present:
model = YOLO("yolo11x.pt")
```

For OSNet via Torchreid:
```bash
pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
```

### Manual Download (Offline / Air-Gapped Environments)
If running offline, place the weight files in either the project root or the `models/` directory:

1. **YOLO11x**:
   ```bash
   curl -L -o models/yolo11x.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt
   ```
2. **YOLO11x-Pose**:
   ```bash
   curl -L -o models/yolo11x-pose.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-pose.pt
   ```
3. **OSNet MSMT17**:
   Download from Torchreid release drive or model zoo and name it:
   `osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth`

---

## 5. Model Configuration Options

All thresholds and parameters can be adjusted in [`app/config.py`](file:///c:/Users/ritik/Downloads/my%20projects/mine/multi_cam_tracking/app/config.py) or dynamically via the PySide6 desktop GUI:

- `Config.YOLO_MODEL`: Path or name of YOLO model (e.g. `"yolo11x.pt"`, `"yolo11m.pt"`, `"models/yolo11x.pt"`).
- `Config.REID_MODEL`: Identifier for Re-ID network (`"osnet_x1_0"`).
- `Config.REID_MATCH_THRESHOLD`: Cosine similarity threshold for global Re-ID matching (default: `0.75`).
- `Config.ACTIVE_VISITOR_REID_THRESHOLD`: Threshold to re-identify temporarily lost active visitors (default: `0.75`).
- `Config.GALLERY_SIZE`: Multi-view sliding window size per person (default: `8` viewpoints).
