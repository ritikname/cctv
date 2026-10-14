# Model Checkpoints Directory

This directory stores offline pretrained weight files (`.pt`, `.pth`, `.onnx`, etc.) used by the CCTV tracking pipeline.

> **Note:** Model weight binaries are excluded from Git to keep the repository lightweight. See [../MODELS.md](../MODELS.md) for full technical documentation, architecture specifications, and download sources.

## Expected Files (Optional for Offline Execution)
- `yolo11x.pt` or `yolo11x-pose.pt` (Object Detection & Pose Estimation)
- `osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth` (Torchreid OSNet Person Re-ID)

When running online with an active internet connection, Ultralytics and Torchreid will automatically download the required weights upon initial startup.
