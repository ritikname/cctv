import argparse
from app.video.processor import MultiCameraProcessor
import torch

def check_env():
    print("=== Environment Check ===")
    print("PyTorch version:", torch.__version__)
    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU Name:", torch.cuda.get_device_name(0))
        print("VRAM roughly available")
    else:
        print("Running on CPU. Expect slower processing.")
    print("=========================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Camera Person Tracking & Re-ID")
    parser.add_argument("--videos", nargs="+", help="Paths to video files. Multiple files act as multiple cameras.", required=True)
    args = parser.parse_args()

    check_env()
    
    processor = MultiCameraProcessor()
    
    for idx, video_path in enumerate(args.videos):
        cam_id = f"camera_{idx+1:02d}"
        print(f"Adding stream for {cam_id}: {video_path}")
        processor.add_camera_stream(cam_id, video_path)
        
    print("\nStarting processing...")
    processor.run()
