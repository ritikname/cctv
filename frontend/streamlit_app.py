import streamlit as st
import os
import sys
import subprocess
import threading
import queue

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="Multi-Camera Person Tracking", layout="wide")

st.title("Retail CCTV Analytics - Global Identity Tracking")

# Simple state
if 'cameras' not in st.session_state:
    st.session_state.cameras = []
if 'processing' not in st.session_state:
    st.session_state.processing = False

with st.sidebar:
    st.header("Controls")
    new_cam_name = st.text_input("Camera Name (e.g., entrance)")
    new_cam_file = st.text_input("Video File Path")
    
    if st.button("Add Camera"):
        if new_cam_name and new_cam_file:
            st.session_state.cameras.append({
                "id": new_cam_name,
                "path": new_cam_file
            })
            st.success(f"Added {new_cam_name}")
            
    st.markdown("---")
    if st.button("Start Processing" if not st.session_state.processing else "Stop"):
        st.session_state.processing = not st.session_state.processing
        if st.session_state.processing:
            st.success("Processing started (Check console for output and OpenCV windows)")
            # In a real app we'd stream back frames. For MVP, we'll run the script.
            paths = [c['path'] for c in st.session_state.cameras]
            if paths:
                # Run the main.py script
                cmd = f"python main.py --videos {' '.join(paths)}"
                subprocess.Popen(cmd, shell=True)
            else:
                st.error("Add at least one camera first.")
                st.session_state.processing = False

st.header("Camera Configuration")
if not st.session_state.cameras:
    st.info("No cameras added yet.")
else:
    for i, cam in enumerate(st.session_state.cameras):
        st.write(f"**Camera {i+1}**: {cam['id']} -> `{cam['path']}`")

st.markdown("---")
st.subheader("Statistics (Placeholder for Streamlit live updates)")
col1, col2, col3 = st.columns(3)
col1.metric("Active People", "0")
col2.metric("Unique People", "0")
col3.metric("Temporarily Lost", "0")

st.info("Note: The actual processing will open OpenCV windows to display the feeds in real-time, as real-time multi-video streaming to Streamlit requires more complex WebRTC setups.")
