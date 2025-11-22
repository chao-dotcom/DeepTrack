"""Main UI entry point"""
import argparse
import subprocess
import sys


def main():
    """Main entry point for UI"""
    parser = argparse.ArgumentParser(description="People Tracking Web UI")
    parser.add_argument("--port", type=int, default=8501, help="Port to run UI on")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    
    args = parser.parse_args()
    
    # Check if streamlit is available
    try:
        import streamlit
    except ImportError:
        print("Error: streamlit is not installed. Install it with: pip install streamlit")
        sys.exit(1)
    
    # Create a simple streamlit app if app.py doesn't exist
    import os
    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    if not os.path.exists(app_path):
        print("Creating basic Streamlit app...")
        create_basic_app(app_path)
    
    print(f"Starting People Tracking UI on http://{args.host}:{args.port}")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", app_path,
        "--server.port", str(args.port),
        "--server.address", args.host
    ])


def create_basic_app(app_path: str):
    """Create a basic Streamlit app"""
    app_content = '''import streamlit as st
import cv2
import numpy as np

st.title("People Tracking System")
st.write("Welcome to the People Tracking System UI")

st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])

if uploaded_file is not None:
    st.video(uploaded_file)
    st.write("Video processing will be implemented here")
else:
    st.info("Please upload a video file to get started")
'''
    with open(app_path, 'w') as f:
        f.write(app_content)


if __name__ == "__main__":
    main()

