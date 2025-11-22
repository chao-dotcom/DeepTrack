import streamlit as st
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

