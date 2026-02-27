import streamlit as st
from datetime import datetime
import cv2
import numpy as np

def add_risk_border(image, risk_level, thickness=10):
    """
    Adds a colored border to the image based on risk level.
    """
    if risk_level == "SAFE":
        color = (0, 255, 0)       # Green
    elif risk_level == "SUSPICIOUS":
        color = (0, 255, 255)     # Yellow
    else:
        color = (0, 0, 255)       # Red

    bordered_image = cv2.copyMakeBorder(
        image,
        thickness, thickness, thickness, thickness,
        cv2.BORDER_CONSTANT,
        value=color
    )
    return bordered_image

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Real-Time CCTV Surveillance",
    layout="wide"
)

# -------------------------------------------------
# Session State Initialization
# -------------------------------------------------
if "running" not in st.session_state:
    st.session_state.running = False

if "logs" not in st.session_state:
    st.session_state.logs = []

if "current_caption" not in st.session_state:
    st.session_state.current_caption = ""

if "current_risk" not in st.session_state:
    st.session_state.current_risk = "SAFE"

if "current_explanation" not in st.session_state:
    st.session_state.current_explanation = ""

if "current_frame" not in st.session_state:
    st.session_state.current_frame = None


# -------------------------------------------------
# Header
# -------------------------------------------------
st.title("Real-Time CCTV Surveillance using Vision-Language Models")
st.caption("Automated Detection of Accidents and Suspicious Activities")

st.markdown("---")

# -------------------------------------------------
# Sidebar – Controls
# -------------------------------------------------
st.sidebar.header("Controls")

# ---- Video Source ----
st.sidebar.subheader("Input Video / Camera")

video_source = st.sidebar.selectbox(
    label="Video Input Source",
    options=["Upload Video (MP4)", "Webcam (Coming Soon)"],
    label_visibility="collapsed"
)

uploaded_video = None
if video_source == "Upload Video (MP4)":
    uploaded_video = st.sidebar.file_uploader(
        label="Upload MP4 Video",
        type=["mp4"],
        label_visibility="collapsed"
    )
else:
    st.sidebar.info("Webcam support will be added in future.")

# ---- Surveillance Control ----
st.sidebar.subheader("Surveillance Control")

col_start, col_pause = st.sidebar.columns(2)

with col_start:
    if st.button("Start Surveillance"):
        st.session_state.running = True

with col_pause:
    if st.button("Pause Surveillance"):
        st.session_state.running = False

# ---- Analysis Frequency ----
st.sidebar.subheader("Analysis Frequency")

frame_interval = st.sidebar.slider(
    label="Frame Interval",
    min_value=1,
    max_value=30,
    value=10,
    label_visibility="collapsed"
)

st.sidebar.caption("Lower value = higher accuracy, slower performance")

# ---- Clear Logs ----
if st.sidebar.button("Clear Logs"):
    st.session_state.logs.clear()

# -------------------------------------------------
# Main Layout
# -------------------------------------------------
left_col, right_col = st.columns([2, 1])

# -------------------------------------------------
# Live Frame View
# -------------------------------------------------
with left_col:
    st.subheader("Live Frame View")

    if st.session_state.current_frame is not None:
        frame = st.session_state.current_frame

    # Ensure frame is numpy array (OpenCV format)
        if not isinstance(frame, np.ndarray):
            frame = np.array(frame)

    # Add risk-based border
        bordered_frame = add_risk_border(
            frame,
             st.session_state.current_risk
        )

    # Convert BGR → RGB for Streamlit
        bordered_frame = cv2.cvtColor(bordered_frame, cv2.COLOR_BGR2RGB)

        st.image(
            bordered_frame,
            width="stretch"
         )

    else:
        st.info("No frame available. Start surveillance to begin.")

# -------------------------------------------------
# Analysis Panel
# -------------------------------------------------
with right_col:
    st.subheader("Generated Caption")

    st.markdown(
        st.session_state.current_caption
        if st.session_state.current_caption
        else "_Caption will appear here_"
    )

    st.subheader("Threat Assessment")

    risk = st.session_state.current_risk
    if risk == "SAFE":
        st.markdown("### 🟢 SAFE")
    elif risk == "SUSPICIOUS":
        st.markdown("### 🟡 SUSPICIOUS")
    else:
        st.markdown("### 🔴 DANGEROUS")

    st.subheader("Analysis Details")

    st.markdown(
        st.session_state.current_explanation
        if st.session_state.current_explanation
        else "_Explanation will appear here_"
    )

# -------------------------------------------------
# Detection Log
# -------------------------------------------------
st.markdown("---")
st.subheader("Detection Log")

if st.session_state.logs:
    st.table(st.session_state.logs)
else:
    st.info("No incidents logged yet.")

# -------------------------------------------------
# Backend Integration Placeholder
# -------------------------------------------------
"""
BACKEND INTEGRATION (NEXT STEP):

if st.session_state.running and uploaded_video is not None:
    - Read frame using OpenCV
    - Generate caption using BLIP
    - Run IncidentDetector
    - Update:
        st.session_state.current_frame
        st.session_state.current_caption
        st.session_state.current_risk
        st.session_state.current_explanation
    - Append to st.session_state.logs:
        {
            "Timestamp": datetime.now().strftime("%H:%M:%S"),
            "Risk Level": risk,
            "Caption Snippet": caption[:60] + "..."
        }
"""
