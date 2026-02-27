import os
import tempfile
from datetime import datetime

import cv2
import numpy as np
import streamlit as st

from audio_alert import AlertPlayer
from main2 import SurveillancePipeline
from report_exporter import build_incident_pdf_report


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


@st.cache_resource
def get_pipeline():
    return SurveillancePipeline()


def estimate_total_samples(video_path, frame_interval):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if frame_count <= 0:
        return None
    return max(1, (frame_count + frame_interval - 1) // frame_interval)


def render_live_frame(frame_slot, frame, risk_level):
    if frame is None:
        frame_slot.info("No frame available. Start surveillance to begin.")
        return

    if not isinstance(frame, np.ndarray):
        frame = np.array(frame)

    bordered_frame = add_risk_border(frame, risk_level)
    bordered_frame = cv2.cvtColor(bordered_frame, cv2.COLOR_BGR2RGB)
    frame_slot.image(bordered_frame, width="stretch")


def render_analysis_panel(caption_slot, risk_slot, explanation_slot, caption, risk, explanation):
    caption_slot.markdown(caption if caption else "_Caption will appear here_")

    if risk == "SAFE":
        risk_slot.markdown("### 🟢 SAFE")
    elif risk == "SUSPICIOUS":
        risk_slot.markdown("### 🟡 SUSPICIOUS")
    else:
        risk_slot.markdown("### 🔴 DANGEROUS")

    explanation_slot.markdown(explanation if explanation else "_Explanation will appear here_")


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

if "is_processing" not in st.session_state:
    st.session_state.is_processing = False


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

# ---- Audio Alerts ----
st.sidebar.subheader("Audio Alerts")
audio_enabled = st.sidebar.checkbox("Enable audio alerts", value=True)
audio_trigger_level = st.sidebar.selectbox(
    "Play alerts for",
    options=["Dangerous + Suspicious", "Dangerous only"]
)
audio_mode_label = st.sidebar.selectbox(
    "Alert sound mode",
    options=["Beep", "Custom WAV"]
)
uploaded_alert_audio = None
if audio_mode_label == "Custom WAV":
    uploaded_alert_audio = st.sidebar.file_uploader(
        "Upload WAV alert sound",
        type=["wav"]
    )

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
    live_frame_slot = st.empty()
    render_live_frame(
        live_frame_slot,
        st.session_state.current_frame,
        st.session_state.current_risk
    )

# -------------------------------------------------
# Analysis Panel
# -------------------------------------------------
with right_col:
    st.subheader("Generated Caption")
    caption_slot = st.empty()

    st.subheader("Threat Assessment")
    risk_slot = st.empty()

    st.subheader("Analysis Details")
    explanation_slot = st.empty()

    render_analysis_panel(
        caption_slot,
        risk_slot,
        explanation_slot,
        st.session_state.current_caption,
        st.session_state.current_risk,
        st.session_state.current_explanation
    )

# -------------------------------------------------
# Detection Log
# -------------------------------------------------
st.markdown("---")
st.subheader("Detection Log")

if st.session_state.logs:
    st.table(st.session_state.logs)
    try:
        pdf_bytes = build_incident_pdf_report(st.session_state.logs)
        report_filename = f"incident_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        st.download_button(
            label="Download PDF Report",
            data=pdf_bytes,
            file_name=report_filename,
            mime="application/pdf",
            use_container_width=True,
        )
    except Exception as error:
        st.error(f"Unable to generate PDF report: {error}")
else:
    st.info("No incidents logged yet.")

# -------------------------------------------------
# Backend Integration
# -------------------------------------------------
if st.session_state.running and uploaded_video is not None and not st.session_state.is_processing:
    st.session_state.is_processing = True
    temp_video_path = None
    temp_alert_audio_path = None

    try:
        pipeline = get_pipeline()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(uploaded_video.getbuffer())
            temp_video_path = temp_file.name

        audio_mode = "beep"
        if audio_mode_label == "Custom WAV" and uploaded_alert_audio is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as audio_file:
                audio_file.write(uploaded_alert_audio.getbuffer())
                temp_alert_audio_path = audio_file.name
            audio_mode = "custom_wav"

        alert_player = AlertPlayer(
            enabled=audio_enabled,
            mode=audio_mode,
            wav_path=temp_alert_audio_path,
            trigger_level=audio_trigger_level,
        )

        total_samples = estimate_total_samples(temp_video_path, frame_interval)
        processed_count = 0

        st.markdown("---")
        st.subheader("Processing Status")
        progress_text = st.empty()
        progress_bar = st.progress(0)

        with st.spinner("Running surveillance analysis..."):
            for result in pipeline.iter_video_analysis(
                video_path=temp_video_path,
                output_dir="frames_interval",
                frame_interval=frame_interval,
                cleanup=True,
                include_frame=True,
            ):
                processed_count += 1
                st.session_state.current_frame = result["frame_bgr"]
                st.session_state.current_caption = result["caption"]
                st.session_state.current_risk = result["risk_level"]
                st.session_state.current_explanation = result["explanation"]

                render_live_frame(
                    live_frame_slot,
                    st.session_state.current_frame,
                    st.session_state.current_risk
                )
                render_analysis_panel(
                    caption_slot,
                    risk_slot,
                    explanation_slot,
                    st.session_state.current_caption,
                    st.session_state.current_risk,
                    st.session_state.current_explanation
                )

                try:
                    alert_player.play(result["risk_level"])
                except RuntimeError:
                    pass

                if total_samples:
                    percent = min(1.0, processed_count / total_samples)
                    progress_text.markdown(
                        f"Processing frame sample **{processed_count}/{total_samples}**"
                    )
                    progress_bar.progress(percent)
                else:
                    progress_text.markdown(
                        f"Processing frame sample **{processed_count}**"
                    )

                caption_snippet = result["caption"][:60]
                if len(result["caption"]) > 60:
                    caption_snippet += "..."

                st.session_state.logs.append(
                    {
                        "Timestamp": result["timestamp"] or datetime.now().strftime("%H:%M:%S"),
                        "Risk Level": result["risk_level"],
                        "Caption Snippet": caption_snippet,
                    }
                )

        progress_bar.progress(1.0)
        progress_text.markdown(f"Processed **{processed_count}** frame samples.")
        st.session_state.running = False
        st.success("Surveillance analysis completed.")
        st.rerun()

    except Exception as error:
        st.session_state.running = False
        st.error(f"Backend error: {error}")

    finally:
        st.session_state.is_processing = False
        if temp_video_path and os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        if temp_alert_audio_path and os.path.exists(temp_alert_audio_path):
            os.remove(temp_alert_audio_path)

elif st.session_state.running and uploaded_video is None:
    st.warning("Please upload an MP4 file before starting surveillance.")
