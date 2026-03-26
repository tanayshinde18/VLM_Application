import inspect
import tempfile
import time
from datetime import datetime

import cv2
import numpy as np
import streamlit as st

from audio_alert import AlertPlayer
from main2 import SurveillancePipeline
from report_exporter import build_incident_pdf_report
from webcam_backend import WebcamBackend


def _supports_kwarg(func, kwarg_name):
    try:
        return kwarg_name in inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False


def _call_streamlit(func, *args, **kwargs):
    supported_kwargs = {
        key: value for key, value in kwargs.items() if _supports_kwarg(func, key)
    }
    return func(*args, **supported_kwargs)


if hasattr(st, "cache_resource"):
    cache_resource = st.cache_resource
else:
    def cache_resource(func):
        return st.cache(allow_output_mutation=True)(func)


def rerun_app():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


@cache_resource
def get_pipeline():
    return SurveillancePipeline()


@cache_resource
def get_webcam_controller():
    return WebcamBackend()


def create_alert_player(audio_enabled, audio_mode_label, uploaded_alert_audio, trigger_level):
    temp_alert_audio_path = None
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
        trigger_level=trigger_level,
    )
    return alert_player, temp_alert_audio_path


def render_live_frame(frame_slot, frame):
    if frame is None:
        frame_slot.info("No frame available. Start the webcam to begin live monitoring.")
        return

    if not isinstance(frame, np.ndarray):
        frame = np.array(frame)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if _supports_kwarg(frame_slot.image, "use_container_width"):
        frame_slot.image(rgb_frame, use_container_width=True)
    elif _supports_kwarg(frame_slot.image, "use_column_width"):
        frame_slot.image(rgb_frame, use_column_width=True)
    else:
        frame_slot.image(rgb_frame)


st.set_page_config(page_title="Live CCTV Monitoring", layout="wide")

st.title("Live CCTV Monitoring")
st.caption("Low-latency live view with background clip analysis, sentiment logging, unsafe clip retention, and optional SMS alerts.")

webcam_controller = get_webcam_controller()
webcam_snapshot = webcam_controller.snapshot()

st.sidebar.header("Controls")

clip_duration_seconds = st.sidebar.slider("Clip Duration (seconds)", 2, 3, 3)
target_webcam_fps = st.sidebar.slider("Target Webcam FPS", 5, 25, 15)
frame_interval = st.sidebar.slider("Analysis Frame Interval", 1, 30, 10)
st.sidebar.caption("Live video stays direct. Only saved clips are analyzed in the background.")

st.sidebar.subheader("SMS Alerts")
enable_sms = st.sidebar.checkbox("Send SMS for unsafe clips", value=True)
st.sidebar.caption("Configure `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM_NUMBER`, and `TWILIO_TO_NUMBER` in your environment.")

st.sidebar.subheader("Audio Alerts")
audio_enabled = st.sidebar.checkbox("Enable audio alerts", value=True)
audio_trigger_level = st.sidebar.selectbox(
    "Play alerts for",
    options=["Dangerous + Suspicious", "Dangerous only"],
)
audio_mode_label = st.sidebar.selectbox(
    "Alert sound mode",
    options=["Beep", "Custom WAV"],
)
uploaded_alert_audio = None
if audio_mode_label == "Custom WAV":
    uploaded_alert_audio = st.sidebar.file_uploader("Upload WAV alert sound", type=["wav"])

start_col, stop_col = st.sidebar.columns(2)
with start_col:
    start_webcam_clicked = st.button("Start Webcam")
with stop_col:
    stop_webcam_clicked = st.button("Stop Webcam")

if st.sidebar.button("Clear Backend Logs"):
    webcam_controller.clear_logs()
    webcam_snapshot = webcam_controller.snapshot()

if start_webcam_clicked and not webcam_snapshot["is_running"]:
    pipeline = get_pipeline()
    alert_player, temp_alert_audio_path = create_alert_player(
        audio_enabled=audio_enabled,
        audio_mode_label=audio_mode_label,
        uploaded_alert_audio=uploaded_alert_audio,
        trigger_level=audio_trigger_level,
    )
    webcam_controller.start(
        pipeline=pipeline,
        frame_interval=frame_interval,
        clip_duration=clip_duration_seconds,
        target_fps=target_webcam_fps,
        alert_player=alert_player,
        temp_audio_path=temp_alert_audio_path,
        enable_sms=enable_sms,
    )
    webcam_snapshot = webcam_controller.snapshot()

if stop_webcam_clicked and webcam_snapshot["is_running"]:
    webcam_controller.stop()
    webcam_snapshot = webcam_controller.snapshot()

left_col, right_col = st.columns([2.4, 1.2])

with left_col:
    st.subheader("Live Feed")
    live_frame_slot = st.empty()
    render_live_frame(live_frame_slot, webcam_snapshot["latest_frame_bgr"])

    status_items = [
        f"Status: {'Running' if webcam_snapshot['is_running'] else 'Stopped'}",
        f"FPS: {webcam_snapshot['fps']:.1f}",
        f"Frames in current clip: {webcam_snapshot['buffer_size']}",
        f"Queued clips: {webcam_snapshot['pending_queue_size']}",
        f"Processed clips: {webcam_snapshot['processed_clip_count']}",
    ]
    if webcam_snapshot["active_device_index"] is not None:
        status_items.append(f"Camera: {webcam_snapshot['active_device_index']}")
    st.caption(" | ".join(status_items))

with right_col:
    st.subheader("Backend Status")
    latest_result = webcam_snapshot["latest_result"]
    if latest_result:
        st.write(f"Last processed at: `{webcam_snapshot['last_result_at']}`")
        st.write(f"Latest risk: `{latest_result.get('risk_level', 'SAFE')}`")
        st.write(f"Latest sentiment: `{latest_result.get('sentiment_label', 'unknown')}`")
        st.write(f"Unsafe decision: `{'YES' if latest_result.get('unsafe') else 'NO'}`")
        st.write(f"SMS sent: `{'YES' if latest_result.get('sms_sent') else 'NO'}`")
    else:
        st.write("No clips processed yet.")

    st.markdown("**Counters**")
    st.write(f"Submitted clips: `{webcam_snapshot['submitted_clip_count']}`")
    st.write(f"Safe clips deleted: `{webcam_snapshot['safe_clip_count']}`")
    st.write(f"Unsafe clips stored: `{webcam_snapshot['unsafe_clip_count']}`")
    st.write(f"SMS sent count: `{webcam_snapshot['sms_sent_count']}`")

    clip_directories = webcam_snapshot["clip_directories"]
    st.markdown("**Clip Storage**")
    st.write(f"Pending folder: `{clip_directories['pending']}`")
    st.write(f"Unsafe folder: `{clip_directories['unsafe']}`")
    st.write(f"Safe temp folder: `{clip_directories['safe']}`")

st.markdown("---")
st.subheader("Analysis Logs")

analysis_logs = webcam_snapshot["analysis_logs"]
if analysis_logs:
    st.table(analysis_logs)
    try:
        pdf_bytes = build_incident_pdf_report(analysis_logs)
        report_filename = f"incident_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        _call_streamlit(
            st.download_button,
            label="Download PDF Report",
            data=pdf_bytes,
            file_name=report_filename,
            mime="application/pdf",
            use_container_width=True,
        )
    except Exception as error:
        st.error(f"Unable to generate PDF report: {error}")
else:
    st.info("Processed clip logs will appear here.")

if (
    webcam_snapshot["is_running"]
    and webcam_snapshot["submitted_clip_count"] > 0
    and webcam_snapshot["processed_clip_count"] == 0
):
    st.caption("The first log can take a little time while the vision model warms up and processes the first saved clip.")

if webcam_snapshot["error_message"]:
    st.error(webcam_snapshot["error_message"])

if webcam_controller.snapshot()["is_running"]:
    time.sleep(0.4)
    rerun_app()
