from __future__ import annotations

import os
import shutil
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2

from sms_notifier import SMSNotifier


def open_webcam(max_devices: int = 3) -> Tuple[Optional[cv2.VideoCapture], Optional[int]]:
    backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else None

    for device_index in range(max_devices):
        capture = cv2.VideoCapture(device_index, backend) if backend is not None else cv2.VideoCapture(device_index)
        if capture.isOpened():
            return capture, device_index
        capture.release()

    return None, None


class WebcamBackend:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.capture_worker: Optional[threading.Thread] = None
        self.inference_worker: Optional[threading.Thread] = None
        self.pipeline = None
        self.alert_player = None
        self.sms_notifier = SMSNotifier(enabled=True)

        self.frame_interval = 10
        self.clip_duration = 3
        self.target_fps = 15
        self.temp_audio_path: Optional[str] = None
        self.storage_root: Path = Path(__file__).resolve().parent / "clip_storage"
        self.pending_dir: Optional[Path] = None
        self.unsafe_dir: Optional[Path] = None
        self.safe_dir: Optional[Path] = None
        self.pending_clips: Deque[str] = deque(maxlen=8)
        self.inference_event = threading.Event()

        self.is_running = False
        self.is_inference_running = False
        self.latest_frame_bgr = None
        self.current_clip_started_at = ""
        self.last_result_at = ""
        self.active_device_index: Optional[int] = None
        self.buffer_size = 0
        self.fps = 0.0
        self.error_message = ""
        self.submitted_clip_count = 0
        self.processed_clip_count = 0
        self.safe_clip_count = 0
        self.unsafe_clip_count = 0
        self.sms_sent_count = 0
        self.latest_result: Optional[Dict] = None
        self.result_counter = 0
        self.analysis_logs: List[Dict] = []

    def start(
        self,
        pipeline,
        frame_interval: int,
        clip_duration: int,
        target_fps: int,
        alert_player,
        temp_audio_path: Optional[str] = None,
        enable_sms: bool = True,
    ) -> None:
        if self.is_running:
            return

        self.pipeline = pipeline
        self.alert_player = alert_player
        self.frame_interval = frame_interval
        self.clip_duration = clip_duration
        self.target_fps = target_fps
        self.temp_audio_path = temp_audio_path
        self.sms_notifier = SMSNotifier(enabled=enable_sms)
        self.stop_event = threading.Event()
        self.inference_event = threading.Event()

        self.pending_dir = self.storage_root / "pending"
        self.unsafe_dir = self.storage_root / "unsafe"
        self.safe_dir = self.storage_root / "processed_safe"
        for directory in (self.pending_dir, self.unsafe_dir, self.safe_dir):
            directory.mkdir(parents=True, exist_ok=True)
        for old_clip in self.pending_dir.glob("*.mp4"):
            old_clip.unlink(missing_ok=True)

        with self.lock:
            self.pending_clips.clear()
            self.is_running = True
            self.is_inference_running = False
            self.latest_frame_bgr = None
            self.current_clip_started_at = ""
            self.last_result_at = ""
            self.active_device_index = None
            self.buffer_size = 0
            self.fps = 0.0
            self.error_message = ""
            self.submitted_clip_count = 0
            self.processed_clip_count = 0
            self.safe_clip_count = 0
            self.unsafe_clip_count = 0
            self.sms_sent_count = 0
            self.latest_result = None
            self.result_counter = 0
            self.analysis_logs = []

        self.capture_worker = threading.Thread(target=self._run_capture_loop, daemon=True)
        self.inference_worker = threading.Thread(target=self._run_inference_loop, daemon=True)
        self.capture_worker.start()
        self.inference_worker.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.inference_event.set()

        if self.capture_worker is not None and self.capture_worker.is_alive():
            self.capture_worker.join(timeout=2)
        if self.inference_worker is not None and self.inference_worker.is_alive():
            self.inference_worker.join(timeout=2)

        with self.lock:
            self.is_running = False
            self.is_inference_running = False
            self.buffer_size = 0
            self.active_device_index = None

        self._cleanup_temp_audio()

    def snapshot(self) -> Dict:
        with self.lock:
            return {
                "is_running": self.is_running,
                "is_inference_running": self.is_inference_running,
                "latest_frame_bgr": None if self.latest_frame_bgr is None else self.latest_frame_bgr.copy(),
                "latest_result": None if self.latest_result is None else dict(self.latest_result),
                "result_counter": self.result_counter,
                "buffer_size": self.buffer_size,
                "fps": self.fps,
                "error_message": self.error_message,
                "last_result_at": self.last_result_at,
                "current_clip_started_at": self.current_clip_started_at,
                "active_device_index": self.active_device_index,
                "submitted_clip_count": self.submitted_clip_count,
                "processed_clip_count": self.processed_clip_count,
                "safe_clip_count": self.safe_clip_count,
                "unsafe_clip_count": self.unsafe_clip_count,
                "sms_sent_count": self.sms_sent_count,
                "pending_queue_size": len(self.pending_clips),
                "analysis_logs": [dict(log) for log in self.analysis_logs],
                "clip_directories": {
                    "pending": str(self.pending_dir) if self.pending_dir else "",
                    "unsafe": str(self.unsafe_dir) if self.unsafe_dir else "",
                    "safe": str(self.safe_dir) if self.safe_dir else "",
                },
            }

    def clear_logs(self) -> None:
        with self.lock:
            self.analysis_logs = []
            self.result_counter = 0
            self.latest_result = None

    def _run_capture_loop(self) -> None:
        capture, device_index = open_webcam()
        if capture is None:
            with self.lock:
                self.error_message = "No webcam found. Please connect a camera and try again."
                self.is_running = False
            self._cleanup_temp_audio()
            return

        with self.lock:
            self.active_device_index = device_index

        try:
            self._capture_frames(capture)
        finally:
            capture.release()
            self.inference_event.set()
            with self.lock:
                self.is_running = False
                self.buffer_size = 0
                self.active_device_index = None
            self._cleanup_temp_audio()

    def _capture_frames(self, capture: cv2.VideoCapture) -> None:
        self._configure_capture(capture)
        frame_count_for_clip = 0
        last_frame_time = None
        clip_started_at = time.time()
        writer = None
        current_clip_path = None
        frame_size = None

        while not self.stop_event.is_set():
            loop_started_at = time.time()
            success, frame = capture.read()
            if not success:
                with self.lock:
                    self.error_message = "Unable to read frames from the webcam."
                break

            now = time.time()
            if last_frame_time is None:
                instantaneous_fps = 0.0
            else:
                instantaneous_fps = 1.0 / max(now - last_frame_time, 1e-6)
            last_frame_time = now

            if writer is None:
                frame_size = (frame.shape[1], frame.shape[0])
                current_clip_path = self._build_pending_clip_path()
                writer = self._create_video_writer(current_clip_path, frame_size)
                clip_started_at = now
                frame_count_for_clip = 0
                with self.lock:
                    self.current_clip_started_at = datetime.now().strftime("%H:%M:%S")

            writer.write(frame)
            frame_count_for_clip += 1

            with self.lock:
                self.latest_frame_bgr = frame.copy()
                self.buffer_size = frame_count_for_clip
                if instantaneous_fps:
                    self.fps = instantaneous_fps if self.fps <= 0 else (0.8 * self.fps) + (0.2 * instantaneous_fps)

            enough_time = (now - clip_started_at) >= self.clip_duration
            enough_frames = frame_count_for_clip >= max(4, int(self.target_fps * max(self.clip_duration, 1) * 0.5))
            if enough_time and enough_frames and current_clip_path:
                writer.release()
                writer = None
                self._submit_clip(current_clip_path)
                current_clip_path = None
                frame_count_for_clip = 0
                with self.lock:
                    self.buffer_size = 0

            frame_delay = max(0.0, (1.0 / max(self.target_fps, 1)) - (time.time() - loop_started_at))
            if frame_delay:
                time.sleep(frame_delay)

        if writer is not None:
            writer.release()
            if current_clip_path and os.path.exists(current_clip_path):
                self._submit_clip(current_clip_path)

    def _submit_clip(self, clip_path: str) -> None:
        with self.lock:
            if len(self.pending_clips) == self.pending_clips.maxlen:
                dropped_path = self.pending_clips.popleft()
                if os.path.exists(dropped_path):
                    os.remove(dropped_path)
            self.pending_clips.append(clip_path)
            self.submitted_clip_count += 1
        self.inference_event.set()

    def _run_inference_loop(self) -> None:
        while not self.stop_event.is_set() or self.pending_clips:
            self.inference_event.wait(timeout=0.2)
            self.inference_event.clear()

            clip_path = None
            with self.lock:
                if self.pending_clips:
                    clip_path = self.pending_clips.popleft()

            if clip_path:
                self._process_clip(clip_path)

        with self.lock:
            self.is_inference_running = False

    def _process_clip(self, clip_path: str) -> None:
        with self.lock:
            self.is_inference_running = True

        try:
            result = self.pipeline.analyze_video_clip(
                video_path=clip_path,
                frame_interval=self.frame_interval,
                include_frame=True,
                max_samples=3,
            )
        except Exception as error:
            with self.lock:
                self.error_message = f"Webcam inference error: {error}"
            return

        result = dict(result)
        result["timestamp"] = datetime.now().strftime("%H:%M:%S")

        final_clip_path = clip_path
        if result.get("unsafe"):
            final_clip_path = self._move_clip_to_unsafe(clip_path)
            sms_sent = self._send_unsafe_sms(result, final_clip_path)
        else:
            self._store_or_delete_safe_clip(clip_path)
            sms_sent = False

        result["clip_path"] = final_clip_path
        result["sms_sent"] = sms_sent

        try:
            if self.alert_player is not None:
                self.alert_player.play(result["risk_level"])
        except RuntimeError:
            pass

        log_entry = {
            "timestamp": result["timestamp"],
            "caption": result.get("caption", ""),
            "risk_level": result.get("risk_level", "SAFE"),
            "sentiment": result.get("sentiment_label", "unknown"),
            "unsafe": "YES" if result.get("unsafe") else "NO",
            "sms_sent": "YES" if sms_sent else "NO",
            "clip_path": result["clip_path"],
            "reason": result.get("unsafe_reason", ""),
        }

        with self.lock:
            self.latest_result = result
            self.result_counter += 1
            self.processed_clip_count += 1
            self.last_result_at = result["timestamp"]
            if result.get("unsafe"):
                self.unsafe_clip_count += 1
            else:
                self.safe_clip_count += 1
            if sms_sent:
                self.sms_sent_count += 1
            self.analysis_logs.insert(0, log_entry)
            self.analysis_logs = self.analysis_logs[:100]
            self.is_inference_running = False

    def _build_pending_clip_path(self) -> str:
        assert self.pending_dir is not None
        clip_name = datetime.now().strftime("clip_%Y%m%d_%H%M%S_%f.mp4")
        return str(self.pending_dir / clip_name)

    def _create_video_writer(self, output_path: str, frame_size: Tuple[int, int]):
        codec = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, codec, float(max(self.target_fps, 1)), frame_size)
        if not writer.isOpened():
            raise RuntimeError("Unable to create clip writer for webcam capture.")
        return writer

    def _move_clip_to_unsafe(self, clip_path: str) -> str:
        assert self.unsafe_dir is not None
        target_path = str(self.unsafe_dir / Path(clip_path).name)
        shutil.move(clip_path, target_path)
        return target_path

    def _store_or_delete_safe_clip(self, clip_path: str) -> None:
        if os.path.exists(clip_path):
            os.remove(clip_path)

    def _send_unsafe_sms(self, result: Dict, clip_path: str) -> bool:
        message = (
            f"Unsafe surveillance event at {result.get('timestamp')}. "
            f"Risk={result.get('risk_level')}, sentiment={result.get('sentiment_label')}. "
            f"Caption: {result.get('caption', '')[:120]}. "
            f"Clip saved at {clip_path}."
        )
        return self.sms_notifier.send(message)

    def _configure_capture(self, capture: cv2.VideoCapture) -> None:
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        capture.set(cv2.CAP_PROP_FPS, float(self.target_fps))
        if hasattr(cv2, "CAP_PROP_FOURCC"):
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    def _cleanup_temp_audio(self) -> None:
        if self.temp_audio_path and os.path.exists(self.temp_audio_path):
            os.remove(self.temp_audio_path)
        self.temp_audio_path = None
