from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Iterator, List, Optional

import cv2

from caption_generator import CaptionGenerator
from frame_extractor import FrameExtractor
from incident_detector import IncidentDetector
from sentiment_analyzer import SentimentAnalyzer


@dataclass
class FrameAnalysis:
    timestamp: str
    frame_path: str
    caption: str
    risk_level: str
    risk_score: int
    explanation: str
    matched_signals: List[str]
    frame_bgr: Optional[object] = None

    def to_dict(self) -> Dict:
        return asdict(self)


class SurveillancePipeline:
    """
    Callable backend pipeline for video surveillance analysis.
    Reuses existing FrameExtractor, CaptionGenerator, and IncidentDetector.
    """

    def __init__(
        self,
        model_name: str = "Salesforce/blip-image-captioning-base",
        context: str = "",
    ) -> None:
        self.context = context
        self.caption_generator = CaptionGenerator(model_name=model_name)
        self.incident_detector = IncidentDetector()
        self.sentiment_analyzer = SentimentAnalyzer()

    def analyze_frame(self, frame_path: str, include_frame: bool = False) -> Dict:
        caption = self.caption_generator.generate_caption(
            frame_path,
            context=self.context,
        )
        incident = self.incident_detector.detect(caption)

        frame_bgr = None
        if include_frame:
            frame_bgr = cv2.imread(frame_path)

        result = FrameAnalysis(
            timestamp=datetime.now().strftime("%H:%M:%S"),
            frame_path=frame_path,
            caption=caption,
            risk_level=incident["risk_level"],
            risk_score=incident["risk_score"],
            explanation=incident["explanation"],
            matched_signals=incident.get("matched_signals", []),
            frame_bgr=frame_bgr,
        )
        return result.to_dict()

    def analyze_frame_array(self, frame_bgr, include_frame: bool = False) -> Dict:
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                temp_path = temp_file.name

            if not cv2.imwrite(temp_path, frame_bgr):
                raise RuntimeError("Unable to write temporary frame for analysis.")

            result = self.analyze_frame(temp_path, include_frame=False)
            if include_frame:
                result["frame_bgr"] = frame_bgr.copy()
            return result
        finally:
            if temp_path and os.path.isfile(temp_path):
                os.remove(temp_path)

    def analyze_clip_frames(
        self,
        frames_bgr: List[object],
        frame_interval: int = 10,
        include_frame: bool = False,
        max_samples: int = 1,
    ) -> List[Dict]:
        if not frames_bgr:
            return []

        step = max(1, frame_interval)
        candidate_indices = list(range(0, len(frames_bgr), step))
        if candidate_indices[-1] != len(frames_bgr) - 1:
            candidate_indices.append(len(frames_bgr) - 1)

        if max_samples and len(candidate_indices) > max_samples:
            sampled_positions = []
            last_position = len(candidate_indices) - 1
            for sample_idx in range(max_samples):
                position = round(sample_idx * last_position / max(1, max_samples - 1))
                sampled_positions.append(position)
            candidate_indices = [candidate_indices[position] for position in sorted(set(sampled_positions))]

        if max_samples == 1:
            middle_index = candidate_indices[len(candidate_indices) // 2]
            candidate_indices = [middle_index]

        sampled_frames = [frames_bgr[index] for index in candidate_indices]

        return [
            self.analyze_frame_array(frame, include_frame=include_frame)
            for frame in sampled_frames
        ]

    def analyze_video_clip(
        self,
        video_path: str,
        frame_interval: int = 10,
        include_frame: bool = False,
        max_samples: int = 3,
    ) -> Dict:
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open clip for analysis: {video_path}")

        frames_bgr: List[object] = []
        try:
            while True:
                success, frame = capture.read()
                if not success:
                    break
                frames_bgr.append(frame)
        finally:
            capture.release()

        if not frames_bgr:
            raise RuntimeError("Clip did not contain any readable frames.")

        clip_results = self.analyze_clip_frames(
            frames_bgr=frames_bgr,
            frame_interval=frame_interval,
            include_frame=include_frame,
            max_samples=max_samples,
        )
        if not clip_results:
            raise RuntimeError("No clip samples were produced for analysis.")

        best_result = max(
            clip_results,
            key=lambda result: (
                result.get("risk_score", 0),
                {"SAFE": 0, "SUSPICIOUS": 1, "DANGEROUS": 2}.get(
                    result.get("risk_level", "SAFE"), 0
                ),
            ),
        )

        sentiment_label, sentiment_score = self.sentiment_analyzer.analyze(
            best_result["caption"]
        )
        sentiment_label = sentiment_label.lower()
        is_unsafe = best_result.get("risk_level") != "SAFE" or sentiment_label == "negative"

        enriched_result = dict(best_result)
        enriched_result["timestamp"] = datetime.now().strftime("%H:%M:%S")
        enriched_result["clip_path"] = video_path
        enriched_result["sample_count"] = len(clip_results)
        enriched_result["sentiment_label"] = sentiment_label
        enriched_result["sentiment_score"] = sentiment_score
        enriched_result["unsafe"] = is_unsafe
        enriched_result["unsafe_reason"] = self._build_unsafe_reason(
            risk_level=best_result.get("risk_level", "SAFE"),
            sentiment_label=sentiment_label,
        )
        return enriched_result

    def iter_video_analysis(
        self,
        video_path: str,
        output_dir: str = "frames_interval",
        frame_interval: int = 10,
        cleanup: bool = False,
        include_frame: bool = False,
    ) -> Iterator[Dict]:
        extractor = FrameExtractor(
            video_path=video_path,
            output_dir=output_dir,
            frame_interval=frame_interval,
        )
        frame_paths = extractor.extract_frames()

        try:
            for frame_path in frame_paths:
                yield self.analyze_frame(frame_path, include_frame=include_frame)
        finally:
            if cleanup:
                self._cleanup_files(frame_paths)

    def analyze_video(
        self,
        video_path: str,
        output_dir: str = "frames_interval",
        frame_interval: int = 10,
        cleanup: bool = False,
        include_frame: bool = False,
    ) -> List[Dict]:
        return list(
            self.iter_video_analysis(
                video_path=video_path,
                output_dir=output_dir,
                frame_interval=frame_interval,
                cleanup=cleanup,
                include_frame=include_frame,
            )
        )

    @staticmethod
    def _cleanup_files(frame_paths: List[str]) -> None:
        for frame_path in frame_paths:
            if os.path.isfile(frame_path):
                os.remove(frame_path)

    @staticmethod
    def _build_unsafe_reason(risk_level: str, sentiment_label: str) -> str:
        reasons = []
        if risk_level != "SAFE":
            reasons.append(f"risk level {risk_level}")
        if sentiment_label == "negative":
            reasons.append("negative sentiment")
        if not reasons:
            return "No unsafe indicators detected."
        return "Marked unsafe due to " + " and ".join(reasons) + "."
