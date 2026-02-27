from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Iterator, List, Optional

import cv2

from caption_generator import CaptionGenerator
from frame_extractor import FrameExtractor
from incident_detector import IncidentDetector


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
