# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np
from omegaconf import OmegaConf


def read_yaml(config_path: Union[str, Path]):
    return OmegaConf.load(config_path)


class ReadVideo:
    @classmethod
    def get_frame(
        cls, video_path: Union[str, Path], frame_idx: int
    ) -> Optional[np.ndarray]:
        with cls.video_capture(str(video_path)) as v:
            v.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = v.read()
            if not ret:
                return None
            return frame

    @classmethod
    def get_video_info(cls, video_path: Union[str, Path]) -> Dict[str, Any]:
        with cls.video_capture(str(video_path)) as v:
            fps = v.get(cv2.CAP_PROP_FPS)
            width = int(v.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(v.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(v.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0

            return {
                "fps": fps,
                "width": width,
                "height": height,
                "total_frames": total_frames,
                "duration": duration,
            }

    @staticmethod
    @contextmanager
    def video_capture(img):
        cap = cv2.VideoCapture(img)
        try:
            yield cap
        finally:
            cap.release()
