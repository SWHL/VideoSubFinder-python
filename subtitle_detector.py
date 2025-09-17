#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VideoSubFinder Python实现 - 关键帧检测模块
基于原版VideoSubFinder的核心算法，用Python重新实现字幕检测功能
"""

import logging
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class SubtitleFrame:
    """字幕帧信息"""

    frame_number: int
    timestamp: float
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float
    image: Optional[np.ndarray] = None


@dataclass
class DetectionConfig:
    """检测配置参数"""

    # 基础参数
    sub_frame_length: int = 6  # 字幕最小持续帧数
    text_percent: float = 0.3  # 文本占比阈值
    min_text_length: float = 0.022  # 最小文本长度占比

    # 图像处理参数
    moderate_threshold: float = 0.25  # 限制阈值
    segment_width: int = 8  # 分段宽度
    segment_height: int = 8  # 分段高度
    min_segments_count: int = 3  # 最小分段数

    # 颜色过滤参数
    use_color_filter: bool = False
    color_ranges: Optional[List[Tuple[int, int, int, int, int, int]]] = (
        None  # Lab颜色范围
    )
    outline_color_ranges: Optional[List[Tuple[int, int, int, int, int, int]]] = None

    # 性能参数
    max_threads: int = 0  # 0表示自动检测
    use_gpu: bool = False

    def __post_init__(self):
        if self.max_threads == 0:
            self.max_threads = min(mp.cpu_count(), 8)


class SubtitleDetector:
    """字幕检测器"""

    def __init__(self, config: Optional[DetectionConfig] = None):
        self.config = config or DetectionConfig()
        self.video_cap = None
        self.frame_count = 0
        self.fps = 0
        self.detected_frames: List[SubtitleFrame] = []

        # 初始化OpenCV
        if self.config.use_gpu:
            self._init_gpu()

    def _init_gpu(self):
        """初始化GPU支持"""
        try:
            # 检查CUDA支持
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                logger.info(f"检测到 {cv2.cuda.getCudaEnabledDeviceCount()} 个CUDA设备")
            else:
                logger.warning("未检测到CUDA设备，将使用CPU处理")
                self.config.use_gpu = False
        except Exception as e:
            logger.warning(f"GPU初始化失败: {e}")
            self.config.use_gpu = False

    def load_video(self, video_path: str) -> bool:
        """加载视频文件"""
        try:
            self.video_cap = cv2.VideoCapture(video_path)
            if not self.video_cap.isOpened():
                logger.error(f"无法打开视频文件: {video_path}")
                return False

            self.frame_count = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)

            logger.info(f"视频加载成功: {self.frame_count} 帧, {self.fps:.2f} FPS")
            return True

        except Exception as e:
            logger.error(f"加载视频失败: {e}")
            return False

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """预处理视频帧"""
        # 转换为灰度图
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # 应用对比度和伽马调整
        if self.config.moderate_threshold != 1.0:
            gray = cv2.convertScaleAbs(gray, alpha=self.config.moderate_threshold)

        return gray

    def _detect_edges(self, gray: np.ndarray) -> np.ndarray:
        """边缘检测 - 基于Sobel算子"""
        # 垂直边缘检测
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_x = np.absolute(sobel_x)
        sobel_x = np.uint8(sobel_x)

        # 水平边缘检测
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_y = np.absolute(sobel_y)
        sobel_y = np.uint8(sobel_y)

        # 对角线边缘检测 (45度和135度)
        sobel_45 = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
        sobel_45 = np.absolute(sobel_45)
        sobel_45 = np.uint8(sobel_45)

        sobel_135 = cv2.Sobel(gray, cv2.CV_64F, 1, -1, ksize=3)
        sobel_135 = np.absolute(sobel_135)
        sobel_135 = np.uint8(sobel_135)

        # 合并所有边缘
        edges = cv2.addWeighted(sobel_x, 0.25, sobel_y, 0.25, 0)
        edges = cv2.addWeighted(edges, 1.0, sobel_45, 0.25, 0)
        edges = cv2.addWeighted(edges, 1.0, sobel_135, 0.25, 0)

        return edges

    def _apply_color_filter(self, frame: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """应用颜色过滤"""
        if not self.config.use_color_filter or not self.config.color_ranges:
            return edges

        # 转换到Lab颜色空间
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # 创建颜色掩码
        mask = np.zeros(edges.shape, dtype=np.uint8)

        for color_range in self.config.color_ranges:
            l_min, l_max, a_min, a_max, b_min, b_max = color_range

            # 创建颜色范围掩码
            color_mask = (
                (l >= l_min)
                & (l <= l_max)
                & (a >= a_min)
                & (a <= a_max)
                & (b >= b_min)
                & (b <= b_max)
            )

            mask = cv2.bitwise_or(mask, color_mask.astype(np.uint8) * 255)

        # 应用掩码到边缘图像
        filtered_edges = cv2.bitwise_and(edges, mask)

        return filtered_edges

    def _analyze_text_regions(
        self, edges: np.ndarray
    ) -> Tuple[bool, float, Tuple[int, int, int, int]]:
        """分析文本区域"""
        h, w = edges.shape

        # 分段分析
        segment_h = self.config.segment_height
        segment_w = self.config.segment_width

        # 计算分段数量
        segments_y = h // segment_h
        segments_x = w // segment_w

        if segments_y == 0 or segments_x == 0:
            return False, 0.0, (0, 0, 0, 0)

        # 分析每个分段
        text_segments = 0
        total_segments = segments_y * segments_x

        # 找到文本区域边界
        min_x, min_y = w, h
        max_x, max_y = 0, 0

        for y in range(segments_y):
            for x in range(segments_x):
                # 提取分段
                y_start = y * segment_h
                y_end = min((y + 1) * segment_h, h)
                x_start = x * segment_w
                x_end = min((x + 1) * segment_w, w)

                segment = edges[y_start:y_end, x_start:x_end]

                # 计算分段中的边缘强度
                edge_strength = np.sum(segment > 0)
                segment_area = segment.size

                # 判断是否为文本分段
                if edge_strength > segment_area * 0.1:  # 边缘像素占比阈值
                    text_segments += 1

                    # 更新边界
                    min_x = min(min_x, x_start)
                    min_y = min(min_y, y_start)
                    max_x = max(max_x, x_end - 1)
                    max_y = max(max_y, y_end - 1)

        # 计算文本占比
        text_ratio = text_segments / total_segments if total_segments > 0 else 0

        # 判断是否包含文本
        has_text = (
            text_ratio >= self.config.text_percent
            and text_segments >= self.config.min_segments_count
            and (max_x - min_x) >= w * self.config.min_text_length
        )

        bbox = (
            (min_x, min_y, max_x - min_x, max_y - min_y) if has_text else (0, 0, 0, 0)
        )
        confidence = text_ratio

        return has_text, confidence, bbox

    def _detect_subtitle_in_frame(
        self, frame: np.ndarray, frame_number: int
    ) -> Optional[SubtitleFrame]:
        """检测单帧中的字幕"""
        try:
            # 预处理
            gray = self._preprocess_frame(frame)

            # 边缘检测
            edges = self._detect_edges(gray)

            # 应用颜色过滤
            if self.config.use_color_filter:
                edges = self._apply_color_filter(frame, edges)

            # 分析文本区域
            has_text, confidence, bbox = self._analyze_text_regions(edges)

            if has_text:
                timestamp = frame_number / self.fps if self.fps > 0 else 0

                return SubtitleFrame(
                    frame_number=frame_number,
                    timestamp=timestamp,
                    bbox=bbox,
                    confidence=confidence,
                    image=frame.copy(),
                )

            return None

        except Exception as e:
            logger.error(f"处理帧 {frame_number} 时出错: {e}")
            return None

    def detect_subtitles(
        self,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> List[SubtitleFrame]:
        """检测视频中的字幕帧"""
        if self.video_cap is None:
            logger.error("请先加载视频文件")
            return []

        if end_frame is None:
            end_frame = self.frame_count

        logger.info(f"开始检测字幕: 帧 {start_frame} 到 {end_frame}")

        # 设置ROI区域
        if roi:
            x, y, w, h = roi
            logger.info(f"使用ROI区域: ({x}, {y}, {w}, {h})")

        self.detected_frames = []
        consecutive_frames = []

        # 跳转到起始帧
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        start_time = time.time()

        for frame_number in range(start_frame, end_frame):
            ret, frame = self.video_cap.read()
            if not ret:
                break

            # 应用ROI
            if roi:
                x, y, w, h = roi
                frame = frame[y : y + h, x : x + w]

            # 检测字幕
            subtitle_frame = self._detect_subtitle_in_frame(frame, frame_number)

            if subtitle_frame:
                consecutive_frames.append(subtitle_frame)
            else:
                # 检查连续帧是否满足最小长度要求
                if len(consecutive_frames) >= self.config.sub_frame_length:
                    self.detected_frames.extend(consecutive_frames)
                    logger.info(
                        f"检测到字幕序列: 帧 {consecutive_frames[0].frame_number} 到 {consecutive_frames[-1].frame_number}"
                    )

                consecutive_frames = []

        # 处理最后的连续帧
        if len(consecutive_frames) >= self.config.sub_frame_length:
            self.detected_frames.extend(consecutive_frames)

        elapsed_time = time.time() - start_time
        logger.info(
            f"检测完成: 找到 {len(self.detected_frames)} 个字幕帧, 耗时 {elapsed_time:.2f} 秒"
        )

        return self.detected_frames

    def save_detected_frames(self, output_dir: str, prefix: str = "subtitle"):
        """保存检测到的字幕帧"""
        if not self.detected_frames:
            logger.warning("没有检测到字幕帧")
            return

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"保存 {len(self.detected_frames)} 个字幕帧到 {output_dir}")

        for i, frame_info in enumerate(self.detected_frames):
            if frame_info.image is not None:
                filename = f"{prefix}_{i:04d}_frame{frame_info.frame_number:06d}.jpg"
                filepath = output_path / filename
                cv2.imwrite(str(filepath), frame_info.image)

        # 保存检测结果信息
        info_file = output_path / f"{prefix}_info.txt"
        with open(info_file, "w", encoding="utf-8") as f:
            f.write("字幕帧检测结果\n")
            f.write("=" * 50 + "\n")
            for i, frame_info in enumerate(self.detected_frames):
                f.write(
                    f"帧 {i + 1}: 帧号={frame_info.frame_number}, "
                    f"时间={frame_info.timestamp:.3f}s, "
                    f"置信度={frame_info.confidence:.3f}, "
                    f"边界框={frame_info.bbox}\n"
                )

    def get_detection_summary(self) -> Dict[str, Any]:
        """获取检测结果摘要"""
        if not self.detected_frames:
            return {"total_frames": 0, "subtitle_sequences": 0}

        # 计算字幕序列数量
        sequences = 1
        for i in range(1, len(self.detected_frames)):
            if (
                self.detected_frames[i].frame_number
                - self.detected_frames[i - 1].frame_number
                > 1
            ):
                sequences += 1

        return {
            "total_frames": len(self.detected_frames),
            "subtitle_sequences": sequences,
            "first_frame": self.detected_frames[0].frame_number
            if self.detected_frames
            else 0,
            "last_frame": self.detected_frames[-1].frame_number
            if self.detected_frames
            else 0,
            "average_confidence": np.mean([f.confidence for f in self.detected_frames])
            if self.detected_frames
            else 0,
        }

    def close(self):
        """关闭视频文件"""
        if self.video_cap:
            self.video_cap.release()
            self.video_cap = None


def main():
    """主函数 - 演示用法"""
    # 创建检测器
    config = DetectionConfig(
        sub_frame_length=6,
        text_percent=0.3,
        moderate_threshold=0.25,
        use_color_filter=False,
    )

    detector = SubtitleDetector(config)

    # 示例用法
    video_path = "example_video.mp4"  # 替换为实际视频路径

    if detector.load_video(video_path):
        # 检测字幕
        detected_frames = detector.detect_subtitles()

        # 保存结果
        detector.save_detected_frames("output", "subtitle")

        # 显示摘要
        summary = detector.get_detection_summary()
        print("检测结果摘要:")
        for key, value in summary.items():
            print(f"  {key}: {value}")

    detector.close()


if __name__ == "__main__":
    main()
