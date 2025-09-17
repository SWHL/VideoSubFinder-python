#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VideoSubFinder Python实现 - 优化版字幕检测模块
基于原版VideoSubFinder的核心算法，用Python重新实现并优化字幕检测功能

主要优化：
1. 基于C++核心算法的分段分析文本检测
2. 多线程并行处理
3. 颜色空间转换和过滤
4. 边缘检测优化
5. 文本区域分析算法
"""

import logging
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_dilation, binary_erosion, label

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TextAlignment(Enum):
    """文本对齐方式"""

    ANY = "any"
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


@dataclass
class SubtitleFrame:
    """字幕帧信息"""

    frame_number: int
    timestamp: float
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float
    image: Optional[np.ndarray] = None
    text_lines: Optional[List[Tuple[int, int, int, int]]] = None  # 文本行边界框


@dataclass
class ColorRange:
    """颜色范围定义"""

    l_min: int
    l_max: int
    a_min: int
    a_max: int
    b_min: int
    b_max: int
    color_space: str = "Lab"  # Lab, BGR, HSV


@dataclass
class DetectionConfig:
    """检测配置参数 - 基于C++原版算法优化"""

    # 基础参数 (对应C++中的g_DL, g_tp, g_mtpl等)
    sub_frame_length: int = 6  # 字幕最小持续帧数 (g_DL)
    text_percent: float = 0.3  # 文本占比阈值 (g_tp)
    min_text_length: float = 0.022  # 最小文本长度占比 (g_mtpl)

    # 边缘检测参数 (对应g_veple, g_ilaple)
    vedge_points_line_error: float = 0.30  # 垂直边缘点线误差
    ila_points_line_error: float = 0.30  # ILA点线误差

    # 图像处理参数 (对应C++中的分段参数)
    moderate_threshold: float = 0.25  # 限制阈值 (g_mthr)
    segment_width: int = 8  # 分段宽度 (g_segw)
    segment_height: int = 8  # 分段高度 (g_segh)
    min_segments_count: int = 3  # 最小分段数 (g_msegc)
    min_sum_color_diff: int = 0  # 最小颜色差异和 (g_scd)

    # 文本分析参数
    min_points_number: int = 10  # 最小点数 (g_mpn)
    min_points_density: float = 0.1  # 最小点密度 (g_mpd)
    min_symbol_height: float = 0.02  # 最小符号高度占比 (g_msh)
    min_symbol_density: float = 0.1  # 最小符号密度 (g_msd)

    # 文本对齐
    text_alignment: TextAlignment = TextAlignment.CENTER

    # 颜色过滤参数
    use_color_filter: bool = False
    color_ranges: Optional[List[ColorRange]] = None
    outline_color_ranges: Optional[List[ColorRange]] = None

    # 高级处理选项
    use_isa_images: bool = True  # 使用ISA图像
    use_ila_images: bool = True  # 使用ILA图像
    replace_isa_by_filtered: bool = True  # 用过滤版本替换ISA

    # 性能参数
    max_threads: int = 0  # 0表示自动检测
    use_gpu: bool = False
    use_parallel_processing: bool = True

    def __post_init__(self):
        if self.max_threads == 0:
            self.max_threads = min(mp.cpu_count(), 8)


class SubtitleDetector:
    """字幕检测器 - 基于C++原版算法优化"""

    def __init__(self, config: Optional[DetectionConfig] = None):
        self.config = config or DetectionConfig()
        self.video_cap = None
        self.frame_count = 0
        self.fps = 0
        self.detected_frames: List[SubtitleFrame] = []

        # 缓存变量
        self._frame_cache: Dict[int, np.ndarray] = {}
        self._processed_cache: Dict[int, Dict[str, np.ndarray]] = {}

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

    def _get_transformed_image(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """获取变换后的图像 - 基于C++ GetTransformedImage函数"""
        h, w = frame.shape[:2]

        # 颜色空间转换
        if len(frame.shape) == 3:
            # BGR to YUV
            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            y, u, v = cv2.split(yuv)
        else:
            y = frame.copy()
            u = np.zeros_like(y)
            v = np.zeros_like(y)

        # 应用限制阈值
        if self.config.moderate_threshold != 1.0:
            y = cv2.convertScaleAbs(y, alpha=self.config.moderate_threshold)

        # 获取FF图像 (基于C++ GetImFF)
        im_ff = self._get_im_ff(y, u, v, w, h)

        # 获取NE图像 (基于C++ GetImNE)
        im_ne = self._get_im_ne(y, u, v, w, h)

        # 获取HE图像 (基于C++ GetImHE)
        im_he = self._get_im_he(y, u, v, w, h)

        return {"y": y, "u": u, "v": v, "ff": im_ff, "ne": im_ne, "he": im_he}

    def _get_im_ff(
        self, y: np.ndarray, u: np.ndarray, v: np.ndarray, w: int, h: int
    ) -> np.ndarray:
        """获取FF图像 - 基于C++ GetImFF函数"""
        # 计算颜色差异
        color_diff = np.abs(u.astype(np.float32) - 128) + np.abs(
            v.astype(np.float32) - 128
        )

        # 应用阈值
        threshold = self.config.moderate_threshold * 255
        ff = np.where(color_diff > threshold, 255, 0).astype(np.uint8)

        return ff

    def _get_im_ne(
        self, y: np.ndarray, u: np.ndarray, v: np.ndarray, w: int, h: int
    ) -> np.ndarray:
        """获取NE图像 - 基于C++ GetImNE函数"""
        # 计算边缘强度
        sobel_x = cv2.Sobel(y, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(y, cv2.CV_64F, 0, 1, ksize=3)

        # 计算梯度幅值
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

        # 应用阈值
        threshold = self.config.moderate_threshold * 255
        ne = np.where(gradient_magnitude > threshold, 255, 0).astype(np.uint8)

        return ne

    def _get_im_he(
        self, y: np.ndarray, u: np.ndarray, v: np.ndarray, w: int, h: int
    ) -> np.ndarray:
        """获取HE图像 - 基于C++ GetImHE函数"""
        # 计算水平边缘
        sobel_x = cv2.Sobel(y, cv2.CV_64F, 1, 0, ksize=3)
        sobel_x = np.absolute(sobel_x)

        # 应用阈值
        threshold = self.config.moderate_threshold * 255
        he = np.where(sobel_x > threshold, 255, 0).astype(np.uint8)

        return he

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

    def _analyze_image(
        self, im: np.ndarray, im_ila: Optional[np.ndarray] = None
    ) -> bool:
        """分析图像是否包含文本 - 基于C++ AnalyseImage函数"""
        h, w = im.shape
        segh = self.config.segment_height
        tp = self.config.text_percent
        mtl = int(self.config.min_text_length * w)

        # 应用ILA图像过滤
        if self.config.use_ila_images and im_ila is not None:
            im = cv2.bitwise_and(im, im_ila)

        # 计算分段数量
        n = h // segh
        if n == 0:
            return False

        # 分析每个分段
        max_pl = 0
        max_pl_idx = 0
        segment_lines = []

        for k in range(n):
            ia = k * w * segh
            lines = []
            pl = 0  # 像素计数

            # 扫描分段中的文本行
            for x in range(w):
                in_line = False
                line_start = x

                for y in range(segh):
                    i = ia + y * w + x
                    if i < len(im.flat) and im.flat[i] == 255:
                        pl += 1
                        if not in_line:
                            line_start = x
                            in_line = True
                    elif in_line:
                        # 行结束
                        lines.append((line_start, x - 1))
                        in_line = False

                if in_line and x == w - 1:
                    lines.append((line_start, x))

            segment_lines.append(lines)

            # 记录最大密度分段
            if pl > max_pl:
                max_pl = pl
                max_pl_idx = k

        if max_pl == 0:
            return False

        # 分析最大密度分段的文本长度
        k = max_pl_idx
        lines = segment_lines[k]

        if not lines:
            return False

        # 计算文本总长度
        total_len = sum(line[1] - line[0] + 1 for line in lines)

        # 应用文本对齐过滤
        if self.config.text_alignment != TextAlignment.ANY:
            filtered_lines = self._apply_text_alignment_filter(lines, w)
            if not filtered_lines:
                return False
            total_len = sum(line[1] - line[0] + 1 for line in filtered_lines)

        # 检查最小文本长度
        if total_len < mtl:
            return False

        # 计算文本密度
        if len(lines) > 0:
            text_span = lines[-1][1] - lines[0][0] + 1
            if text_span > 0:
                text_density = total_len / text_span
                if text_density >= tp:
                    return True

        return False

    def _apply_text_alignment_filter(
        self, lines: List[Tuple[int, int]], w: int
    ) -> List[Tuple[int, int]]:
        """应用文本对齐过滤 - 基于C++文本对齐逻辑"""
        if not lines:
            return lines

        filtered_lines = lines.copy()

        while len(filtered_lines) > 1:
            total_len = sum(line[1] - line[0] + 1 for line in filtered_lines)
            text_span = filtered_lines[-1][1] - filtered_lines[0][0] + 1

            if text_span > 0 and total_len / text_span >= self.config.text_percent:
                return filtered_lines

            if self.config.text_alignment == TextAlignment.CENTER:
                # 中心对齐：移除最不居中的行
                if len(filtered_lines) < 2:
                    break

                left_dist = abs(filtered_lines[0][0] + filtered_lines[-2][1] + 1 - w)
                right_dist = abs(filtered_lines[-1][1] + filtered_lines[1][0] + 1 - w)

                if left_dist <= right_dist:
                    filtered_lines.pop()  # 移除右端行
                else:
                    filtered_lines.pop(0)  # 移除左端行

            elif self.config.text_alignment == TextAlignment.LEFT:
                # 左对齐：移除右端行
                filtered_lines.pop()
            else:  # RIGHT
                # 右对齐：移除左端行
                filtered_lines.pop(0)

        return filtered_lines

    def _detect_subtitle_in_frame(
        self, frame: np.ndarray, frame_number: int
    ) -> Optional[SubtitleFrame]:
        """检测单帧中的字幕 - 基于C++核心算法"""
        # 获取变换后的图像
        transformed = self._get_transformed_image(frame)

        # 使用FF图像进行文本检测
        im_ff = transformed["ff"]

        # 应用颜色过滤
        if self.config.use_color_filter and self.config.color_ranges:
            im_ff = self._apply_color_filter(frame, im_ff)

        # 分析图像是否包含文本
        has_text = self._analyze_image(im_ff)

        if has_text:
            timestamp = frame_number / self.fps if self.fps > 0 else 0

            # 计算边界框
            bbox = self._calculate_text_bbox(im_ff)

            return SubtitleFrame(
                frame_number=frame_number,
                timestamp=timestamp,
                bbox=bbox,
                confidence=1.0,  # 基于C++算法，检测到即置信度为1
                image=frame.copy(),
            )

        return None

    def _calculate_text_bbox(self, im: np.ndarray) -> Tuple[int, int, int, int]:
        """计算文本边界框"""
        h, w = im.shape

        # 找到非零像素的位置
        y_coords, x_coords = np.where(im > 0)

        if len(y_coords) == 0:
            return (0, 0, 0, 0)

        min_x = np.min(x_coords)
        max_x = np.max(x_coords)
        min_y = np.min(y_coords)
        max_y = np.max(y_coords)

        return (min_x, min_y, max_x - min_x, max_y - min_y)

    def detect_subtitles(
        self,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> List[SubtitleFrame]:
        """检测视频中的字幕帧 - 基于C++ FastSearchSubtitles算法优化"""
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

        # 基于C++算法的连续帧检测
        if self.config.use_parallel_processing and (end_frame - start_frame) > 100:
            return self._detect_subtitles_parallel(start_frame, end_frame, roi)
        else:
            return self._detect_subtitles_sequential(start_frame, end_frame, roi)

    def _detect_subtitles_sequential(
        self, start_frame: int, end_frame: int, roi: Optional[Tuple[int, int, int, int]]
    ) -> List[SubtitleFrame]:
        """顺序检测字幕帧"""
        consecutive_frames = []
        frame_buffer = []  # 用于缓存连续帧

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
                frame_buffer.append(frame.copy())
            else:
                # 检查连续帧是否满足最小长度要求
                if len(consecutive_frames) >= self.config.sub_frame_length:
                    self.detected_frames.extend(consecutive_frames)
                    logger.info(
                        f"检测到字幕序列: 帧 {consecutive_frames[0].frame_number} 到 {consecutive_frames[-1].frame_number}"
                    )

                consecutive_frames = []
                frame_buffer = []

        # 处理最后的连续帧
        if len(consecutive_frames) >= self.config.sub_frame_length:
            self.detected_frames.extend(consecutive_frames)

        elapsed_time = time.time() - start_time
        logger.info(
            f"检测完成: 找到 {len(self.detected_frames)} 个字幕帧, 耗时 {elapsed_time:.2f} 秒"
        )

        return self.detected_frames

    def _detect_subtitles_parallel(
        self, start_frame: int, end_frame: int, roi: Optional[Tuple[int, int, int, int]]
    ) -> List[SubtitleFrame]:
        """并行检测字幕帧"""
        logger.info("使用并行处理模式")

        # 计算每个线程处理的帧数
        total_frames = end_frame - start_frame
        frames_per_thread = max(1, total_frames // self.config.max_threads)

        # 创建任务列表
        tasks = []
        for i in range(0, total_frames, frames_per_thread):
            thread_start = start_frame + i
            thread_end = min(start_frame + i + frames_per_thread, end_frame)
            tasks.append((thread_start, thread_end, roi))

        # 并行处理
        with ThreadPoolExecutor(max_workers=self.config.max_threads) as executor:
            futures = [
                executor.submit(self._process_frame_range, task) for task in tasks
            ]
            results = [future.result() for future in futures]

        # 合并结果
        all_frames = []
        for result in results:
            all_frames.extend(result)

        # 按帧号排序
        all_frames.sort(key=lambda x: x.frame_number)

        # 应用连续帧过滤
        self.detected_frames = self._filter_consecutive_frames(all_frames)

        logger.info(f"并行检测完成: 找到 {len(self.detected_frames)} 个字幕帧")
        return self.detected_frames

    def _process_frame_range(
        self, task: Tuple[int, int, Optional[Tuple[int, int, int, int]]]
    ) -> List[SubtitleFrame]:
        """处理帧范围 - 用于并行处理"""
        start, end, roi = task
        frames = []

        # 创建临时视频捕获对象
        temp_cap = cv2.VideoCapture(self.video_cap.get(cv2.CAP_PROP_FILENAME))
        temp_cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        for frame_number in range(start, end):
            ret, frame = temp_cap.read()
            if not ret:
                break

            # 应用ROI
            if roi:
                x, y, w, h = roi
                frame = frame[y : y + h, x : x + w]

            # 检测字幕
            subtitle_frame = self._detect_subtitle_in_frame(frame, frame_number)
            if subtitle_frame:
                frames.append(subtitle_frame)

        temp_cap.release()
        return frames

    def _filter_consecutive_frames(
        self, frames: List[SubtitleFrame]
    ) -> List[SubtitleFrame]:
        """过滤连续帧 - 基于C++算法"""
        if not frames:
            return []

        # 按帧号排序
        frames.sort(key=lambda x: x.frame_number)

        filtered_frames = []
        consecutive_group = [frames[0]]

        for i in range(1, len(frames)):
            # 检查是否连续
            if frames[i].frame_number - frames[i - 1].frame_number <= 1:
                consecutive_group.append(frames[i])
            else:
                # 检查连续组是否满足最小长度
                if len(consecutive_group) >= self.config.sub_frame_length:
                    filtered_frames.extend(consecutive_group)
                consecutive_group = [frames[i]]

        # 处理最后一组
        if len(consecutive_group) >= self.config.sub_frame_length:
            filtered_frames.extend(consecutive_group)

        return filtered_frames

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
    """主函数 - 演示优化后的用法"""
    # 创建优化的检测器配置
    config = DetectionConfig(
        # 基础参数
        sub_frame_length=6,
        text_percent=0.3,
        min_text_length=0.022,
        # 边缘检测参数
        vedge_points_line_error=0.30,
        ila_points_line_error=0.30,
        # 图像处理参数
        moderate_threshold=0.25,
        segment_width=8,
        segment_height=8,
        min_segments_count=3,
        # 文本分析参数
        min_points_number=10,
        min_points_density=0.1,
        min_symbol_height=0.02,
        min_symbol_density=0.1,
        # 文本对齐
        text_alignment=TextAlignment.CENTER,
        # 高级处理选项
        use_isa_images=True,
        use_ila_images=True,
        replace_isa_by_filtered=True,
        # 性能参数
        max_threads=4,
        use_gpu=False,
        use_parallel_processing=True,
    )

    detector = SubtitleDetector(config)

    # 示例用法
    video_path = "2.mp4"  # 使用项目中的测试视频

    if detector.load_video(video_path):
        logger.info("开始字幕检测...")

        # 检测字幕
        x, y, w, h = 0, 700, 19200, 100
        detected_frames = detector.detect_subtitles(roi=(x, y, w, h))

        # 保存结果
        detector.save_detected_frames("output", "subtitle")

        # 显示摘要
        summary = detector.get_detection_summary()
        print("\n=== 检测结果摘要 ===")
        for key, value in summary.items():
            print(f"  {key}: {value}")

        print("\n优化特性:")
        print("  - 基于C++原版算法实现")
        print("  - 支持多线程并行处理")
        print("  - 智能文本对齐过滤")
        print("  - 分段分析文本检测")
        print("  - 连续帧智能过滤")

    detector.close()


if __name__ == "__main__":
    main()
