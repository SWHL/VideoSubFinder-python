#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VideoSubFinder Python优化版演示脚本
展示基于C++原版算法的优化功能
"""

import time
from pathlib import Path

from subtitle_detector import (
    ColorRange,
    DetectionConfig,
    SubtitleDetector,
    TextAlignment,
)


def demo_basic_detection():
    """基础检测演示"""
    print("=== 基础字幕检测演示 ===")

    # 创建基础配置
    config = DetectionConfig(
        sub_frame_length=6,
        text_percent=0.3,
        moderate_threshold=0.25,
        use_parallel_processing=True,
        max_threads=4,
    )

    detector = SubtitleDetector(config)

    # 加载视频
    video_path = "2.mp4"
    if not Path(video_path).exists():
        print(f"视频文件 {video_path} 不存在，请确保文件存在")
        return

    if detector.load_video(video_path):
        print(f"视频加载成功: {detector.frame_count} 帧")

        # 检测字幕
        start_time = time.time()
        detected_frames = detector.detect_subtitles()
        elapsed_time = time.time() - start_time

        print(f"检测完成: 找到 {len(detected_frames)} 个字幕帧")
        print(f"处理时间: {elapsed_time:.2f} 秒")

        # 保存结果
        detector.save_detected_frames("output", "basic_demo")

        # 显示摘要
        summary = detector.get_detection_summary()
        print("\n检测结果摘要:")
        for key, value in summary.items():
            print(f"  {key}: {value}")

    detector.close()


def demo_advanced_detection():
    """高级检测演示"""
    print("\n=== 高级字幕检测演示 ===")

    # 创建高级配置
    config = DetectionConfig(
        # 基础参数
        sub_frame_length=8,
        text_percent=0.25,
        min_text_length=0.02,
        # 边缘检测参数
        vedge_points_line_error=0.25,
        ila_points_line_error=0.25,
        # 图像处理参数
        moderate_threshold=0.3,
        segment_width=6,
        segment_height=6,
        min_segments_count=2,
        # 文本分析参数
        min_points_number=8,
        min_points_density=0.08,
        min_symbol_height=0.015,
        min_symbol_density=0.08,
        # 文本对齐
        text_alignment=TextAlignment.CENTER,
        # 高级处理选项
        use_isa_images=True,
        use_ila_images=True,
        replace_isa_by_filtered=True,
        # 性能参数
        max_threads=6,
        use_parallel_processing=True,
    )

    detector = SubtitleDetector(config)

    # 加载视频
    video_path = "2.mp4"
    if not Path(video_path).exists():
        print(f"视频文件 {video_path} 不存在，请确保文件存在")
        return

    if detector.load_video(video_path):
        print(f"视频加载成功: {detector.frame_count} 帧")

        # 检测字幕
        start_time = time.time()
        detected_frames = detector.detect_subtitles()
        elapsed_time = time.time() - start_time

        print(f"高级检测完成: 找到 {len(detected_frames)} 个字幕帧")
        print(f"处理时间: {elapsed_time:.2f} 秒")

        # 保存结果
        detector.save_detected_frames("output", "advanced_demo")

        # 显示摘要
        summary = detector.get_detection_summary()
        print("\n高级检测结果摘要:")
        for key, value in summary.items():
            print(f"  {key}: {value}")

    detector.close()


def demo_color_filtering():
    """颜色过滤演示"""
    print("\n=== 颜色过滤检测演示 ===")

    # 定义颜色范围 (Lab颜色空间)
    color_ranges = [
        ColorRange(
            l_min=0,
            l_max=255,
            a_min=120,
            a_max=140,  # 绿色范围
            b_min=120,
            b_max=140,
            color_space="Lab",
        ),
        ColorRange(
            l_min=0,
            l_max=255,
            a_min=140,
            a_max=160,  # 黄色范围
            b_min=100,
            b_max=120,
            color_space="Lab",
        ),
    ]

    # 创建带颜色过滤的配置
    config = DetectionConfig(
        sub_frame_length=6,
        text_percent=0.3,
        moderate_threshold=0.25,
        use_color_filter=True,
        color_ranges=color_ranges,
        use_parallel_processing=True,
        max_threads=4,
    )

    detector = SubtitleDetector(config)

    # 加载视频
    video_path = "2.mp4"
    if not Path(video_path).exists():
        print(f"视频文件 {video_path} 不存在，请确保文件存在")
        return

    if detector.load_video(video_path):
        print(f"视频加载成功: {detector.frame_count} 帧")

        # 检测字幕
        start_time = time.time()
        detected_frames = detector.detect_subtitles()
        elapsed_time = time.time() - start_time

        print(f"颜色过滤检测完成: 找到 {len(detected_frames)} 个字幕帧")
        print(f"处理时间: {elapsed_time:.2f} 秒")

        # 保存结果
        detector.save_detected_frames("output", "color_filtered_demo")

        # 显示摘要
        summary = detector.get_detection_summary()
        print("\n颜色过滤检测结果摘要:")
        for key, value in summary.items():
            print(f"  {key}: {value}")

    detector.close()


def demo_roi_detection():
    """ROI区域检测演示"""
    print("\n=== ROI区域检测演示 ===")

    # 创建配置
    config = DetectionConfig(
        sub_frame_length=6,
        text_percent=0.3,
        moderate_threshold=0.25,
        use_parallel_processing=True,
        max_threads=4,
    )

    detector = SubtitleDetector(config)

    # 加载视频
    video_path = "2.mp4"
    if not Path(video_path).exists():
        print(f"视频文件 {video_path} 不存在，请确保文件存在")
        return

    if detector.load_video(video_path):
        print(f"视频加载成功: {detector.frame_count} 帧")

        # 定义ROI区域 (字幕通常出现在视频下方)
        # 假设视频高度为720，字幕区域在下半部分
        roi = (0, 360, 1280, 360)  # (x, y, width, height)
        print(f"使用ROI区域: {roi}")

        # 检测字幕
        start_time = time.time()
        detected_frames = detector.detect_subtitles(roi=roi)
        elapsed_time = time.time() - start_time

        print(f"ROI检测完成: 找到 {len(detected_frames)} 个字幕帧")
        print(f"处理时间: {elapsed_time:.2f} 秒")

        # 保存结果
        detector.save_detected_frames("output", "roi_demo")

        # 显示摘要
        summary = detector.get_detection_summary()
        print("\nROI检测结果摘要:")
        for key, value in summary.items():
            print(f"  {key}: {value}")

    detector.close()


def main():
    """主函数"""
    print("VideoSubFinder Python优化版演示")
    print("=" * 50)

    # 创建输出目录
    Path("output").mkdir(exist_ok=True)

    try:
        # 基础检测演示
        demo_basic_detection()

        # 高级检测演示
        demo_advanced_detection()

        # 颜色过滤演示
        demo_color_filtering()

        # ROI区域检测演示
        demo_roi_detection()

        print("\n" + "=" * 50)
        print("所有演示完成！")
        print("检查 'output' 目录查看检测结果")

    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
