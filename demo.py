#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VideoSubFinder Python实现 - 演示脚本
展示如何使用Python实现的关键帧检测功能
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from subtitle_detector import DetectionConfig, SubtitleDetector


def create_sample_config():
    """创建示例配置文件"""
    config = DetectionConfig(
        # 基础检测参数
        sub_frame_length=6,  # 字幕最小持续帧数
        text_percent=0.3,  # 文本占比阈值
        min_text_length=0.022,  # 最小文本长度占比
        # 图像处理参数
        moderate_threshold=0.25,  # 限制阈值 (0.1-0.6)
        segment_width=8,  # 分段宽度
        segment_height=8,  # 分段高度
        min_segments_count=3,  # 最小分段数
        # 颜色过滤 (可选)
        use_color_filter=False,  # 是否使用颜色过滤
        color_ranges=None,  # 字幕颜色范围 (Lab格式)
        # 性能参数
        max_threads=0,  # 线程数 (0=自动)
        use_gpu=False,  # 是否使用GPU加速
    )

    return config


def demo_basic_detection(video_path: str, output_dir: str = "output"):
    """基础检测演示"""
    print("=" * 60)
    print("基础字幕检测演示")
    print("=" * 60)

    # 创建检测器
    config = create_sample_config()
    detector = SubtitleDetector(config)

    # 加载视频
    if not detector.load_video(video_path):
        print(f"❌ 无法加载视频: {video_path}")
        return False

    print("✅ 视频加载成功")
    print(f"   总帧数: {detector.frame_count}")
    print(f"   帧率: {detector.fps:.2f} FPS")

    # 检测字幕
    print("\n🔍 开始检测字幕...")
    detected_frames = detector.detect_subtitles()

    if detected_frames:
        print(f"✅ 检测到 {len(detected_frames)} 个字幕帧")

        # 保存结果
        detector.save_detected_frames(output_dir, "subtitle")
        print(f"💾 结果已保存到: {output_dir}")

        # 显示摘要
        summary = detector.get_detection_summary()
        print("\n📊 检测结果摘要:")
        for key, value in summary.items():
            print(f"   {key}: {value}")
    else:
        print("❌ 未检测到字幕帧")

    detector.close()
    return len(detected_frames) > 0


def demo_roi_detection(video_path: str, roi: tuple, output_dir: str = "output_roi"):
    """ROI区域检测演示"""
    print("=" * 60)
    print("ROI区域字幕检测演示")
    print("=" * 60)

    config = create_sample_config()
    detector = SubtitleDetector(config)

    if not detector.load_video(video_path):
        print(f"❌ 无法加载视频: {video_path}")
        return False

    print("✅ 视频加载成功")
    print(f"🎯 使用ROI区域: {roi}")

    # 检测指定区域的字幕
    detected_frames = detector.detect_subtitles(roi=roi)

    if detected_frames:
        print(f"✅ 在ROI区域检测到 {len(detected_frames)} 个字幕帧")
        detector.save_detected_frames(output_dir, "roi_subtitle")
        print(f"💾 结果已保存到: {output_dir}")
    else:
        print("❌ 在ROI区域未检测到字幕帧")

    detector.close()
    return len(detected_frames) > 0


def demo_color_filtering(video_path: str, output_dir: str = "output_color"):
    """颜色过滤检测演示"""
    print("=" * 60)
    print("颜色过滤字幕检测演示")
    print("=" * 60)

    # 配置颜色过滤 (白色字幕示例)
    config = DetectionConfig(
        sub_frame_length=6,
        text_percent=0.3,
        moderate_threshold=0.25,
        use_color_filter=True,
        color_ranges=[
            # Lab颜色范围: (L_min, L_max, a_min, a_max, b_min, b_max)
            (180, 255, 108, 148, 108, 148),  # 白色字幕范围
        ],
        max_threads=0,
        use_gpu=False,
    )

    detector = SubtitleDetector(config)

    if not detector.load_video(video_path):
        print(f"❌ 无法加载视频: {video_path}")
        return False

    print("✅ 视频加载成功")
    print("🎨 使用颜色过滤: 白色字幕")

    # 检测字幕
    detected_frames = detector.detect_subtitles()

    if detected_frames:
        print(f"✅ 使用颜色过滤检测到 {len(detected_frames)} 个字幕帧")
        detector.save_detected_frames(output_dir, "color_subtitle")
        print(f"💾 结果已保存到: {output_dir}")
    else:
        print("❌ 使用颜色过滤未检测到字幕帧")

    detector.close()
    return len(detected_frames) > 0


def demo_parameter_tuning(video_path: str):
    """参数调优演示"""
    print("=" * 60)
    print("参数调优演示")
    print("=" * 60)

    # 测试不同的参数组合
    test_configs = [
        ("默认参数", DetectionConfig()),
        (
            "高精度",
            DetectionConfig(
                moderate_threshold=0.5, text_percent=0.2, min_text_length=0.015
            ),
        ),
        (
            "快速模式",
            DetectionConfig(
                moderate_threshold=0.1, text_percent=0.4, sub_frame_length=3
            ),
        ),
    ]

    results = {}

    for name, config in test_configs:
        print(f"\n🔧 测试配置: {name}")

        detector = SubtitleDetector(config)
        if detector.load_video(video_path):
            detected_frames = detector.detect_subtitles()
            results[name] = len(detected_frames)
            print(f"   检测到 {len(detected_frames)} 个字幕帧")
        else:
            results[name] = 0
            print(f"   ❌ 无法加载视频")

        detector.close()

    # 显示比较结果
    print("\n📊 参数调优结果比较:")
    for name, count in results.items():
        print(f"   {name}: {count} 个字幕帧")


def create_test_video():
    """创建测试视频 (如果不存在)"""
    test_video_path = "test_video.mp4"

    if os.path.exists(test_video_path):
        print(f"✅ 测试视频已存在: {test_video_path}")
        return test_video_path

    print("🎬 创建测试视频...")

    # 创建简单的测试视频
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(test_video_path, fourcc, 30.0, (640, 480))

    # 生成100帧测试视频
    for i in range(100):
        # 创建黑色背景
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # 在某些帧添加白色文字
        if 20 <= i <= 40 or 60 <= i <= 80:
            # 添加白色文字
            cv2.putText(
                frame,
                "Test Subtitle",
                (200, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 255, 255),
                3,
            )

        out.write(frame)

    out.release()
    print("✅ 测试视频创建完成: " + test_video_path)
    return test_video_path


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="VideoSubFinder Python实现演示")
    parser.add_argument("--video", "-v", type=str, help="视频文件路径", default="2.mp4")
    parser.add_argument("--output", "-o", type=str, default="output", help="输出目录")
    parser.add_argument(
        "--demo",
        "-d",
        choices=["basic", "roi", "color", "tuning", "all"],
        default="all",
        help="演示类型",
    )

    args = parser.parse_args()

    # 确定视频路径
    if args.video:
        video_path = args.video
        if not os.path.exists(video_path):
            print(f"❌ 视频文件不存在: {video_path}")
            return
    else:
        video_path = create_test_video()

    print(f"🎬 使用视频: {video_path}")
    print(f"📁 输出目录: {args.output}")

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    # 运行演示
    if args.demo in ["basic", "all"]:
        demo_basic_detection(video_path, f"{args.output}/basic")

    if args.demo in ["roi", "all"]:
        # ROI区域检测 (视频下半部分)
        roi = (0, 240, 640, 240)  # (x, y, width, height)
        demo_roi_detection(video_path, roi, f"{args.output}/roi")

    if args.demo in ["color", "all"]:
        demo_color_filtering(video_path, f"{args.output}/color")

    if args.demo in ["tuning", "all"]:
        demo_parameter_tuning(video_path)

    print("\n🎉 演示完成!")


if __name__ == "__main__":
    main()
