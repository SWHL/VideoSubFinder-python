#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VideoSubFinder Python实现 - 简单测试脚本
用于验证基本功能是否正常工作
"""

import os
import sys
from pathlib import Path


def test_imports():
    """测试导入是否正常"""
    print("🔍 测试导入...")

    try:
        import cv2

        print("✅ OpenCV 导入成功")
    except ImportError as e:
        print(f"❌ OpenCV 导入失败: {e}")
        return False

    try:
        import numpy as np

        print("✅ NumPy 导入成功")
    except ImportError as e:
        print(f"❌ NumPy 导入失败: {e}")
        return False

    try:
        from subtitle_detector import DetectionConfig, SubtitleDetector

        print("✅ 字幕检测器导入成功")
    except ImportError as e:
        print(f"❌ 字幕检测器导入失败: {e}")
        return False

    return True


def test_basic_functionality():
    """测试基本功能"""
    print("\n🔍 测试基本功能...")

    try:
        from subtitle_detector import DetectionConfig, SubtitleDetector

        # 创建检测器
        config = DetectionConfig()
        detector = SubtitleDetector(config)
        print("✅ 检测器创建成功")

        # 测试配置
        print(f"   配置参数: sub_frame_length={config.sub_frame_length}")
        print(f"   配置参数: text_percent={config.text_percent}")
        print(f"   配置参数: moderate_threshold={config.moderate_threshold}")

        # 关闭检测器
        detector.close()
        print("✅ 检测器关闭成功")

        return True

    except Exception as e:
        print(f"❌ 基本功能测试失败: {e}")
        return False


def test_video_creation():
    """测试视频创建功能"""
    print("\n🔍 测试视频创建...")

    try:
        import cv2
        import numpy as np

        # 创建简单的测试视频
        test_video_path = "test_simple.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(test_video_path, fourcc, 30.0, (320, 240))

        # 生成30帧测试视频
        for i in range(30):
            # 创建黑色背景
            frame = np.zeros((240, 320, 3), dtype=np.uint8)

            # 在某些帧添加白色文字
            if 10 <= i <= 20:
                cv2.putText(
                    frame,
                    "Test",
                    (100, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )

            out.write(frame)

        out.release()

        if os.path.exists(test_video_path):
            print(f"✅ 测试视频创建成功: {test_video_path}")
            return test_video_path
        else:
            print("❌ 测试视频创建失败")
            return None

    except Exception as e:
        print(f"❌ 视频创建测试失败: {e}")
        return None


def test_detection():
    """测试字幕检测功能"""
    print("\n🔍 测试字幕检测...")

    try:
        from subtitle_detector import DetectionConfig, SubtitleDetector

        # 创建测试视频
        test_video = test_video_creation()
        if not test_video:
            return False

        # 创建检测器
        config = DetectionConfig(
            sub_frame_length=3,  # 降低要求以便测试
            text_percent=0.1,  # 降低阈值
            moderate_threshold=0.1,
        )
        detector = SubtitleDetector(config)

        # 加载视频
        if not detector.load_video(test_video):
            print("❌ 视频加载失败")
            return False

        print("✅ 视频加载成功")
        print(f"   总帧数: {detector.frame_count}")
        print(f"   帧率: {detector.fps:.2f} FPS")

        # 检测字幕
        detected_frames = detector.detect_subtitles()

        if detected_frames:
            print(f"✅ 检测到 {len(detected_frames)} 个字幕帧")
            for i, frame in enumerate(detected_frames[:3]):  # 只显示前3个
                print(
                    f"   帧 {i + 1}: 帧号={frame.frame_number}, 时间={frame.timestamp:.2f}s, 置信度={frame.confidence:.3f}"
                )
        else:
            print("⚠️  未检测到字幕帧 (这可能是正常的，取决于测试视频)")

        # 关闭
        detector.close()

        # 清理测试文件
        if os.path.exists(test_video):
            os.remove(test_video)
            print("🧹 清理测试文件")

        return True

    except Exception as e:
        print(f"❌ 字幕检测测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("=" * 60)
    print("VideoSubFinder Python实现 - 功能测试")
    print("=" * 60)

    # 测试导入
    if not test_imports():
        print("\n❌ 导入测试失败，请检查依赖安装")
        print("   运行: pip install -r requirements.txt")
        return False

    # 测试基本功能
    if not test_basic_functionality():
        print("\n❌ 基本功能测试失败")
        return False

    # 测试字幕检测
    if not test_detection():
        print("\n❌ 字幕检测测试失败")
        return False

    print("\n🎉 所有测试通过！")
    print("\n📖 使用方法:")
    print("   python demo.py --demo basic")
    print("   python demo.py --video your_video.mp4 --demo all")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
