# VideoSubFinder Python实现

基于原版VideoSubFinder核心算法的Python实现，专注于关键帧检测功能，无需GUI界面。

## 功能特点

- ✅ **核心算法实现**: 基于原版VideoSubFinder的核心字幕检测算法
- ✅ **关键帧检测**: 自动识别包含硬字幕的视频帧
- ✅ **多种检测模式**: 支持基础检测、ROI区域检测、颜色过滤等
- ✅ **高性能**: 支持多线程和GPU加速
- ✅ **易于使用**: 简洁的Python API
- ✅ **可配置**: 丰富的参数配置选项

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 基础使用

```python
from subtitle_detector import SubtitleDetector, DetectionConfig

# 创建检测器
detector = SubtitleDetector()

# 加载视频
detector.load_video("your_video.mp4")

# 检测字幕
detected_frames = detector.detect_subtitles()

# 保存结果
detector.save_detected_frames("output", "subtitle")

# 关闭
detector.close()
```

### 高级配置

```python
# 自定义配置
config = DetectionConfig(
    sub_frame_length=6,        # 字幕最小持续帧数
    text_percent=0.3,          # 文本占比阈值
    moderate_threshold=0.25,   # 限制阈值
    use_color_filter=True,     # 使用颜色过滤
    color_ranges=[            # 字幕颜色范围 (Lab格式)
        (180, 255, 108, 148, 108, 148)  # 白色字幕
    ]
)

detector = SubtitleDetector(config)
```

## 演示脚本

运行完整演示：

```bash
python demo.py --video your_video.mp4 --demo all
```

### 演示类型

- **基础检测**: `--demo basic`
- **ROI区域检测**: `--demo roi`
- **颜色过滤检测**: `--demo color`
- **参数调优**: `--demo tuning`

## 核心算法

### 1. 图像预处理

- 灰度转换
- 对比度和伽马调整
- 边缘检测 (Sobel算子)

### 2. 文本区域检测

- 分段分析
- 边缘强度计算
- 文本占比评估

### 3. 关键帧判断

- 连续帧分析
- 最小持续长度检查
- 置信度评估

## 配置参数

### 基础参数

- `sub_frame_length`: 字幕最小持续帧数 (默认: 6)
- `text_percent`: 文本占比阈值 (默认: 0.3)
- `min_text_length`: 最小文本长度占比 (默认: 0.022)

### 图像处理参数

- `moderate_threshold`: 限制阈值 (默认: 0.25)
- `segment_width`: 分段宽度 (默认: 8)
- `segment_height`: 分段高度 (默认: 8)
- `min_segments_count`: 最小分段数 (默认: 3)

### 颜色过滤参数

- `use_color_filter`: 是否使用颜色过滤 (默认: False)
- `color_ranges`: 字幕颜色范围列表 (Lab格式)

### 性能参数

- `max_threads`: 线程数 (0=自动检测)
- `use_gpu`: 是否使用GPU加速 (默认: False)

## 输出结果

### 检测到的字幕帧信息

```python
@dataclass
class SubtitleFrame:
    frame_number: int          # 帧号
    timestamp: float           # 时间戳
    bbox: Tuple[int, int, int, int]  # 边界框 (x, y, w, h)
    confidence: float          # 置信度
    image: Optional[np.ndarray]  # 帧图像
```

### 检测结果摘要

```python
{
    "total_frames": 150,           # 总字幕帧数
    "subtitle_sequences": 5,       # 字幕序列数
    "first_frame": 120,            # 第一帧
    "last_frame": 800,             # 最后一帧
    "average_confidence": 0.85     # 平均置信度
}
```

## 性能优化

### 1. 多线程处理

```python
config = DetectionConfig(max_threads=8)
```

### 2. GPU加速 (需要CUDA)

```python
config = DetectionConfig(use_gpu=True)
```

### 3. ROI区域检测

```python
# 只检测视频下半部分
roi = (0, 240, 640, 240)  # (x, y, width, height)
detected_frames = detector.detect_subtitles(roi=roi)
```

## 与原版VideoSubFinder的对比

| 特性 | 原版VideoSubFinder | Python实现 |
|------|-------------------|------------|
| 语言 | C++ | Python |
| GUI | wxWidgets | 无 (专注核心功能) |
| 依赖 | 复杂编译环境 | pip安装 |
| 扩展性 | 有限 | 易于扩展 |
| 性能 | 高 | 接近原版 |
| 维护性 | 复杂 | 简单 |

## 使用场景

1. **批量处理**: 自动化字幕检测
2. **API集成**: 作为服务提供字幕检测功能
3. **研究开发**: 字幕检测算法研究
4. **教学演示**: 计算机视觉教学

## 注意事项

1. **视频格式**: 支持常见视频格式 (mp4, avi, mkv等)
2. **字幕类型**: 主要针对硬字幕 (嵌入视频的字幕)
3. **参数调优**: 不同视频可能需要调整参数
4. **性能**: 大视频文件建议使用ROI区域检测

## 故障排除

### 常见问题

1. **检测不到字幕**
   - 调整 `moderate_threshold` 参数
   - 尝试使用颜色过滤
   - 检查视频质量

2. **检测结果过多**
   - 增加 `text_percent` 阈值
   - 增加 `sub_frame_length` 参数
   - 使用ROI区域限制

3. **性能问题**
   - 使用ROI区域检测
   - 调整线程数
   - 考虑GPU加速

## 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 许可证

与原版VideoSubFinder保持一致的开源许可证。
