# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

import cv2
import numpy as np

# -----------------------------
# 全局参数（模拟 C++ 中的 g_XXX）
# -----------------------------
g_segw = 16  # 每个水平段的宽度（像素）
g_scd = 300  # 颜色变化阈值（Sum of Color Differences）
g_msegc = 2  # 连续满足阈值的最小段数
g_min_h = 0.05  # 最小行块高度占图像高度的比例
g_show_results = False  # 是否保存中间图像
debug_save_dir = "./DebugImages"  # 调试图像保存目录

os.makedirs(debug_save_dir, exist_ok=True)


def color_filtration(
    im_bgr: np.ndarray, w: int, h: int
) -> Tuple[List[int], List[int], int]:
    """
    Python 版 ColorFiltration
    从 BGR 图像中检测具有显著颜色变化的行区域

    Args:
        im_bgr: 输入图像 (H x W x 3)，dtype=np.uint8
        w: 宽度
        h: 高度

    Returns:
        LB: 起始行列表
        LE: 结束行列表
        N: 有效块数量
    """
    assert g_segw > 0, "g_segw must be > 0"
    assert im_bgr.shape == (h, w, 3), (
        f"im_bgr shape should be ({h}, {w}, 3), got {im_bgr.shape}"
    )

    scd = g_scd
    segw = g_segw
    msegc = g_msegc
    mx = (w - 1) // segw  # 每行最多分多少段

    # 用于标记每一行是否为“活跃行”
    line = np.zeros(h, dtype=np.int32)

    # -----------------------------
    # 并行处理每一行
    # -----------------------------
    def process_row(y: int) -> int:
        """处理第 y 行，返回是否为活跃行（1 或 0）"""
        row = im_bgr[y]
        cnt = 0  # 连续满足条件的段计数

        for nx in range(mx):
            ia = nx * segw
            mi = min(ia + segw, w - 1)  # 防止越界

            # 取第一个像素
            b0, g0, r0 = row[ia]
            dif = 0

            # 遍历该段内相邻像素的颜色差
            for i in range(ia + 1, mi + 1):
                b1, g1, r1 = row[i]

                rdif = abs(int(r1) - int(r0))
                gdif = abs(int(g1) - int(g0))
                bdif = abs(int(b1) - int(b0))

                dif += rdif + gdif + bdif

                r0, g0, b0 = r1, g1, b1

            if dif >= scd:
                cnt += 1
            else:
                cnt = 0

            if cnt >= msegc:
                return 1  # 标记为活跃行

        return 0

    # 使用线程池并行处理所有行（CPU-bound 但 I/O 小，ThreadPool 足够）
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_row, range(h)))

    line = np.array(results, dtype=np.int32)

    # -----------------------------
    # 合并连续活跃行为“行块” lb, le
    # -----------------------------
    lb = []
    le = []
    sbegin = True

    for y in range(h):
        if line[y] == 1:
            if sbegin:
                lb.append(y)
                sbegin = False
        else:
            if not sbegin:
                le.append(y - 1)
                sbegin = True

    # 处理最后一行仍为活跃的情况
    if not sbegin:
        le.append(h - 1)

    n = len(lb)
    if n == 0:
        return [], [], 0

    # 调试：保存初步检测结果
    if g_show_results:
        save_path = os.path.join(
            debug_save_dir, "ColorFiltration_01_ImBGRWithLinesInfo.png"
        )
        _save_bgr_with_lines(im_bgr.copy(), lb, le, n, save_path)

    # -----------------------------
    # 过滤：只保留足够高且间隔大的块
    # -----------------------------
    dd = 12
    bd = 2 * dd  # 块间最小间隔
    md = int(g_min_h * h)  # 最小高度阈值

    LB = []
    LE = []

    # 第一个块
    start = max(0, lb[0] - dd)
    LB.append(start)

    k = 0
    for i in range(n - 1):
        gap = lb[i + 1] - le[i] - 1  # 块间空白行数
        height = le[i] - LB[k] + 1  # 当前块高度（含padding）

        if gap >= bd:  # 块间距离足够
            if height >= md:  # 当前块足够高
                end = min(h - 1, le[i] + dd)
                LE.append(end)
                k += 1
                # 开始下一个块
                if i + 1 < n:
                    LB.append(max(0, lb[i + 1] - dd))
            else:
                # 不够高，跳过，但起始位置更新？（原C++逻辑）
                if i + 1 < n:
                    LB[k] = max(0, lb[i + 1] - dd)  # 注意：这里可能覆盖前一个起始位置
        # else: 距离不够，视为同一块？但原代码未处理，跳过

    # 处理最后一个块
    if n > 0:
        height = le[n - 1] - LB[k] + 1
        if height >= md:
            end = min(h - 1, le[n - 1] + dd)
            if k >= len(LE):
                LE.append(end)
            else:
                LE[k] = end
            k += 1

    N = k
    if N == 0:
        return [], [], 0

    # 截断多余部分
    LB = LB[:N]
    LE = LE[:N]

    # 调试：保存最终结果
    if g_show_results and N > 0:
        save_path = os.path.join(
            debug_save_dir, "ColorFiltration_02_ImBGRWithLinesInfoUpdated.png"
        )
        _save_bgr_with_lines(im_bgr.copy(), LB, LE, N, save_path)

    return LB, LE, N


def _save_bgr_with_lines(
    img: np.ndarray, lb: List[int], le: List[int], n: int, path: str
):
    """在图像上绘制行块区域并保存"""
    for i in range(n):
        cv2.rectangle(
            img, (0, lb[i]), (img.shape[1] - 1, le[i]), (0, 255, 0), thickness=2
        )
    cv2.imwrite(path, img)
    print(f"Saved: {path}")


if __name__ == "__main__":
    # 示例：读取图像并运行 color_filtration
    image_path = "tmp/1.jpg"
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    h, w = img.shape[:2]

    # 可选：开启调试模式
    g_show_results = True

    # 执行检测
    LB, LE, N = color_filtration(img, w=w, h=h)

    print(f"Detected {N} valid line blocks:")
    for i in range(N):
        print(f"  Block {i + 1}: [{LB[i]}, {LE[i]}] (height = {LE[i] - LB[i] + 1})")
