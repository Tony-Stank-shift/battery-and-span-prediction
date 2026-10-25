"""项目全局配置：路径、常量、绘图风格。

设计原则
--------
1. 所有"魔法数字"集中在这里，但**只放数据/物理常量**，不放某个断言专用的期望值
   （期望值属于 sanity_check.py，见该文件顶部的 EXPECTED_* 块）。
2. 不在这里做 `sys.path` 注入之类的隐式副作用——除了 MPLCONFIGDIR 这一条，
   原因见下方注释。
"""

from __future__ import annotations

import os
from pathlib import Path

# ----------------------------------------------------------------------------
# 路径
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / "battery_capacity.csv"
FIG_DIR = ROOT / "outputs" / "figures"
TABLE_DIR = ROOT / "outputs" / "tables"
METRIC_DIR = ROOT / "outputs" / "metrics"

# ----------------------------------------------------------------------------
# matplotlib 缓存目录
# ----------------------------------------------------------------------------
# 为什么在 import 期就设置：matplotlib 一旦被 import 就会去读字体缓存，而默认位置
# 是 %USERPROFILE%\.matplotlib —— 在 DSH 的文件沙箱下那是只读区，写缓存会失败。
# 这个环境变量必须在 import matplotlib 之前生效，所以只能放在这里。
MPLCONFIG_DIR = ROOT / "outputs" / "_tmp" / "mplconfig"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))


def ensure_dirs() -> None:
    """确保图表/表格/指标目录存在（幂等）。"""
    for d in (FIG_DIR, TABLE_DIR, METRIC_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------
# 电池与寿命定义
# ----------------------------------------------------------------------------
BATTERIES: list[str] = ["B0005", "B0006", "B0007", "B0018"]

RATED_CAPACITY_AH = 2.0
"""额定容量。数据说明给的容量范围是 1.15–2.04 Ah，B0006 前 7 个循环略微超出额定值。"""

EOL_RATIO = 0.70
"""NASA 惯例：额定容量的 70% 视为寿命终点（End of Life）。"""

EOL_THRESHOLD_AH = RATED_CAPACITY_AH * EOL_RATIO
"""= 1.4 Ah。主结论使用的阈值；它的人为性由阈值敏感性分析（W7）显式处理。"""

EOL_COMPARISON_IS_INCLUSIVE = True
"""EOL 判定用 ≤ 还是 <。

采用 **≤（含边界）**：容量恰好等于阈值也算「到达 EOL」。
这条必须写死并写进报告，因为 D4 的整个右删失结论就悬在边界上——
B0007 最低 1.4005 Ah，若某次阈值扫描恰好取到 1.4005，≤ 与 < 会给出不同答案。
"""


EOL_SEARCH_MULTIPLE = 5
"""模型外推求寿命时，向未来最多找「观测长度 × 该倍数」个循环。

为什么放在 config 而不是某个模型文件里：M2（求根）与 M3（暴力扫描）都要在同一个
搜索区间里判断「有没有穿越阈值」，若各写一份，改一处不会同步——这正是 W5/W6 被记过的
「两个模块各写一份 window=11」。唯一出处。
（M1 的解析式直接解方程，用不到这个倍数；别把它当成三个模型共用。）
取 5 的理由：最长观测 168 循环 × 5 = 840，而 M1 在这份数据上最远的外推寿命
约 159 循环，留足余量且不会让暴力扫描变慢。"""


def is_at_eol(capacity, tau: float | None = None):
    """按上面的判定口径判断容量是否到达 EOL —— **唯一判定入口**。

    为什么要有这个函数：第一轮第三方审查发现，此前 sanity_check 和 preprocess
    各自硬编码 `<=`，而 `EOL_COMPARISON_IS_INCLUSIVE` 全仓库零消费者、是死代码。
    判定口径必须只有一个出处，否则改开关不会生效、报告里写的规则和代码对不上。

    参数
    ----
    capacity : 标量或数组
    tau : 阈值，默认取 EOL_THRESHOLD_AH。
          **W7 做阈值扫描时必须显式传 tau，不要改 EOL_RATIO** —— 改常量会让
          图表、决策表、数据体检三者的口径悄悄分叉。

    返回
    ----
    numpy 布尔数组（输入是标量时返回 0 维数组）
    """
    import numpy as np

    threshold = EOL_THRESHOLD_AH if tau is None else float(tau)
    cap = np.asarray(capacity, dtype=float)
    return (cap <= threshold) if EOL_COMPARISON_IS_INCLUSIVE else (cap < threshold)


def is_missing(value) -> bool:
    """统一判空：`pd.NA` / `np.nan` / `None` / `pd.NaT` 都要能认出来。

    踩过的坑：`value is not None` 对 `pd.NA` 是 True，而
    `isinstance(pd.NA, float)` 是 False，于是 `pd.NA` 一路溜到 `int()` 才炸。
    放在 config 里是因为 explore / individual 都要用，重复实现会被审查记成缺陷。
    """
    import pandas as pd

    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):  # 非标量
        return False

# ----------------------------------------------------------------------------
# 容量再生检测
# ----------------------------------------------------------------------------
REGEN_MIN_JUMP_AH = 0.05
"""相邻循环容量回升超过该阈值，记为一次容量再生事件。"""

TREND_WINDOW = 11
"""相对趋势（居中滚动中位数）的窗口。

W5 的 σ 与 W6 的残差都基于这个窗口，所以放在 config 里做唯一出处——
第三轮审查指出两个模块各写一份 window=11 属重复实现。
"""

EDGE_TRIM = TREND_WINDOW // 2
"""首尾各剔除多少个循环。

居中窗口在边界只能看到单侧数据，残差不可靠（实测边界 |r| 中位数约是内部的 20 倍）。
W5 与 W6 必须用同一个 trim，否则两处的波动尺度不可比。
"""

# ----------------------------------------------------------------------------
# 可复现性
# ----------------------------------------------------------------------------
RANDOM_SEED = 42
"""统一随机种子。消费方：preprocess（置换检验）、features（扰动臂）、
regeneration（置换相关）、models_ml（随机森林）。"""

# ----------------------------------------------------------------------------
# 绘图
# ----------------------------------------------------------------------------
COLORS: dict[str, str] = {
    "B0005": "#1f77b4",
    "B0006": "#d62728",
    "B0007": "#2ca02c",
    "B0018": "#ff7f0e",
}

FIG_DPI = 300

_FONT_CANDIDATES = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "DejaVu Sans",
]


def setup_matplotlib() -> None:
    """统一 matplotlib 风格；中文字体做**可用性探测**，找不到就显式告警。

    以前的写法是把字体名硬编码进 rcParams——在缺字体的机器上 matplotlib 只会
    静默回退，图里的中文全变方框，而且警告常被日志吞掉。这里改成主动探测。
    """
    import matplotlib

    matplotlib.use("Agg")  # 无界面后端，保证脚本可在任意环境跑出图
    from matplotlib import font_manager
    import matplotlib.pyplot as plt

    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((c for c in _FONT_CANDIDATES if c in available), None)
    if chosen is None:
        print(
            "[warn] 未找到任何中文字体（尝试过："
            + ", ".join(_FONT_CANDIDATES)
            + "），图中中文可能显示为方框。"
        )
        chosen = "DejaVu Sans"

    plt.rcParams.update(
        {
            "font.sans-serif": [chosen, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 110,
            "savefig.dpi": FIG_DPI,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
        }
    )
