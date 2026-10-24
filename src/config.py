"""项目全局配置：路径、常量、绘图风格。

所有魔法数字集中在这里，避免散落在各个脚本里。
"""

from __future__ import annotations

import sys
from pathlib import Path

# ----------------------------------------------------------------------------
# 路径
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / "battery_capacity.csv"
FIG_DIR = ROOT / "outputs" / "figures"
TABLE_DIR = ROOT / "outputs" / "tables"
METRIC_DIR = ROOT / "outputs" / "metrics"


def ensure_dirs() -> None:
    """确保输出目录存在（幂等）。"""
    for d in (FIG_DIR, TABLE_DIR, METRIC_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------
# 电池与寿命定义
# ----------------------------------------------------------------------------
BATTERIES: list[str] = ["B0005", "B0006", "B0007", "B0018"]

RATED_CAPACITY_AH = 2.0
"""额定容量。数据说明里容量范围 1.15–2.04 Ah，B0006 前几个循环略微超出额定值。"""

EOL_RATIO = 0.70
"""NASA 惯例：额定容量的 70% 视为寿命终点（End of Life）。"""

EOL_THRESHOLD_AH = RATED_CAPACITY_AH * EOL_RATIO
"""= 1.4 Ah。这是主结论使用的阈值，其人为性由阈值敏感性分析（W7）显式处理。"""

TAU_SCAN_MIN = 1.25
TAU_SCAN_MAX = 1.60
TAU_SCAN_STEP = 0.005
"""阈值敏感性扫描：τ ∈ [1.25, 1.60]，步长 0.005 Ah。"""

# ----------------------------------------------------------------------------
# 容量再生检测
# ----------------------------------------------------------------------------
REGEN_MIN_JUMP_AH = 0.05
"""相邻循环容量回升超过该阈值，记为一次容量再生事件。"""

# ----------------------------------------------------------------------------
# 可复现性
# ----------------------------------------------------------------------------
RANDOM_SEED = 42

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


def setup_matplotlib() -> None:
    """统一 matplotlib 风格，并解决中文与负号显示。"""
    import matplotlib

    matplotlib.use("Agg")  # 无界面后端，保证脚本可在任意环境跑出图
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
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


def add_src_to_path() -> None:
    """让 src/ 下的脚本能被直接运行（python src/xxx.py）。"""
    src = str(Path(__file__).resolve().parent)
    if src not in sys.path:
        sys.path.insert(0, src)
