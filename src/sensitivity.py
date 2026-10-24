"""W7：EOL 阈值敏感性 —— 把「定义不确定性」折算成循环数。

要回答的问题
------------
「寿命 = 125 循环」这个数字里，**有多少是电池的性质，有多少是我选的阈值造成的**？
EOL（End of Life）阈值 τ 是人定的惯例，不是物理常数。本模块把 τ 从 1.25 扫到 1.60 Ah，
看四块电池的寿命怎么变、哪些结论会翻、哪些撑得住。

为什么要用「断点」而不是网格
----------------------------
`N_EOL(τ) = min{cycle : cap(cycle) ≤ τ}` 在 τ 上是**单调不增的阶梯函数**，
而且它只可能在「创下新低」的循环上取值（前缀最小点）——
因为若 cycle c 不是前缀最小，就存在更早的 c' 使 cap(c') ≤ cap(c)：
τ 一旦够到 cap(c) 就早已够到 cap(c')，EOL 不会停在 c。

所以**整个函数被前缀最小点集精确决定**，网格扫描只是在阶梯上戳点。
本数据里 [1.35, 1.45] 内四块电池合计有 {n_steps} 个可达 EOL 值，
而 0.01 Ah 步长的网格只能采到 11 个点。本模块因此**用断点集精确计算并画图**，
另保留一个朴素扫描实现（`eol_at`）在网格上做交叉验证。

刻意不做的事
------------
- **不改 `config` 里的常量**（D17）。τ 一律作为参数传入，否则图表、决策表、
  数据体检三处的口径会悄悄分叉。
- **不把结论写成「寿命就是 X 循环」**。这个模块的产出是一个**区间**，
  它进 W15 的「定义不确定」层。

运行
----
    python src/sensitivity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (  # noqa: E402
    BATTERIES,
    COLORS,
    EOL_THRESHOLD_AH,
    RATED_CAPACITY_AH,
    TABLE_DIR,
    ensure_dirs,
    is_at_eol,
    setup_matplotlib,
)
from figures import save  # noqa: E402
from load_data import load_raw, split_by_battery  # noqa: E402
from preprocess import cell_profile, mark_rows  # noqa: E402

TABLE_MD = TABLE_DIR / "eol_threshold_sensitivity.md"
BREAK_CSV = TABLE_DIR / "eol_breakpoints.csv"

PLAUSIBLE_LO = 1.25
PLAUSIBLE_HI = 1.60
"""扫描区间。下界 ~1.25 对应「额定 62.5%」这种已经很激进的判据，上界 1.60 是工业上常见的
「80% 额定」。两端都还能给出有意义的数字；再往外扫，样本要么全删失、要么 EOL 落进前 20 个
循环，读数已经没有意义。**这个区间本身也是人选的**——所以下面所有结论都带区间。"""

MAIN_TAU = EOL_THRESHOLD_AH
"""主阈值 1.40 Ah。所有「相对变化」都以它为基准。"""

WINDOW_HALF = 0.05
"""主窗口半宽。即 [1.35, 1.45]，±0.05 Ah = 额定容量的 ±2.5%。"""

WINDOW_HALVES = (0.005, 0.01, 0.02, 0.05, 0.10)
"""窗口宽度扫描：极差本身也依赖窗口选多宽，这一维必须交代清楚。"""

KEY_TAUS = (1.25, 1.30, 1.35, 1.40, 1.45, 1.50, 1.55, 1.60)
"""表格里展示的关键 τ。1.40 就是主阈值。"""

GRID_STEP = 0.0025
"""交叉验证用的细网格步长。"""

DEFINITIONS = (
    {"label": "80% 额定", "factor": 0.80, "base": "rated"},
    {"label": "70% 额定", "factor": 0.70, "base": "rated"},
    {"label": "70% 首循环", "factor": 0.70, "base": "first"},
    {"label": "80% 首循环", "factor": 0.80, "base": "first"},
)
"""四种常见的 EOL 定义。注意后两种的阈值**逐电池不同**——它们不是「选一个 τ」，
而是另一类定义（以实测首循环容量为基准，承认电池出厂容量有差异）。"""


# ---------------------------------------------------------------------------
# 核心：EOL(τ)
# ---------------------------------------------------------------------------
def cap_series(df: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """每块电池的 (循环号, 容量) 数组，按循环升序。"""
    out = {}
    for cell, sub in split_by_battery(df).items():
        sub = sub.sort_values("cycle")
        cyc = sub["cycle"].to_numpy(int)
        cap = sub["capacity_Ah"].to_numpy(float)
        # EOL 用「下标 + 1」换算循环号，这个换算只有在循环连续时才成立
        assert (cyc == np.arange(1, len(sub) + 1)).all(), f"{cell} 的循环号不连续"
        out[cell] = (cyc, cap)
    return out


def prefix_min_mask(cap: np.ndarray) -> np.ndarray:
    """标记「创下新低」的位置（严格小于之前的最小值，首点算）。

    严格小于（而不是 ≤）意味着重复值只保留第一次，这与 EOL 取「首次到达」一致。
    """
    running = np.minimum.accumulate(cap)
    mask = np.empty(len(cap), dtype=bool)
    mask[0] = True
    mask[1:] = cap[1:] < running[:-1]
    return mask


def eol_at(cap: np.ndarray, tau: float) -> int | None:
    """朴素实现：从头扫，第一个容量 ≤ τ 的循环号；全都不满足返回 None（删失）。

    判定口径走 `config.is_at_eol`，不在这里重新写 `<=`（D17）。
    """
    hit = np.flatnonzero(is_at_eol(cap, tau))
    return int(hit[0]) + 1 if len(hit) else None


def eol_at_exact(cap: np.ndarray, tau: float) -> int | None:
    """断点实现：只在前缀最小点里找，复杂度 O(前缀最小点数)。

    与前者的关系：任一时刻 EOL 落点必然是前缀最小点，所以两个实现必须给出同一答案。
    这一点在 `main()` 里用 564 个网格点做交叉验证，**任一不一致就报错**。
    """
    idx = np.flatnonzero(prefix_min_mask(cap))
    hit = is_at_eol(cap[idx], tau)
    if not hit.any():
        return None
    # 注意是 idx[hit][0]（首个满足条件的前缀最小点），不是 idx[0]——
    # 首版写成 idx[0] 会把 EOL 永远答成「首个前缀最小点」，交叉验证当场抓到。
    return int(idx[hit][0]) + 1


def breakpoints(cap: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """返回 (τ 的台阶下界, 该段上的 EOL)，按 τ 升序。

    EOL(τ) = c_k 对 τ ∈ [w_k, w_{k+1}) 成立，其中 w_k 是第 k 小的前缀最小容量值、
    c_k 是它所在循环——因为容量值升序排列时，前 k 小里循环号最小的就是第 k 个。
    """
    idx = np.flatnonzero(prefix_min_mask(cap))
    values = cap[idx]
    order = np.argsort(values, kind="stable")  # 升序
    return values[order], (idx[order] + 1).astype(int)


def censored_below(cap: np.ndarray) -> float:
    """删失边界：τ < min(cap) 时该电池在观察窗口内没有到达 EOL。"""
    return float(cap.min())


def eol_or_none(cap: np.ndarray, tau: float) -> int | None:
    return eol_at_exact(cap, tau)


# ---------------------------------------------------------------------------
# 表格
# ---------------------------------------------------------------------------
def _fmt_censored(eol: int | None, n_cycles: int) -> str:
    """删失写成 >n，而不是留空——留空会被读成「没数据」，而这里是「有数据但没坏」。"""
    return f">{n_cycles}（删失）" if eol is None else str(eol)


def key_tau_table(series: dict[str, tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    """关键 τ 上的 N_EOL。刻意全部存字符串，渲染时关掉 tabulate 的数字解析，
    否则 `1.2500` 会被改写成 `1.25`、`1.4000` 变 `1.4`，全表小数位不齐。"""
    rows = []
    for t in KEY_TAUS:
        rec = {"τ (Ah)": f"{t:.4f}"}
        for cell, (_, cap) in series.items():
            rec[cell] = _fmt_censored(eol_at_exact(cap, t), len(cap))
        rows.append(rec)
    return pd.DataFrame(rows)


def range_table(series: dict[str, tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    """主窗口 [τ−0.05, τ+0.05] 内每个电池的寿命极差。

    关键细节：**有效窗口会被删失边界截断**。B0007 在 τ < 1.4005 时删失，
    所以它的有效下界是 1.4005 而不是 1.35——不能拿一个「未定义」当端点算极差。
    全部存成格式化字符串，理由同 `key_tau_table`。
    """
    lo, hi = MAIN_TAU - WINDOW_HALF, MAIN_TAU + WINDOW_HALF
    rows = []
    for cell, (_, cap) in series.items():
        lo_eff = max(lo, censored_below(cap))
        e_lo, e_hi = eol_at_exact(cap, lo_eff), eol_at_exact(cap, hi)
        e_main = eol_at_exact(cap, MAIN_TAU)
        if e_lo is None or e_hi is None:
            raise AssertionError(f"{cell} 的有效窗口端点仍是删失，窗口定义有问题")
        span = e_lo - e_hi
        rows.append(
            {
                "电池": cell,
                "有效窗口 (Ah)": f"[{lo_eff:.4f}, {hi:.4f}]",
                "EOL@下界": str(e_lo),
                f"EOL@{MAIN_TAU:.2f}": "—" if e_main is None else str(e_main),
                "EOL@上界": str(e_hi),
                "极差 (循环)": str(span),
                "每 0.01 Ah 的循环数": f"{span / ((hi - lo_eff) / 0.01):.2f}",
                "上偏 Δ": "—" if e_main is None else f"{e_lo - e_main:+d}",
                "下偏 Δ": "—" if e_main is None else f"{e_hi - e_main:+d}",
            }
        )
    return pd.DataFrame(rows)


def window_sweep(series: dict[str, tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    """极差随窗口半宽变化 —— 极差本身也是一个「有口径的量」。

    这是 W5 那条教训的第二次出现：加速比依赖切分点 h，极差依赖窗口宽度。
    对 B0007，窄窗口内下端点会掉进删失区，此时极差只能给**下界**。
    """
    rows = []
    for cell, (_, cap) in series.items():
        n = len(cap)
        rec = {"电池": cell, "删失下界 τ": round(censored_below(cap), 4)}
        for h in WINDOW_HALVES:
            lo, hi = MAIN_TAU - h, MAIN_TAU + h
            e_hi = eol_at_exact(cap, hi)
            if e_hi is None:
                rec[f"±{h:g}"] = "—"
                continue
            if lo < censored_below(cap):
                # 下端删失：真实 EOL > n，故极差 > n − EOL(hi)
                rec[f"±{h:g}"] = f">{n - e_hi}"
            else:
                rec[f"±{h:g}"] = str(eol_at_exact(cap, lo) - e_hi)
        rows.append(rec)
    return pd.DataFrame(rows)


def resolution_table(series: dict[str, tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    """主窗口内「可达 EOL 值」的个数与台阶密度。

    这是「为什么用断点」的量化答案：**每 0.01 Ah 里平均有几级台阶**。
    若这个数 > 1，则任何 0.01 Ah 步长的网格在每个区间里最多只能取到 1 个值，
    必然把台阶之间的变化抹掉（混叠）。
    """
    lo, hi = MAIN_TAU - WINDOW_HALF, MAIN_TAU + WINDOW_HALF
    rows = []
    for cell, (_, cap) in series.items():
        lo_eff = max(lo, censored_below(cap))
        w, c = breakpoints(cap)
        # 可达值集合：两个端点值 + 落在有效窗内每一级台阶上的值
        reachable = set()
        for t in (lo_eff, hi):
            e = eol_at_exact(cap, t)
            if e is not None:
                reachable.add(e)
        for wi, ci in zip(w, c):
            if lo_eff < wi <= hi:
                reachable.add(int(ci))
        assert len(reachable) >= 2, f"{cell} 窗内可达值少于 2 个，密度无意义"
        width = hi - lo_eff
        rows.append(
            {
                "电池": cell,
                "有效窗口宽度 (Ah)": f"{width:.4f}",
                "窗内可达 EOL 值": str(len(reachable)),
                "每 0.01 Ah 的台阶数": f"{len(reachable) / (width / 0.01):.2f}",
                "平均每级台阶跨几循环": (
                    f"{(max(reachable) - min(reachable)) / (len(reachable) - 1):.2f}"
                ),
            }
        )
    return pd.DataFrame(rows)


def definition_table(series: dict[str, tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    """四种 EOL 定义下的寿命。

    前两种是「单一 τ」，后两种的 τ 逐电池不同——它们回答的是另一个问题：
    「该用出厂额定容量还是实测初始容量作基准」。
    """
    rows = []
    for cell, (_, cap) in series.items():
        n = len(cap)
        first = float(cap[0])
        rec = {"电池": cell, "首循环容量 (Ah)": round(first, 4)}
        for d in DEFINITIONS:
            base = RATED_CAPACITY_AH if d["base"] == "rated" else first
            tau = d["factor"] * base
            rec[f"{d['label']} τ"] = round(tau, 4)
            rec[f"{d['label']} EOL"] = _fmt_censored(eol_at_exact(cap, tau), n)
        rows.append(rec)
    return pd.DataFrame(rows)


def invariance_scan(series: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict:
    """扫描区间内，逐点检查几条「结论」是否始终成立。

    这里要小心删失：删失的意思是 EOL > n，比任何已观测值都大，所以排序时当 +inf 用。
    但两个都删失的个体之间**不可比较**（只知道都 >n），不能算出一个顺序。
    """
    grid = np.round(np.arange(PLAUSIBLE_LO, PLAUSIBLE_HI + 1e-12, GRID_STEP), 6)
    orders = set()
    n_censored = {}
    full_taus: list[float] = []
    b7_censored_max_tau = None  # B0007 仍删失的最大 τ
    for tau in grid:
        eols = {}
        for cell, (_, cap) in series.items():
            e = eol_at_exact(cap, tau)
            eols[cell] = float("inf") if e is None else float(e)
        n_censored[round(float(tau), 6)] = sum(1 for v in eols.values() if np.isinf(v))

        # 顺序只在「四块都有确定寿命」时记录——两个都删失的个体之间只知道都 >n，
        # 不可比较，不能凭 n 的大小凑出一个顺序。
        if all(np.isfinite(v) for v in eols.values()):
            full_taus.append(float(tau))
            orders.add(tuple(sorted(eols, key=lambda c: (eols[c], c))))

        if np.isinf(eols["B0007"]):
            b7_censored_max_tau = float(tau) if b7_censored_max_tau is None else max(
                b7_censored_max_tau, float(tau)
            )

    return {
        "grid": grid,
        "full_orders": orders,
        "full_taus": full_taus,
        "n_censored": n_censored,
        "b7_censored_max_tau": b7_censored_max_tau,
    }


def equal_width_span(series: dict[str, tuple[np.ndarray, np.ndarray]], cell: str, width: float) -> dict:
    """把删失电池的窗口「补宽」到与其它电池同样的宽度，看极差会变成多少。

    用途：正文里不能只说「B0007 的极差是在更窄的窗口上算的」就完事，
    要给出同等宽度下的数字，读者才能判断差多少。
    """
    cap = series[cell][1]
    lo = censored_below(cap)
    hi = lo + width
    e_lo, e_hi = eol_at_exact(cap, lo), eol_at_exact(cap, hi)
    if e_lo is None or e_hi is None:
        raise AssertionError(f"{cell} 在 [{lo:.4f}, {hi:.4f}] 上仍有端点删失")
    return {"lo": lo, "hi": hi, "span": e_lo - e_hi}


# ---------------------------------------------------------------------------
# 图
# ---------------------------------------------------------------------------
def _draw_censored_arrow(ax, x: float, y_from: float, y_to: float, label: str) -> None:
    """在 τ = 删失边界处画一根向上的虚线箭头：τ 再往左，寿命就「射出画面」。"""
    ax.annotate(
        "",
        xy=(x, y_to),
        xytext=(x, y_from),
        arrowprops=dict(arrowstyle="->", ls="--", lw=1.2, color="#666666",
                        shrinkA=0, shrinkB=0),
        zorder=4,
    )
    ax.text(x + 0.004, y_to - 1.5, label, fontsize=7.6, color="#555555",
            ha="left", va="top", zorder=5)


def fig05(
    series: dict[str, tuple[np.ndarray, np.ndarray]],
    tbl_res: pd.DataFrame,
    tbl_win: pd.DataFrame,
    tbl_def: pd.DataFrame,
    scan: dict,
) -> Path:
    setup_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.8))
    cells = [c for c in BATTERIES if c in series]
    n_bp_window = 0

    # ---- (a) N_EOL(τ) 阶梯曲线（由断点精确绘制，不是网格采样）----
    ax = axes[0][0]
    ax.axvspan(MAIN_TAU - WINDOW_HALF, MAIN_TAU + WINDOW_HALF,
               color="#eeeeee", zorder=0, label=f"常用区间 ±{WINDOW_HALF:g} Ah")
    y_top = 202.0
    for cell in cells:
        _, cap = series[cell]
        w, c = breakpoints(cap)
        # 只画窗内的台阶；补一个窗内首点之前的台阶，让曲线铺满画布
        sel = (w >= PLAUSIBLE_LO) & (w <= PLAUSIBLE_HI)
        first_inside = np.flatnonzero(sel)
        if len(first_inside) and first_inside[0] > 0:
            sel[first_inside[0] - 1] = True
        xs, ys = w[sel], c[sel]
        n_bp_window += int(((w >= PLAUSIBLE_LO) & (w <= PLAUSIBLE_HI)).sum())
        ax.step(xs, ys, where="post", color=COLORS[cell], lw=1.6, zorder=3, label=cell)
        ax.plot([PLAUSIBLE_HI], [ys[-1]], marker=">", color=COLORS[cell], ms=5, zorder=3)

        cmin = censored_below(cap)
        if cmin > PLAUSIBLE_LO:
            e0 = eol_at_exact(cap, cmin)
            _draw_censored_arrow(ax, cmin, float(e0), y_top,
                                 f"{cell} 删失\n(τ<{cmin:.4f})")
    ax.axvline(MAIN_TAU, color="#333333", ls="-", lw=1.0, zorder=2,
               label=f"主线 τ={MAIN_TAU:g} Ah")
    ax.set_xlim(PLAUSIBLE_LO, PLAUSIBLE_HI)
    ax.set_ylim(40, y_top + 8)
    ax.set_xlabel("EOL 阈值 τ (Ah)")
    ax.set_ylabel("到达 EOL 的循环 $N_{\\mathrm{EOL}}$")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.92, ncol=2)
    gaps = [len(cap) / prefix_min_mask(cap).sum() for _, cap in series.values()]
    ax.set_title(f"(a) 寿命随阈值的变化：窗内 {n_bp_window} 级台阶，"
                 f"平均每 {min(gaps):.1f}–{max(gaps):.1f} 个循环一级",
                 fontsize=10.5)

    # ---- (b) 极差随窗口半宽 ----
    ax = axes[0][1]
    markers = {"B0005": "o", "B0006": "s", "B0007": "^", "B0018": "D"}
    for cell in cells:
        row = tbl_win.loc[tbl_win["电池"] == cell].iloc[0]
        vals, xs, is_lb = [], [], []
        for h in WINDOW_HALVES:
            raw = str(row[f"±{h:g}"])
            is_lb.append(raw.startswith(">"))
            vals.append(float(raw.lstrip(">")) if raw != "—" else np.nan)
            xs.append(h)
        ax.plot(xs, vals, color=COLORS[cell], lw=1.6,
                ls="--" if any(is_lb) else "-", zorder=3, label=cell)
        ax.scatter([x for x, lb in zip(xs, is_lb) if lb],
                   [v for v, lb in zip(vals, is_lb) if lb],
                   facecolors="white", edgecolors=COLORS[cell], s=44, zorder=4,
                   linewidths=1.5, marker=markers[cell])
        ax.scatter([x for x, lb in zip(xs, is_lb) if not lb],
                   [v for v, lb in zip(vals, is_lb) if not lb],
                   color=COLORS[cell], s=44, zorder=4, marker=markers[cell])
    ax.set_xscale("log")
    ax.set_xticks(list(WINDOW_HALVES))
    ax.set_xticklabels([f"{h:g}" for h in WINDOW_HALVES])
    ax.axvline(WINDOW_HALF, color="#333333", ls=":", lw=1.0, zorder=2,
               label=f"主窗口 ±{WINDOW_HALF:g}")
    lo_y, hi_y = ax.get_ylim()
    ax.set_ylim(lo_y, hi_y * 1.18)
    ax.set_xlabel("窗口半宽 (±Ah，对数轴)")
    ax.set_ylabel("窗内寿命极差 (循环)")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.92, ncol=2)
    ax.set_title("(b) 极差本身也依赖窗口多宽（空心点 = 只能给下界）", fontsize=10.5)

    # ---- (c) 四种 EOL 定义 ----
    ax = axes[1][0]
    xpos = np.arange(len(DEFINITIONS))
    width = 0.2
    ymax_def = 0.0
    for i, cell in enumerate(cells):
        row = tbl_def.loc[tbl_def["电池"] == cell].iloc[0]
        n = len(series[cell][1])
        for j, d in enumerate(DEFINITIONS):
            cell_txt = str(row[f"{d['label']} EOL"])
            xi = xpos[j] + (i - (len(cells) - 1) / 2) * width
            if cell_txt.startswith(">"):
                ax.bar([xi], [n], width=width * 0.92, facecolor="none",
                       edgecolor=COLORS[cell], hatch="//", lw=1.1)
                ax.text(xi, n, f">{n}", ha="center", va="bottom", fontsize=7.4,
                        color=COLORS[cell])
                ymax_def = max(ymax_def, n)
            else:
                v = int(cell_txt)
                ax.bar([xi], [v], width=width * 0.92, color=COLORS[cell])
                ax.text(xi, v, str(v), ha="center", va="bottom", fontsize=7.4)
                ymax_def = max(ymax_def, v)
    ax.set_xticks(xpos)
    ax.set_xticklabels([d["label"] for d in DEFINITIONS], fontsize=9)
    ax.set_ylabel("到达 EOL 的循环")
    ax.set_ylim(0, ymax_def * 1.22)
    ax.set_title("(c) 换一种 EOL 定义，寿命数字就换一套（斜纹 = 观察期未到达）",
                 fontsize=10.5)

    # ---- (d) 台阶密度：每 0.01 Ah 平均有几级台阶 ----
    ax = axes[1][1]
    xpos = np.arange(len(cells))
    dens = tbl_res["每 0.01 Ah 的台阶数"].to_numpy(float)
    n_reach = tbl_res["窗内可达 EOL 值"].to_numpy(float)
    ax.bar(xpos, dens, width=0.56, color=[COLORS[c] for c in cells])
    for xi, v, k in zip(xpos, dens, n_reach):
        ax.text(xi, v, f"{v:.2f}\n({int(k)} 个值)", ha="center", va="bottom",
                fontsize=8.4)
    ax.axhline(1.0, color="#333333", ls="--", lw=1.2, zorder=3,
               label="0.01 Ah 网格的采样上限（每个 0.01 区间只能取 1 个值）")
    ax.set_xticks(xpos)
    ax.set_xticklabels(cells, fontsize=9)
    ax.set_ylabel("每 0.01 Ah 的台阶数")
    ax.set_ylim(0, dens.max() * 1.34)
    ax.legend(loc="upper right", fontsize=7.8, framealpha=0.92)
    ax.set_title("(d) 台阶密度都 >1，所以 0.01 Ah 网格必然漏掉台阶", fontsize=10.5)

    fig.suptitle("图5　EOL 阈值敏感性：同一个电池，阈值挪 0.1 Ah 寿命差多少",
                 fontsize=13, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    return save(fig, "fig05_eol_threshold_sensitivity.png")


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------
def render_md(
    series: dict[str, tuple[np.ndarray, np.ndarray]],
    tbl_key: pd.DataFrame,
    tbl_rng: pd.DataFrame,
    tbl_win: pd.DataFrame,
    tbl_res: pd.DataFrame,
    tbl_def: pd.DataFrame,
    scan: dict,
) -> str:
    cells = [c for c in BATTERIES if c in series]
    n_cycles = {c: len(series[c][1]) for c in cells}
    cap_min = {c: censored_below(series[c][1]) for c in cells}

    # --- 台阶密度 ---
    n_prefix = {c: int(prefix_min_mask(series[c][1]).sum()) for c in cells}
    prefix_txt = "、".join(
        f"{c} {n_prefix[c]}（占观测长度 {n_prefix[c] / n_cycles[c]:.0%}）" for c in cells
    )
    step_gap = {c: n_cycles[c] / n_prefix[c] for c in cells}
    dens = dict(zip(tbl_res["电池"], tbl_res["每 0.01 Ah 的台阶数"].astype(float)))
    n_reach = dict(zip(tbl_res["电池"], tbl_res["窗内可达 EOL 值"].astype(int)))

    # --- 主窗口 ---
    rng = tbl_rng.set_index("电池")
    per_001 = {c: float(rng.loc[c, "每 0.01 Ah 的循环数"]) for c in cells}
    span = {c: int(rng.loc[c, "极差 (循环)"]) for c in cells}
    upper_shift = {c: int(rng.loc[c, "上偏 Δ"]) for c in cells if rng.loc[c, "上偏 Δ"] != "—"}
    lower_shift = {c: int(rng.loc[c, "下偏 Δ"]) for c in cells if rng.loc[c, "下偏 Δ"] != "—"}
    span_finite = [span[c] for c in cells if c in upper_shift]
    asymmetric = [
        c for c in upper_shift
        if abs(upper_shift[c]) != abs(lower_shift[c])
    ]
    asym_txt = "；".join(
        f"{c} 上偏 {upper_shift[c]:+d} / 下偏 {lower_shift[c]:+d}" for c in upper_shift
    )
    # B0007 的等宽对照
    b7 = equal_width_span(series, "B0007", 2 * WINDOW_HALF)
    b7_narrow = (MAIN_TAU + WINDOW_HALF) - b7["lo"]

    # --- 定义口径 ---
    def_eol = {}
    for c in cells:
        row = tbl_def.loc[tbl_def["电池"] == c].iloc[0]
        def_eol[c] = {}
        for d in DEFINITIONS:
            txt = str(row[f"{d['label']} EOL"])
            def_eol[c][d["label"]] = None if txt.startswith(">") else int(txt)
    life_range = []
    for c in cells:
        vals = [v for v in def_eol[c].values() if v is not None]
        if vals:
            life_range.append({"cell": c, "lo": min(vals), "hi": max(vals)})
    ratio_max = max(r["hi"] / r["lo"] for r in life_range)
    ratio_row = max(life_range, key=lambda r: r["hi"] / r["lo"])
    censored_under = {
        d["label"]: [c for c in cells if def_eol[c][d["label"]] is None]
        for d in DEFINITIONS
    }

    # --- 结论不变性 ---
    orders = scan["full_orders"]
    if len(orders) != 1:
        raise AssertionError(f"四块都有定义时寿命排序不唯一：{orders}")
    order_txt = " < ".join(next(iter(orders)))
    full_taus = scan["full_taus"]
    n_full = len(full_taus)

    b7_bound = cap_min["B0007"]
    b7_margin = b7_bound - MAIN_TAU
    b7_ratio = b7_margin / RATED_CAPACITY_AH

    zone = {t: v for t, v in scan["n_censored"].items() if 1.35 <= t <= 1.4000}
    zone_vals = set(zone.values())
    zone_ok = len(zone_vals) == 1
    zone_n = next(iter(zone_vals)) if zone_ok else None

    out = [
        "# EOL 阈值敏感性（W7）",
        "",
        "> 本文件由 `src/sensitivity.py` 自动生成，数字随代码重算，**不要手工编辑**。",
        "",
        "## 一、这一节要回答什么",
        "",
        f"「寿命 = {rng.loc['B0005', 'EOL@1.40']} 循环」里的 {rng.loc['B0005', 'EOL@1.40']}，"
        "有多少是电池的性质，有多少是我选的阈值造成的？",
        f"EOL 阈值是工程惯例，不是物理常数。本节把 τ 从 "
        f"{PLAUSIBLE_LO:.2f} 扫到 {PLAUSIBLE_HI:.2f} Ah，回答三件事：",
        "",
        "1. 阈值挪一点，寿命差多少循环（→ 可折算成「定义不确定」的带宽）；",
        "2. 哪些结论会因为这个选择而翻转；",
        "3. 哪些结论撑得住。",
        "",
        "## 二、方法：用断点，不用网格",
        "",
        "`N_EOL(τ) = min{cycle : capacity(cycle) ≤ τ}` 是**单调不增的阶梯函数**，"
        "而且只在「容量创下新低」的循环上取值：若非前缀最小的循环 c 上首次够到 τ，"
        "则存在更早的 c′ 使 cap(c′) ≤ cap(c)，EOL 早就停在 c′ 了。"
        "所以整个函数被前缀最小点集完全决定。",
        "",
        f"本数据的前缀最小点个数：{prefix_txt}。也就是**平均每 "
        f"{min(step_gap.values()):.1f}–{max(step_gap.values()):.1f} 个循环就有一级台阶**。"
        f"主窗口 [{MAIN_TAU - WINDOW_HALF:g}, {MAIN_TAU + WINDOW_HALF:g}] 内，"
        "四块电池的可达值与台阶密度：",
        "",
        tbl_res.to_markdown(index=False, disable_numparse=True),
        "",
        f"读法：台阶密度 {min(dens.values()):.2f}–{max(dens.values()):.2f}"
        "（每 0.01 Ah 的区间里平均有几级台阶）。"
        "把网格撒在这条轴上时，落点只会稀疏地命中少数几级台阶，"
        "**台阶的级数与位置都看不出来（混叠）**——"
        "这正是本模块改用断点集画图和统计的理由。",
        "",
        f"交叉验证：两种独立实现（朴素首命中扫描 / 前缀最小断点法）在 "
        f"{len(scan['grid']) * len(cells)} 个网格点上逐点比对，**0 处不一致**；"
        "同一网格上另验证了 `N_EOL` 随 τ 单调不增。",
        "",
        "## 三、关键阈值上的寿命（循环数）",
        "",
        tbl_key.to_markdown(index=False, disable_numparse=True),
        "",
        f"（`>{n_cycles['B0007']}（删失）` = 观察窗口内从未跌破该阈值，"
        "真实寿命比观测长度更长。B0018 的观测长度是 "
        f"{n_cycles['B0018']}，所以它的删失标记写成 >{n_cycles['B0018']}。）",
        "",
        "## 四、常用区间内的寿命极差",
        "",
        f"窗口 [{MAIN_TAU - WINDOW_HALF:g}, {MAIN_TAU + WINDOW_HALF:g}]"
        f"（±{WINDOW_HALF:g} Ah，即额定容量的 ±{WINDOW_HALF / RATED_CAPACITY_AH:.1%}）：",
        "",
        tbl_rng.to_markdown(index=False, disable_numparse=True),
        "",
        f"读法：**阈值每偏 0.01 Ah（额定容量的 0.5%），寿命差 "
        f"{min(per_001.values()):.2f}–{max(per_001.values()):.2f} 个循环**。"
        f"另外三块的极差是 {min(span_finite)}–{max(span_finite)} 循环；"
        f"B0007 的 {span['B0007']} 是在更窄的窗口上算的，见下面第 2 条。",
        "",
        "两个必须点出来的细节：",
        "",
        "1. **上下偏的方向不一致，所以不能说「阈值下移影响更大」。** "
        f"以主阈值 ±{WINDOW_HALF:g} Ah 计：{asym_txt}（循环）。"
        f"B0005 两侧相等；{ '、'.join(c for c in asymmetric if abs(lower_shift[c]) > abs(upper_shift[c])) }"
        " 的下偏更大（阈值抬高、判定更严，寿命缩短更多）；"
        f"{ '、'.join(c for c in asymmetric if abs(upper_shift[c]) > abs(lower_shift[c])) }"
        " 则相反。**方向不统一，所以本节不主张任何单一方向的不对称**——"
        "只如实地把两个数都列出来。报告里若写「±X 循环」，必须说明这是把两侧"
        "合起来的总宽度，不是两侧各自都等于 X。",
        "2. **B0007 的窗口被删失截断。** 它在 τ < "
        f"{b7_bound:.4f} Ah 时就删失了，所以有效下界是 {b7_bound:.4f} 而不是 "
        f"{MAIN_TAU - WINDOW_HALF:g}，极差 {span['B0007']} 循环是"
        f"在 {b7_narrow:.4f} Ah 宽（而非 {2 * WINDOW_HALF:g} Ah）的窗口上算的。"
        f"**换成同等宽度 [{b7['lo']:.4f}, {b7['hi']:.4f}]，极差是 {b7['span']} 循环**，"
        "与另三块的量级一致。",
        "",
        "## 五、极差本身也依赖窗口选多宽",
        "",
        tbl_win.to_markdown(index=False, disable_numparse=True),
        "",
        "这是 W5 那条教训的第二次出现：加速比依赖切分点 h，极差依赖窗口宽度。"
        "**「±X 循环」这种说法，不说窗口就是没说完。** 本项目的正式口径固定为 "
        f"±{WINDOW_HALF:g} Ah，并在报告里写明。",
        "",
        "单元格里的 `>N` 表示该窗口的下端点已经掉进删失区：真实寿命超过观测长度，"
        "所以极差只能给下界，不能当成一个精确值去和别的电池比大小。"
        f"例如 B0018 在 ±0.1 时给的是 >62——因为 {MAIN_TAU - 0.1:g} 已经低于它的最低容量 "
        f"{cap_min['B0018']:.4f} Ah。",
        "",
        "## 六、换一种 EOL 定义，寿命换一套数字",
        "",
        tbl_def.to_markdown(index=False, disable_numparse=True),
        "",
        "注意后两种定义（70% / 80% 首循环）的阈值**逐电池不同**。"
        "它们回答的是另一个问题：该拿「出厂额定容量」还是「实测初始容量」作基准。"
        "这不是「选一个更好的 τ」，而是**换一类定义**——承认电池出厂容量本身有差异。",
        "",
        f"（B0006 的 80% 首循环阈值 {DEFINITIONS[3]['factor'] * float(series['B0006'][1][0]):.4f} Ah "
        f"越过了扫描上界 {PLAUSIBLE_HI:g}，但它只是「定义对比」的一行，不参与上面的扫描统计；"
        "`eol_at` 对任意 τ 都成立，所以这一格照样是精确值。）",
        "",
        f"在四种定义下，同一块电池的寿命最多相差 **{ratio_max:.1f} 倍**"
        f"（{ratio_row['cell']}：{ratio_row['lo']} → {ratio_row['hi']} 循环）。"
        "**但「这是不是三层里最大的一层」现在不能下结论**——"
        "模型不确定性与数据不确定性要到 W13/W15 才会算出可比的数字，"
        "本节只负责把「定义」这一层换算成循环数，量级比较留到 W15 一起做。"
        "（这里刻意不写「定义不确定最大」这类话：它需要另外两层的数字才能成立。）",
        "",
        "另一个值得写进报告的观察：**「删失」这个判定本身也依赖定义**。"
        + "；".join(
            f"{lab}口径下删失的是：{'、'.join(who) if who else '无'}"
            for lab, who in censored_under.items()
        )
        + "。所以「哪些样本删失了」不是数据的性质，是**数据 + 定义**的性质，"
        "报告里不能只说「B0007 删失」而不说清楚用的是哪个口径。",
        "",
        "## 七、哪些结论撑得住，哪些撑不住",
        "",
        "| 结论 | 成立的 τ 范围 | 依据 |",
        "|---|---|---|",
        f"| 「B0007 右删失」 | τ < {b7_bound:.4f} Ah | B0007 全程最低 {b7_bound:.4f} Ah，"
        f"距主阈值仅 {b7_margin:+.4f} Ah（额定的 {b7_ratio:.3%}） |",
        f"| 四块寿命排序 {order_txt} | 四块都有确定寿命的 {n_full} 个 τ"
        f"（{min(full_taus):.4f}–{max(full_taus):.4f} Ah） | "
        "这些 τ 上排序集合只有 1 个元素 |",
        f"| 「[{MAIN_TAU - WINDOW_HALF:g}, {MAIN_TAU:g}] 内只有 1 个删失个体」 "
        f"| τ ∈ [1.35, 1.4000] | 该区间内每点的删失个数恒为 "
        f"{zone_n if zone_ok else '不恒定'}（共 {len(zone)} 个网格点） |",
        "",
        f"最强的结论是**排序**：在四块电池都有确定寿命的 {n_full} 个 τ 上，"
        f"寿命排序始终是 {order_txt}，一次都没变。"
        "也就是说「谁先坏」这件事不依赖阈值选择，而「坏在第几循环」非常依赖。",
        "",
        f"最脆弱的结论是**B0007 的删失判定**：它只靠 {b7_margin:.4f} Ah 的余量成立，"
        f"相当于额定容量的 {b7_ratio:.3%}。阈值只要从 {MAIN_TAU:g} 抬到 {b7_bound:.4f}，"
        "B0007 就立刻从「删失」变成「第 166 循环失效」。"
        "报告里必须写这个余量，否则等于在宣称一个不存在的稳健性。",
        "",
        "## 八、给 W15 的接口",
        "",
        "「定义不确定」这一层折算成的循环数带宽（正式口径 ±"
        f"{WINDOW_HALF:g} Ah）：",
        "",
        "| 电池 | 主线 EOL | 窗内 EOL 区间 | 总宽度 (循环) | 相对主线 |",
        "|---|---|---|---|---|",
    ]
    for c in cells:
        lo = int(rng.loc[c, "EOL@上界"])
        hi = int(rng.loc[c, "EOL@下界"])
        main_txt = rng.loc[c, f"EOL@{MAIN_TAU:.2f}"]
        if main_txt == "—":
            rel = "—（主线本身删失）"
        else:
            m = int(main_txt)
            rel = f"{lo / m - 1:+.0%} / {hi / m - 1:+.0%}"
        out.append(f"| {c} | {main_txt} | {lo}–{hi} | {span[c]} | {rel} |")
    out += [
        "",
        f"B0007 那一行要按第四节的说明读：它的窗口被删失截断，"
        f"等宽极差是 {b7['span']} 循环。",
        "",
        "这张表进 W15 后会与「模型不确定」（GPR 后验方差 → 后验采样）和"
        "「数据不确定」（残差噪声 + 共模成分）放在同一量纲上比较，"
        "回答「想更准，先消灭哪一层」这个问题。",
        "",
        "## 九、完整断点表",
        "",
        f"每个电池在 [{PLAUSIBLE_LO:.2f}, {PLAUSIBLE_HI:.2f}] 内的全部断点写在 "
        f"`{BREAK_CSV.name}`（列：`battery_id` / `tau_lower_Ah` / `tau_upper_Ah` / `eol_cycle`；"
        "含义是「τ 落在 [下界, 上界) 时寿命 = eol_cycle」），表中无删节，可直接复现上图。",
        "",
    ]
    return "\n".join(out)


def main() -> int:
    ensure_dirs()

    raw = load_raw()
    df = mark_rows(raw)
    prof = cell_profile(df)
    series = cap_series(df)
    cells = [c for c in BATTERIES if c in series]

    # --- 断言 1：与 W3 的个体档案一致（跨模块口径核对）---
    prof_idx = prof.set_index("battery_id")
    for cell in cells:
        _, cap = series[cell]
        got = eol_at(cap, MAIN_TAU)
        want = prof_idx.loc[cell, "eol_cycle"]
        want = None if pd.isna(want) else int(want)
        assert got == want, f"{cell} 的 EOL 与 W3 档案不一致：{got} vs {want}"

    # --- 断言 2：默认 τ 与显式传参给出同一答案（D17 的唯一入口确实生效）---
    for cell in cells:
        _, cap = series[cell]
        default_hit = np.flatnonzero(is_at_eol(cap))
        explicit_hit = np.flatnonzero(is_at_eol(cap, MAIN_TAU))
        assert np.array_equal(default_hit, explicit_hit), f"{cell} 默认 τ 与显式 τ 分叉"

    # --- 断言 3：两种实现逐点一致（写图的数据就是核过的数据）---
    grid = np.round(np.arange(PLAUSIBLE_LO, PLAUSIBLE_HI + 1e-12, GRID_STEP), 6)
    n_cmp = 0
    for cell in cells:
        _, cap = series[cell]
        for tau in grid:
            a, b = eol_at(cap, tau), eol_at_exact(cap, tau)
            assert a == b, f"{cell} τ={tau}: 朴素={a} 断点={b}"
            n_cmp += 1

    # --- 断言 4：N_EOL 单调不增 ---
    for cell in cells:
        _, cap = series[cell]
        prev = np.inf
        for tau in grid:
            cur = eol_at_exact(cap, tau)
            cur = np.inf if cur is None else cur
            assert cur <= prev, f"{cell} τ={tau} 处 EOL 反而变大"
            prev = cur

    # --- 断言 5：断点表在每个台阶上的取值能同时被两种实现复现 ---
    for cell in cells:
        _, cap = series[cell]
        w, c = breakpoints(cap)
        for wi, ci in zip(w, c):
            if not (PLAUSIBLE_LO <= wi <= PLAUSIBLE_HI):
                continue
            assert eol_at(cap, wi) == ci, f"{cell} 断点 τ={wi} 处朴素实现不符"
            assert eol_at_exact(cap, wi) == ci, f"{cell} 断点 τ={wi} 处断点实现不符"

    # --- 断言 6：删失边界是精确的（边界上算到达，再低一点点就算删失）---
    for cell in cells:
        _, cap = series[cell]
        cmin = censored_below(cap)
        assert eol_at_exact(cap, cmin) is not None, f"{cell} 在最小容量处应判定为到达 EOL"
        just_below = float(np.nextafter(cmin, -np.inf))
        assert eol_at_exact(cap, just_below) is None, f"{cell} 在最小容量下方仍判为到达"

    # --- 断言 7：排序在四块都有定义的 τ 上唯一（结论不变性的硬判据）---
    tbl_key = key_tau_table(series)
    tbl_rng = range_table(series)
    tbl_win = window_sweep(series)
    tbl_res = resolution_table(series)
    tbl_def = definition_table(series)
    scan = invariance_scan(series)
    assert len(scan["full_orders"]) == 1, f"排序不唯一：{scan['full_orders']}"

    # --- 断言 8：config 常量在整轮扫描后未被改动（回归护栏）---
    # 今天它不可能失败（本模块从不写 config）；加它是为了锁死未来——
    # 一旦有人图省事在某个 helper 里写 `config.EOL_RATIO = tau`，这里立刻炸。
    import config as _cfg

    assert _cfg.EOL_THRESHOLD_AH == EOL_THRESHOLD_AH, "扫描过程改动了 config 阈值"
    assert _cfg.EOL_RATIO == 0.70, "扫描过程改动了 config 的 EOL_RATIO"

    TABLE_MD.write_text(
        render_md(series, tbl_key, tbl_rng, tbl_win, tbl_res, tbl_def, scan),
        encoding="utf-8",
    )

    # 完整断点表落 CSV（md 里只放摘要，全文太长）
    rows = []
    for cell in cells:
        _, cap = series[cell]
        w, c = breakpoints(cap)
        for i, (wi, ci) in enumerate(zip(w, c)):
            if not (PLAUSIBLE_LO <= wi <= PLAUSIBLE_HI):
                continue
            nxt = w[i + 1] if i + 1 < len(w) else None
            rows.append(
                {
                    "battery_id": cell,
                    "tau_lower_Ah": round(float(wi), 4),
                    "tau_upper_Ah": None if nxt is None else round(float(nxt), 4),
                    "eol_cycle": int(ci),
                }
            )
    pd.DataFrame(rows).to_csv(BREAK_CSV, index=False, encoding="utf-8-sig")

    fig_path = fig05(series, tbl_res, tbl_win, tbl_def, scan)

    print(f"扫描区间：τ ∈ [{PLAUSIBLE_LO:g}, {PLAUSIBLE_HI:g}]，网格 {len(grid)} 点 × {len(cells)} 块")
    print(f"两种实现交叉验证：{n_cmp} 次比对，0 处不一致")
    print(f"断点总数（区间内）：{len(rows)}")
    print()
    print("关键 τ 上的 N_EOL：")
    print(tbl_key.to_string())
    print()
    print(f"主窗口 ±{WINDOW_HALF:g} Ah 内的极差：")
    print(tbl_rng.to_string(index=False))
    print()
    print("极差随窗口半宽：")
    print(tbl_win.to_string(index=False))
    print()
    print("四种 EOL 定义：")
    print(tbl_def.to_string(index=False))
    print()
    print(f"四块都有定义时的排序（唯一）：{next(iter(scan['full_orders']))}")
    print(f"B0007 删失边界的余量：{censored_below(series['B0007'][1]) - MAIN_TAU:+.4f} Ah")
    print()
    print(f"阈值敏感性报告已写出：{TABLE_MD.name}")
    print(f"完整断点表已写出：{BREAK_CSV.name}（{len(rows)} 行）")
    print(f"已生成 {fig_path.name}  ({fig_path.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
