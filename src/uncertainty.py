"""W15：三层不确定性——定义 / 模型形式 / 数据，统一折算成「循环数 ± 多少」。

这一节是整篇报告的主线所在，所以先把三层的**定义**钉死（不许事后调整）：

1. **定义层**（我们怎么定义「坏」）：EOL 阈值 τ 是工程惯例，不是物理常数。
   取维护政策的合理范围 **τ = 1.40 ± 0.05 Ah**，量「窗内 EOL 的极差」→ 循环数。
   来源是 W7 的**无模型**扫描（直接在原始容量序列上找穿越点），不是模型外推。
2. **模型形式层**（假设了哪种函数形式）：同一份数据、同一套指标，
   M1 线性 / M2 双指数 / M4 趋势+GPR 给出的 EOL 各不相同。
   我们**没有证据选谁**（W14 的逐点配对检验：M1 与 M4 在远场几乎相同，
   M2 在多数电池上显著更差，但「形式之差」本身就是要报的不确定性）。
   量法：各形式 EOL 的**半极差** (max−min)/2。
3. **数据层**（噪声与参数估不准）：M4 的 90% 后验区间半宽 (q95−q05)/2。
   它含残差噪声 + 趋势参数的估计误差两层（W11 的方差分解），
   但**不含**前两层。这也是 M4 单独覆盖不住真值的原因。

**为什么不能把三层合成一个「95% 区间」**：三层里只有数据层有概率意义；
定义层与模型形式层是**系统性的**（换个惯例/换个形式，偏差朝同一方向走），
不是随机抽样。所以本节给两个合并口径，并明说各自含义：

- **RSS**（按独立假设）：`sqrt(d² + m² + s²)`——只有三层互不相关时才成立，
  对系统性偏差是**乐观**的（会低估）；
- **SUM**（按最坏情况）：`d + m + s`——假设三层都朝同一方向偏到底，
  对决策是**保守**的。

结论要回答的不是「覆盖率很高」，而是：**M4 的区间为什么窄**、
**想让结论更准应该先消灭哪一层**。

关于「共模成分」：W6 发现跨电池存在同步波动（同批三块共模占残差方差 82.4%）。
在**逐个体**模型里它已经被残差项整体吸收，GP 不区分「共模」与「个体噪声」，
所以这里**不再单列一层**——单列会把同一份方差算两次。这一点在 md 里说明。

产物
----
- `outputs/tables/uncertainty_budget.md`
- `outputs/metrics/uncertainty_budget.csv`
- `outputs/figures/fig13_uncertainty_budget.png`

运行
----
    python src/uncertainty.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (  # noqa: E402
    EOL_THRESHOLD_AH,
    METRIC_DIR,
    RATED_CAPACITY_AH,
    TABLE_DIR,
    ensure_dirs,
    setup_matplotlib,
)
from evaluate import SPLIT_M  # noqa: E402
from figures import save  # noqa: E402
from load_data import load_raw, split_by_battery  # noqa: E402
from preprocess import cell_profile, mark_rows  # noqa: E402

TABLE_MD = TABLE_DIR / "uncertainty_budget.md"
CSV_OUT = METRIC_DIR / "uncertainty_budget.csv"

TAU_HALF = 0.05
"""定义层的窗口半宽（Ah）。W7 的正式口径，理由：维护政策上「70% 额定」没有唯一答案，
±0.05 Ah 覆盖了常见的 67.5%–72.5% 各种取法。这个数**必须与 W7 一致**，代码里对得上。"""

MODEL_LAYER = ("M1 线性", "M2 双指数", "M4 趋势+GPR")
"""参与「模型形式」离散度的模型。**不含 M3**：它在四块电池上都不给 EOL（W12），
进不了这个量纲。这是设计上的排除，不是挑好的留下。"""


# ---------------------------------------------------------------------------
# 三层
# ---------------------------------------------------------------------------
def layer_definition(
    series: dict[str, tuple[np.ndarray, np.ndarray]]
) -> dict[str, tuple[float, str]]:
    """定义层：τ 在 [1.40−h, 1.40+h] 内变动时，EOL 的极差（**无模型**）。

    与 W7 的 `window_sweep` 同一算法，但下端点掉进删失区时**不编数**：
    改用该电池的可达窗口 `[censored_below(cap), 1.40+h]`，并在口径列里写明。
    （若硬把下端点当 168 循环用，得到的是下界「>24」，与 W7 发表的 22 打架；
    两个数都对，但含义不同——所以这里选可达窗口，与 W7 的正文一致。）
    """
    import sensitivity

    out = {}
    for cell, (_, cap) in series.items():
        hi_tau = EOL_THRESHOLD_AH + TAU_HALF
        hi = sensitivity.eol_at_exact(cap, hi_tau)
        if hi is None:
            out[cell] = (float("nan"), "上端删失，无法定义")
            continue
        lo_tau = EOL_THRESHOLD_AH - TAU_HALF
        if lo_tau < sensitivity.censored_below(cap):
            lo_tau = sensitivity.censored_below(cap)
            note = f"下端删失，用可达窗口 [{lo_tau:.4f}, {hi_tau:g}]"
        else:
            note = f"完整窗口 [{lo_tau:g}, {hi_tau:g}]"
        lo = sensitivity.eol_at_exact(cap, lo_tau)
        out[cell] = (float(lo - hi), note)
    return out


def layer_model(param_tbl: pd.DataFrame, gpr_tbl: pd.DataFrame) -> pd.DataFrame:
    """模型形式层：同一块电池，不同形式给出的 EOL 的半极差。

    **必须同时给两个版本**（R9-01 抓出来的要害）：
    M1 与 M4 的 EOL 四个数完全相同（146/94/159/97，这是 W11 的结论：GP 不外推），
    所以「含 M2」与「不含 M2」的模型层是两回事：
    - `含 M2`：M2 与 M1 的半极差，19.5 / 2.5 / 22 / 0；
    - `不含 M2`：**恒为 0**——这一层的全部宽度都由双指数形式提供。

    而且 W14 已经判定 M2 在三块电池上显著更差。所以这一层的正确说法是
    「**若把双指数也当作先验可信的形式之一，形式之差的上界有多大**」，
    不是「三种形式势均力敌」。
    """
    rows = []
    for cell in sorted(param_tbl["电池"].unique()):
        vals = {}
        for name in MODEL_LAYER:
            if name.startswith("M4"):
                s = gpr_tbl.loc[gpr_tbl["电池"] == cell, "EOL预测"]
            else:
                s = param_tbl.loc[(param_tbl["电池"] == cell) & (param_tbl["模型"] == name),
                                  "EOL预测"]
            v = s.iloc[0]
            vals[name] = None if str(v) == "—" else int(v)
        finite = [v for v in vals.values() if v is not None]
        assert len(finite) >= 2, f"{cell} 上少于两个模型给出 EOL，算不出形式离散度"
        no_m2 = [v for k, v in vals.items() if k != "M2 双指数" and v is not None]
        rows.append(
            {
                "电池": cell,
                **vals,
                "模型层半极差": (max(finite) - min(finite)) / 2.0,
                "模型层半极差(不含M2)": (max(no_m2) - min(no_m2)) / 2.0,
                "参与模型数": len(finite),
            }
        )
    return pd.DataFrame(rows)


def layer_data(gpr_fits: dict, cells: list[str]) -> pd.DataFrame:
    """数据层：M4 后验 90% 区间半宽。"""
    rows = []
    for cell in cells:
        d = gpr_fits[cell]["dist"]
        rows.append(
            {
                "电池": cell,
                "后验 p05": float(d["q05"]),
                "后验 中位数": float(d["median"]),
                "后验 p95": float(d["q95"]),
                "数据层半宽": (float(d["q95"]) - float(d["q05"])) / 2.0,
            }
        )
    return pd.DataFrame(rows)


def cross_check_definition(
    out: dict[str, tuple[float, str]],
    series: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    """与 W7 已发表的数字对账：同一算法的两条实现，逐块必须相等。

    **必须放在 `layer_definition` 外面**：写在被替换的函数里，
    变异把函数整个换掉就连检查一起换掉了（这条教训 R8 已经付过一次学费）。
    """
    import sensitivity

    # 口径必须与 W7 一致：窗口半宽、以及「哪块电池用了短窗口」
    assert TAU_HALF in sensitivity.WINDOW_HALVES, (
        f"TAU_HALF={TAU_HALF} 不在 W7 的窗口集合 {sensitivity.WINDOW_HALVES} 里——两边口径分叉了"
    )

    ws = sensitivity.window_sweep(series)
    col = f"±{TAU_HALF:g}"
    n_checked = 0
    n_short = 0
    for _, row in ws.iterrows():
        name = row["电池"]
        got, note = out[name]
        try:
            v = float(row[col])
        except (TypeError, ValueError):
            # B0007 在这里：W7 给的是下界「>24」，本模块给的是可达窗口的 22。
            # 两条口径都要显式钉住，否则这块电池的检查就是空白（R9-09）。
            full_span = float(len(series[name][1]) - sensitivity.eol_at_exact(
                series[name][1], EOL_THRESHOLD_AH + TAU_HALF))
            assert abs(full_span - 24.0) < 1e-9, (
                f"{name} 的完整窗口下界应为 24（= 168 − 144），实得 {full_span}"
            )
            assert "下端删失" in note, f"{name} 用了短窗口却没在口径列里写明"
            n_short += 1
            continue
        assert abs(got - v) < 1e-9, (
            f"{name}: 本模块算的定义层极差 {got} 与 W7 发表的 {v} 不一致——同一算法两个实现打架"
        )
        n_checked += 1
    assert n_checked >= 3, (
        f"只对账了 {n_checked} 块——这个检查形同虚设（列名或口径可能改了）"
    )
    assert n_short == 1, f"短窗口电池应有 1 块（B0007），实得 {n_short} 块"


def assert_layers_contribute(tbl: pd.DataFrame) -> None:
    """每一层都必须**真的进入**合并结果（R9-10：原来的三条是恒真式）。

    原来断言「RSS ≥ 任一层」「SUM ≥ RSS」——对任意非负三元组恒成立，
    把某一层整层删掉也照样通过。改成**扰动检查**：把某一层置 0，RSS 必须变小；
    这样「少了一层」就再也藏不住。
    """
    for col in ("定义层半宽", "模型层半极差", "数据层半宽"):
        base = tbl["rss"].to_numpy(float)
        other = [c for c in ("定义层半宽", "模型层半极差", "数据层半宽") if c != col]
        perturbed = np.sqrt(np.sum([tbl[c].to_numpy(float) ** 2 for c in other], axis=0))
        sum_perturbed = sum(tbl[c].to_numpy(float) for c in other)
        mask = tbl[col].to_numpy(float) > 0
        assert mask.any(), f"「{col}」在所有电池上都是 0——这一层实际不存在，报告里不能当一层卖"
        assert (perturbed[mask] < base[mask]).all(), (
            f"把「{col}」整层置 0 之后 RSS 没变小——这一层没进入合并结果"
        )
        assert (sum_perturbed[mask] < tbl["sum"].to_numpy(float)[mask]).all(), (
            f"把「{col}」整层置 0 之后 SUM 没变小——这一层没进入合并结果"
        )


def assert_model_layer_source(tbl: pd.DataFrame) -> None:
    """把 R9-01 的事实钉成断言：模型形式层的宽度**全部来自 M2**。

    M1 线性与 M4 趋势+GPR 的 EOL 四个数完全相同（W11：GP 不外推，点预测=趋势线），
    所以「不含 M2」的半极差恒为 0。这不是 bug，是结构决定的——
    但正因如此，报告里不许把这一层说成「三种形式势均力敌」。
    """
    assert (tbl["模型层半极差(不含M2)"].to_numpy(float) == 0).all(), (
        "排除 M2 之后模型形式层不为 0——说明 M1 与 M4 的 EOL 不再相同，"
        "报告里「形式层全部来自 M2」这句话必须重写"
    )
    assert (tbl["模型层半极差"] >= tbl["模型层半极差(不含M2)"]).all(), (
        "含 M2 的形式层比不含 M2 还窄——半极差的算法写错了"
    )


def assert_combined(tbl: pd.DataFrame) -> None:
    """合并口径的结构性检查（放在调用方，不在 combine 里面）。"""
    assert (tbl["sum"].to_numpy(float) >= tbl["rss"].to_numpy(float) - 1e-9).all(), (
        "最坏情况合并比 RSS 还窄——合并口径写反了"
    )
    assert_layers_contribute(tbl)


def combine(tbl: pd.DataFrame) -> pd.DataFrame:
    """三层合并：RSS（独立假设，乐观）与 SUM（最坏情况，保守）。只算，不检查。"""
    out = tbl.copy()
    d, m, s = (out["定义层极差"] / 2.0, out["模型层半极差"], out["数据层半宽"])
    out["定义层半宽"] = d
    out["rss"] = np.sqrt(d**2 + m**2 + s**2)
    out["sum"] = d + m + s

    out["点估计"] = pd.to_numeric(out["点估计"], errors="coerce")
    out["真值"] = pd.to_numeric(out["真值"], errors="coerce")
    out["误差"] = out["点估计"] - out["真值"]
    has_truth = out["真值"].notna()
    # 布尔列在删失电池上写「—」，所以用 object dtype（直接赋 pd.NA 会 TypeError）
    out["RSS 覆盖"] = (
        (out["真值"] >= out["点估计"] - out["rss"])
        & (out["真值"] <= out["点估计"] + out["rss"])
    ).astype(object).where(has_truth, "—")
    out["SUM 覆盖"] = (
        (out["真值"] >= out["点估计"] - out["sum"])
        & (out["真值"] <= out["点估计"] + out["sum"])
    ).astype(object).where(has_truth, "—")
    return out


# ---------------------------------------------------------------------------
# 图
# ---------------------------------------------------------------------------
def fig13(tbl: pd.DataFrame) -> Path:
    """两联图：① 三层堆叠（循环数）；② 合并区间 vs 真值。"""
    setup_matplotlib()
    cells = tbl["电池"].tolist()
    idx = np.arange(len(cells))

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.6))

    ax = axes[0]
    d = tbl["定义层半宽"].to_numpy(float)
    m = tbl["模型层半极差"].to_numpy(float)
    s = tbl["数据层半宽"].to_numpy(float)
    ax.bar(idx, d, 0.55, label="定义层（τ ±0.05 Ah）", color="#8c8c8c")
    ax.bar(idx, m, 0.55, bottom=d, label="模型形式层（三种形式的半极差）", color="#c0392b")
    ax.bar(idx, s, 0.55, bottom=d + m, label="数据层（M4 后验 90% 半宽）", color="#6a3d9a")
    for i, (dd, mm, ss) in enumerate(zip(d, m, s)):
        ax.plot([i, i], [0, dd + mm + ss], color="#333333", lw=1.2)
        ax.scatter([i], [np.sqrt(dd**2 + mm**2 + ss**2)], marker="_", s=260,
                   color="#1f6f8b", zorder=6, linewidths=2.4)
    ax.set_xticks(idx)
    ax.set_xticklabels(cells)
    ax.set_ylabel("EOL 的不确定度（循环，半宽）")
    ax.set_ylim(0, float(np.nanmax(d + m + s)) * 1.5)
    n_lab = int(tbl["真值"].notna().sum())
    c_rss = int(tbl.loc[tbl["真值"].notna(), "RSS 覆盖"].astype(bool).sum())
    c_sum = int(tbl.loc[tbl["真值"].notna(), "SUM 覆盖"].astype(bool).sum())
    ax.set_title(f"① 三层预算（粗线 = SUM，蓝短横 = RSS）", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1]
    for i, row in enumerate(tbl.itertuples()):
        y = i
        if pd.isna(row.真值):
            ax.text(float(row.点估计), y, "  删失：无真值", va="center", fontsize=8,
                    color="#8a6d3b")
            continue
        ax.plot([row.点估计 - row.rss, row.点估计 + row.rss], [y - 0.12, y - 0.12],
                color="#1f6f8b", lw=4, solid_capstyle="butt", label="RSS" if i == 0 else None)
        ax.plot([row.点估计 - row.sum, row.点估计 + row.sum], [y + 0.12, y + 0.12],
                color="#b8860b", lw=4, solid_capstyle="butt", label="SUM" if i == 0 else None)
        ax.scatter([row.点估计], [y], marker="|", s=200, color="#2c3e50", zorder=5)
        ax.scatter([row.真值], [y], marker="*", s=170, color="#c0392b", zorder=6,
                   label="真值" if i == 0 else None)
    ax.set_yticks(idx)
    ax.set_yticklabels(cells)
    ax.set_xlabel("EOL（循环）")
    ax.set_title(f"② 两个合并口径 vs 真值：RSS 覆盖 {c_rss}/{n_lab}，SUM 覆盖 {c_sum}/{n_lab}",
                 fontsize=10)
    ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("三层不确定性统一折算成循环数（M = 80）", fontsize=13, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save(fig, "fig13_uncertainty_budget.png")


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------
def render_md(tbl: pd.DataFrame, m: int) -> str:
    n_label = int(tbl["真值"].notna().sum())
    cover_rss = int(tbl.loc[tbl["真值"].notna(), "RSS 覆盖"].sum())
    cover_sum = int(tbl.loc[tbl["真值"].notna(), "SUM 覆盖"].sum())
    widest = tbl.loc[tbl["sum"].idxmax()]
    rss_biggest = np.argmax(
        np.column_stack([tbl["定义层半宽"], tbl["模型层半极差"], tbl["数据层半宽"]]), axis=1
    )
    names = ["定义层", "模型形式层", "数据层"]
    counts = {nm: int((rss_biggest == k).sum()) for k, nm in enumerate(names)}
    b7_rows = tbl.loc[tbl["定义层口径"].astype(str).str.contains("下端删失")]
    assert len(b7_rows) == 1, "短窗口电池应当恰好一块"
    b7 = b7_rows.iloc[0]
    b7_short = float(b7["定义层极差"])
    b7_window = str(b7["定义层口径"]).split("窗口", 1)[-1].strip()
    full_short = float(b7["定义层完整窗口极差"])
    ratio_short = (float(b7["定义层半宽"]) + float(b7["模型层半极差"])) / float(b7["数据层半宽"])
    ratio_equal = (float(b7["定义层完整窗口极差"]) / 2 + float(b7["模型层半极差"])) / float(
        b7["数据层半宽"]
    )

    def _num_or_dash(v, nd=1):
        return "—" if pd.isna(v) else round(float(v), nd)

    disp3 = tbl[["电池", "点估计", "真值", "定义层极差", "定义层半宽",
                 "模型层半极差", "数据层半宽"]].copy()
    disp3["真值"] = disp3["真值"].map(lambda v: "删失" if pd.isna(v) else int(v))
    disp4 = tbl[["电池", "点估计", "真值", "误差", "rss", "sum", "RSS 覆盖", "SUM 覆盖"]].copy()
    disp4["真值"] = disp4["真值"].map(lambda v: "删失" if pd.isna(v) else int(v))
    disp4["误差"] = disp4["误差"].map(_num_or_dash)
    disp5 = tbl[["电池", "误差", "数据层半宽", "rss", "sum"]].copy()
    disp5["误差"] = disp5["误差"].map(_num_or_dash)
    out = [
        "# 三层不确定性：定义 / 模型形式 / 数据，统一折算成循环数（W15）",
        "",
        "> 本文件由 `src/uncertainty.py` 自动生成，数字随代码重算，**不要手工编辑**。",
        f"> 机器可读版：`outputs/metrics/{CSV_OUT.name}`。",
        "",
        "## 一、这一节回答什么",
        "",
        "「这块电池还能用 146 个循环」——这句话里的**不确定性有多大，来自哪里**？",
        f"本节的正式口径：切分点 M = {m}，EOL 阈值 τ = {EOL_THRESHOLD_AH:g} Ah"
        f"（额定的 {EOL_THRESHOLD_AH / RATED_CAPACITY_AH:.0%}）。",
        "",
        "## 二、三层的定义（先钉死，不许事后调）",
        "",
        "| 层 | 问的是 | 怎么量 | 为什么这么量 |",
        "|---|---|---|---|",
        f"| **定义层** | 「坏」的标准定在哪 | τ = {EOL_THRESHOLD_AH:g} ± {TAU_HALF:g} Ah 窗内 EOL 的**极差** | "
        "τ 是工程惯例不是物理常数；用**无模型**扫描（原始序列上找穿越点），不掺模型外推 |",
        "| **模型形式层** | 假设了哪种函数形式 | M1 / M2 / M4 三个 EOL 的**半极差** | "
        "见下方「这一层全部来自 M2」的说明 |",
        "| **数据层** | 噪声与参数估不准 | M4 后验 90% 区间的**半宽** | "
        "含残差噪声 + 趋势参数误差（W11 方差分解），不含前两层 |",
        "",
        "**共模成分为什么不单列**：W6 发现跨电池存在同步波动（同批三块共模占残差方差 82.4%），"
        "但那是**跨个体**的现象；在逐个体模型里它已经被残差项整体吸收。"
        "再单列一层会把同一份方差算两次。",
        "",
        "### ⚠️ 这一层的宽度**全部来自 M2**（第七轮审查抓出来的要害）",
        "",
        "M1 线性与 M4 趋势+GPR 的 EOL 四个数**完全相同**（146/94/159/97）——"
        "这是 W11 的结论：GP 在数据外归零，点预测就是趋势线。所以：",
        "",
        f"- 「不含 M2」的模型层半极差：**{tbl['模型层半极差(不含M2)'].min():.1f}–"
        f"{tbl['模型层半极差(不含M2)'].max():.1f}**（恒为 0，已写成断言）；",
        f"- 「含 M2」的模型层半极差：{tbl['模型层半极差'].min():.1f}–{tbl['模型层半极差'].max():.1f}。",
        "",
        "而 W14 的逐点配对检验判定：**M2 在 B0005 / B0006 / B0007 三块上显著更差**"
        "（两判据一致），只在 B0018 上无差别。",
        "用测试集把它剔掉确实有泄漏嫌疑，所以本节**照样计入**——"
        "但这一层的正确读法是「**若把双指数也当作先验可信的形式之一，形式之差的上界有多大**」，"
        "**不是**「三种形式势均力敌」。引用时必须带上这句限定。",
        "",
        "## 三、三层各自的数值（循环）",
        "",
        tbl[["电池", "定义层口径"]].to_markdown(index=False),
        "",
        "（B0007 的下端 1.35 Ah 低于它自己的最低容量 1.4005 Ah，"
        f"完整窗口内的极差在 W7 里只能写成下界「>{full_short:.0f}」；本节要一个能进预算的数，"
        f"所以改用**可达窗口** {b7_window} 得 **{b7_short:.0f}**，并在上表写明口径。"
        "W7 还给了同宽替代（窗口平移到 [1.4005, 1.5005]，极差 40）——"
        "**短尺不与长尺比**：若改用同宽，B0007 的「定义+模型 ÷ 数据」会从 "
        f"{ratio_short:.2f} 变成 {ratio_equal:.2f}。"
        "主结论因此只用「定义层在每块都不小」这种不依赖跨块比值的陈述，避开这个陷阱。）",
        "",
        disp3.round(2).to_markdown(index=False),
        "",
        "读法：**半宽** = 该层单独能造成的偏差幅度。三层不能直接相加当误差棒看，"
        "它们一个来自惯例、一个来自形式选择、一个来自统计，性质不同——见第四节。",
        "",
        "## 四、合并：两个口径，各自说清含义",
        "",
        f"- **RSS**：`sqrt(定义² + 模型² + 数据²)` —— 只有当三层互不相关时才成立，"
        "对系统性偏差**乐观**；",
        f"- **SUM**：`定义 + 模型 + 数据` —— 假设三层朝同一方向偏到底，对决策**保守**。",
        "",
        disp4.round(2).to_markdown(index=False),
        "",
        f"**覆盖情况**（{n_label} 块有真值的电池）：",
        "",
        # 覆盖数的说法必须由数决定，不许写死「仍会漏」——第一次跑出来就是 3/3
        (
            f"- **RSS 覆盖 {cover_rss}/{n_label}** —— 在这份数据上，即便按乐观的独立假设，"
            "三段合起来也已经够宽。但这**不证明**三层相互独立："
            "它只说明这次的误差还没大到穿出去。样本只有 "
            f"{n_label} 块，覆盖率的置信区间几乎覆盖 [0, 1]，不能当校准结论。"
            if cover_rss == n_label else
            f"- **RSS 覆盖 {cover_rss}/{n_label}** —— 按独立假设合并**仍然漏**，"
            "说明三层有共同方向、不是独立随机项。"
        ),
        (
            f"- **SUM 覆盖 {cover_sum}/{n_label}** —— 最坏情况口径同样覆盖（它永远比 RSS 宽）。"
            "给运维建议时该用这个口径：**宁可早换，不要晚换**。"
            if cover_sum == n_label else
            f"- **SUM 覆盖 {cover_sum}/{n_label}** —— 连最坏情况口径都没盖住，"
            "说明还有一层没进预算（要回去找）。"
        ),
        "",
        "⚠️ 这里**不能**说「区间校准好了」：只有 "
        f"{n_label} 块有真值的电池，覆盖率是个位数级别的计数，"
        "没有任何统计意义。真正的校准（PICP / MPIW、名义 vs 实际）在 W18。",
        "",
        "## 五、M4 的区间为什么窄",
        "",
        "M4 的 90% 区间只含**数据层**"
        f"（半宽 {tbl['数据层半宽'].min():.1f}–{tbl['数据层半宽'].max():.1f} 循环），"
        "而真实误差还要叠加定义层与模型形式层。逐块对照：",
        "",
        tbl[["电池", "误差", "数据层半宽", "定义层半宽", "模型层半极差", "rss", "sum"]]
        .round(1).to_markdown(index=False),
        "",
        "**「先消灭哪一层」不能一句话概括**——逐块看，最小的那一层并不固定："
        + "；".join(
            f"{r['电池']} 最小的是 "
            + min(
                (("定义层", r["定义层半宽"]), ("模型层", r["模型层半极差"]),
                 ("数据层", r["数据层半宽"])),
                key=lambda kv: kv[1],
            )[0]
            for _, r in tbl.iterrows()
        )
        + "。",
        "",
        f"能说的是：**定义层在每一块都不小（{tbl['定义层半宽'].min():.1f}–"
        f"{tbl['定义层半宽'].max():.1f} 循环）**，而它按定义就不该由模型负责——"
        "它取决于你把「坏」的标准定在哪。所以在调模型之前先把阈值口径写清楚，"
        "否则调得再准，结论仍然只是一个「相对某个惯例」的数。",
        "",
        "（第一版这里写的是「定义层+模型层是数据层的 1.6–3.7 倍」——那个比值"
        "**完全由 M2 撑着**：剔掉 M2 之后模型层恒为 0，比值变成 0.55–3.25，结论不成立。"
        "审查 R9-01 抓出来后已删，改用上面这种逐块陈述。）",
        "",
        "## 六、哪一层最大？",
        "",
        f"- 逐块看**单层最大**的是：" + "、".join(f"{k} {v} 块" for k, v in counts.items() if v) + "；",
        f"- 合并后最宽的是 **{widest['电池']}**（SUM = {widest['sum']:.1f} 循环，"
        f"其中定义层 {widest['定义层半宽']:.1f} / 模型层 {widest['模型层半极差']:.1f} / "
        f"数据层 {widest['数据层半宽']:.1f}）。",
        "",
        "## 六、这张表不能怎么读",
        "",
        "1. **不能把 SUM 当 95% 区间**：它是**最坏情况**界，不是概率陈述；"
        "报告里出现「95%」只能指数据层那一个。",
        f"2. **不能因为 SUM 覆盖了真值就说模型准**：SUM 宽 "
        f"{tbl['sum'].min():.0f}–{tbl['sum'].max():.0f} 循环，"
        "覆盖是保守的代价，不是精度的证据。",
        "3. **不能拿这份覆盖率和 W18 的 PICP 比**：这里 n = "
        f"{n_label}，W18 用的是逐点区间、样本量大得多。",
        "4. **定义层的 ±0.05 Ah 是政策口径**，换成 ±0.1 Ah 极差会更大（W7 有全表）；"
        "引用时必须带上这个窗口宽度。",
        "",
    ]
    return "\n".join(out)


def build_budget(df: pd.DataFrame, prof: pd.DataFrame, m: int) -> pd.DataFrame:
    """算出三层预算表（不写文件）。供 W19 复用，避免两处各算一遍。"""
    import models_gpr
    import models_parametric
    import sensitivity

    series = {cell: (sub["cycle"].to_numpy(float), sub["capacity_Ah"].to_numpy(float))
              for cell, sub in split_by_battery(df).items()}

    param = models_parametric.run(df, prof, m)
    gpr = models_gpr.run(df, prof, m)

    tbl = pd.DataFrame({"电池": sorted(series)})
    tbl["点估计"] = tbl["电池"].map(
        dict(zip(gpr["table"]["电池"], pd.to_numeric(gpr["table"]["EOL预测"], errors="coerce")))
    )
    tbl["真值"] = tbl["电池"].map(
        dict(zip(gpr["table"]["电池"], pd.to_numeric(gpr["table"]["EOL真值"], errors="coerce")))
    )
    defn = layer_definition(series)
    cross_check_definition(defn, series)  # 对账在调用方（见 docstring 的说明）
    tbl["定义层极差"] = tbl["电池"].map({k: v[0] for k, v in defn.items()})
    tbl["定义层口径"] = tbl["电池"].map({k: v[1] for k, v in defn.items()})
    tbl = tbl.merge(layer_model(param["table"], gpr["table"]), on="电池", how="left")
    tbl = tbl.merge(layer_data(gpr["fits"], sorted(series)), on="电池", how="left")
    tbl = combine(tbl)
    assert_combined(tbl)  # 检查在调用方，不在 combine 里面（变异才抓得到）
    assert_model_layer_source(tbl)

    # 短窗口电池的「完整窗口极差」另算一列：它进 md 的散文，必须由代码生成（R9-08）
    for cell in tbl.loc[tbl["定义层口径"].astype(str).str.contains("下端删失"), "电池"]:
        cap = series[cell][1]
        hi = sensitivity.eol_at_exact(cap, EOL_THRESHOLD_AH + TAU_HALF)
        tbl.loc[tbl["电池"] == cell, "定义层完整窗口极差"] = float(len(cap) - hi)
    return tbl


def main() -> int:
    ensure_dirs()
    raw = load_raw()
    df = mark_rows(raw)
    prof = cell_profile(df)

    tbl = build_budget(df, prof, SPLIT_M)

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(CSV_OUT, index=False, encoding="utf-8-sig")
    TABLE_MD.write_text(render_md(tbl, SPLIT_M), encoding="utf-8")
    fig_path = fig13(tbl)

    print(f"三层不确定性（循环）：M = {SPLIT_M}，τ 窗口 ±{TAU_HALF:g} Ah")
    print(tbl[["电池", "点估计", "真值", "定义层半宽", "模型层半极差", "数据层半宽",
               "rss", "sum"]].round(1).to_string(index=False))
    n_label = int(tbl["真值"].notna().sum())
    print()
    print(f"覆盖：RSS {int(tbl.loc[tbl['真值'].notna(), 'RSS 覆盖'].sum())}/{n_label}，"
          f"SUM {int(tbl.loc[tbl['真值'].notna(), 'SUM 覆盖'].sum())}/{n_label}")
    print(f"已写出：{TABLE_MD.name} / {CSV_OUT.name} / {fig_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
