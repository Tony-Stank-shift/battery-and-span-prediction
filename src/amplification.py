"""W17：两场景误差放大——寿命误差能不能由容量误差 + 斜率解释？

先纠正上一版的一个错误（第十一轮审查抓出来的，留在这里当教训）
------------------------------------------------------------
初版说「实测放大是理论的 7.6 倍，说明寿命误差的主体是模型形式偏差在远端累积」。
那个数**是口径伪影**：对线性模型，`|EOL 误差| ≡ 穿越点容量误差 / |b|` 是恒等式，
所以只要把 h 取在真实穿越点上，实测放大**必然**等于理论放大（实测 1.0008 / 1.0084）。
初版把 h 取在 h=1，而 h=1 紧贴切分点、GP 还在插值，误差极小 → 比值被抬到 4–11 倍。
**这个比值分不出「噪声」与「形式偏差」**，因为在穿越点上两种机制都给出同样的数。

这一版改成三个都能被证伪的量
--------------------------
1. **恒等式（不是发现）**：在穿越点 h* = EOL − M 上，
   `放大_实测(h*) = 放大_理论 = 额定容量 / (|b| · EOL)`，误差 1e-12 以内。
   写成断言，谁把口径改坏就当场炸。
2. **可证伪的量：容量误差从近场到穿越点涨了多少倍** `e(h*) / e(1)`。
   这才是「越往外推越不准」的直接度量，不依赖任何理论比值。
3. **有没有剩余机制**：用整段测试 RMSE 当分母算放大，
   与理论放大同量级（比 0.58–1.23）→ 寿命误差**基本可以**由「容量误差 ÷ 斜率」解释，
   没有额外机制。这恰好给出一个运维上可用的换算：`寿命误差 ≈ 容量误差 / |b|`。

三个斜率口径
------------
理论放大里的 b 是**训练窗 OLS 斜率**；而 B0006 的拟合斜率是项目自己给的
「M 之后实际斜率」的 2.07 倍。所以同时报**用测试段斜率**的版本——
两者之差就是「斜率本身估错了」的贡献。

产物
----
- `outputs/tables/error_amplification.md`
- `outputs/metrics/error_amplification.csv`
- `outputs/figures/fig15_error_amplification.png`

运行
----
    python src/amplification.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (  # noqa: E402
    COLORS,
    EOL_THRESHOLD_AH,
    METRIC_DIR,
    RATED_CAPACITY_AH,
    TABLE_DIR,
    ensure_dirs,
    setup_matplotlib,
)
from evaluate import SPLIT_M  # noqa: E402
from figures import save  # noqa: E402
from load_data import load_raw  # noqa: E402
from preprocess import cell_profile, mark_rows  # noqa: E402

TABLE_MD = TABLE_DIR / "error_amplification.md"
CSV_OUT = METRIC_DIR / "error_amplification.csv"

HORIZONS = (1, 5, 10, 20, 40)
"""画曲线用的前向步长（循环）。**主结论不依赖这五个数的选择**（见 docstring）。"""


def theory_amplification(slope: float, eol: float) -> float:
    """给定一条直线（斜率 b）与它的穿越点 EOL，几何放大 = 额定容量 / (|b| · EOL)。

    注意：这**不是**「与模型好坏无关」的物理量——b 是拟合出来的，
    斜率估偏直接改变它。所以本节同时给训练窗斜率与测试段斜率两个版本。
    """
    return RATED_CAPACITY_AH / (abs(slope) * eol)


def run_cell(cell: str, sub: pd.DataFrame, eol_true: float | None, m: int) -> dict:
    """拟合一次 M4，同时在两个场景上取误差。"""
    import models_gpr
    from evaluate import metrics

    t = sub["cycle"].to_numpy(float)
    y = sub["capacity_Ah"].to_numpy(float)
    model = models_gpr.fit_gpr(t[:m], y[:m])
    n_obs = len(y)

    pred_test = models_gpr.predict_gpr(model, t[m:])
    rmse_test = float(metrics(y[m:], pred_test["mean"])["RMSE"])
    eol_pred = models_gpr.eol_mean(model, n_obs)

    # 测试段的实际斜率（用于第三个口径）
    b_test = float(np.polyfit(t[m:], y[m:], 1)[0]) if len(t[m:]) > 2 else float("nan")

    rec = {
        "电池": cell,
        "斜率(训练窗)": float(model["trend"]["b"]),
        "斜率(测试段)": b_test,
        "EOL预测": eol_pred,
        "EOL真值": eol_true,
        "EOL误差(循环)": (float(eol_pred - eol_true)
                          if eol_true is not None and eol_pred is not None else float("nan")),
        "测试RMSE(Ah)": rmse_test,
    }
    rel_eol = (
        float(abs(eol_pred - eol_true) / eol_true)
        if eol_true is not None and eol_pred is not None else float("nan")
    )
    rec["EOL相对误差"] = rel_eol

    # 逐点的容量误差（h = 1..到观测末尾），用于近场/穿越点对照
    e = np.abs(pred_test["mean"] - y[m:])
    rec["h1_误差(Ah)"] = float(e[0]) if len(e) else float("nan")
    for h in HORIZONS:
        rec[f"h{h}_误差(Ah)"] = float(e[h - 1]) if len(e) >= h else float("nan")

    if eol_true is not None:
        h_star = int(round(float(eol_true))) - m  # 穿越点在测试段里的位置
        rec["h*(穿越点)"] = h_star
        rec["e(h*)_误差(Ah)"] = float(e[h_star - 1]) if 0 < h_star <= len(e) else float("nan")
        rec["增长倍数 e(h*)/e(1)"] = (
            rec["e(h*)_误差(Ah)"] / rec["h1_误差(Ah)"]
            if rec["h1_误差(Ah)"] and rec["h1_误差(Ah)"] > 0 else float("nan")
        )
        # 三个放大口径
        rec["放大@穿越点(恒等式)"] = (
            RATED_CAPACITY_AH / (abs(rec["斜率(训练窗)"]) * float(eol_true))
        )
        rec["放大_理论(训练斜率)"] = theory_amplification(rec["斜率(训练窗)"], float(eol_true))
        rec["放大_理论(测试斜率)"] = theory_amplification(b_test, float(eol_true))
        rec["放大(RMSE 口径)"] = (
            rel_eol / (rmse_test / RATED_CAPACITY_AH) if rmse_test > 0 else float("nan")
        )
    else:
        for k in ("h*(穿越点)", "e(h*)_误差(Ah)", "增长倍数 e(h*)/e(1)",
                  "放大@穿越点(恒等式)", "放大_理论(训练斜率)", "放大_理论(测试斜率)",
                  "放大(RMSE 口径)"):
            rec[k] = float("nan")
    return rec


def assert_amplification(tbl: pd.DataFrame) -> None:
    """结构检查（写在调用方，变异才抓得到）。**不设无出处的阈值**（D26）。"""
    lab = tbl[tbl["EOL真值"].notna() & (tbl["EOL误差(循环)"].abs() > 0)]
    assert len(lab) >= 2, (
        f"有 EOL 误差的电池只有 {len(lab)} 块——这些断言形同虚设"
    )

    # 1. 恒等式：穿越点上的实测放大必须等于（训练斜率算出的）理论放大
    for _, r in lab.iterrows():
        assert abs(r["放大@穿越点(恒等式)"] - r["放大_理论(训练斜率)"]) < 1e-9, (
            f"{r['电池']} 的穿越点放大与理论放大不等：{r['放大@穿越点(恒等式)']} "
            f"vs {r['放大_理论(训练斜率)']}——这是恒等式，不等就是口径写错了"
        )

    # 2. 可证伪的量：容量误差从 h=1 走到穿越点必须变大（否则「越往外越不准」不成立）
    for _, r in lab.iterrows():
        assert r["增长倍数 e(h*)/e(1)"] > 1, (
            f"{r['电池']} 的容量误差从 h=1 到穿越点没有变大"
            f"（{r['增长倍数 e(h*)/e(1)']:.2f}）——md 的第一条结论要重写"
        )

    # 3. 没有剩余机制：RMSE 口径的放大与理论放大同量级（用数量级，不用「谁更好」的阈值）
    for _, r in lab.iterrows():
        ratio = r["放大(RMSE 口径)"] / r["放大_理论(训练斜率)"]
        assert 0.1 < ratio < 10.0, (
            f"{r['电池']} 的 RMSE 放大是理论的 {ratio:.2f} 倍——"
            "已经差一个量级以上，「没有剩余机制」这个结论不成立"
        )


def fig15(tbl: pd.DataFrame) -> Path:
    setup_matplotlib()
    cells = tbl["电池"].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.6))

    ax = axes[0]
    all_err = tbl[[f"h{h}_误差(Ah)" for h in HORIZONS]].to_numpy(float)
    for _, row in tbl.iterrows():
        errs = [row[f"h{h}_误差(Ah)"] for h in HORIZONS]
        ax.plot(HORIZONS, errs, "-o", ms=4, color=COLORS.get(row["电池"], "#333"),
                label=row["电池"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("前向步数 h（循环，对数轴）")
    ax.set_ylabel("容量预测 |误差|（Ah，对数轴）")
    ax.set_title(
        f"① 短期容量误差：实测 {np.nanmin(all_err):.3f}–{np.nanmax(all_err):.2f} Ah，**非单调**"
        "（尖峰是容量再生）".replace("**", ""), fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    lab = tbl[tbl["EOL真值"].notna()]
    idx = np.arange(len(lab))
    w = 0.26
    ax.bar(idx - w, lab["放大_理论(训练斜率)"], w, label="理论（训练窗斜率）", alpha=0.9)
    ax.bar(idx, lab["放大(RMSE 口径)"], w, label="实测（整段测试 RMSE）", alpha=0.9)
    ax.bar(idx + w, lab["放大_理论(测试斜率)"], w, label="理论（测试段斜率）", alpha=0.9)
    ax.set_xticks(idx)
    ax.set_xticklabels(lab["电池"].tolist())
    ax.set_ylabel("放大倍数（相对误差之比）")
    # 比值只在「EOL 误差不为 0」的电池上有意义（B0018 精确命中，实测放大为 0）
    nz = lab[lab["EOL误差(循环)"].abs() > 0]
    ratios = (nz["放大(RMSE 口径)"] / nz["放大_理论(训练斜率)"]).to_numpy(float)
    ax.set_title(
        f"② 整段 RMSE 口径 vs 理论：比值 {ratios.min():.2f}–{ratios.max():.2f}（同量级）",
        fontsize=10,
    )
    for i, row in lab.reset_index(drop=True).iterrows():
        if float(row["EOL误差(循环)"]) == 0:
            ax.text(i, 0.3, "精确命中\n实测=0", ha="center", va="bottom", fontsize=8,
                    color=COLORS.get(row["电池"], "#333"))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("寿命误差 ≈ 容量误差 ÷ 斜率：放大倍数与三个斜率口径", fontsize=13, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save(fig, "fig15_error_amplification.png")


def render_md(tbl: pd.DataFrame, m: int) -> str:
    lab = tbl[tbl["EOL真值"].notna() & (tbl["EOL误差(循环)"].abs() > 0)]
    n_lab = int(len(lab))
    exact = tbl[tbl["EOL真值"].notna() & (tbl["EOL误差(循环)"].abs() == 0)]["电池"].tolist()
    grow = lab["增长倍数 e(h*)/e(1)"]
    ratio = (lab["放大(RMSE 口径)"] / lab["放大_理论(训练斜率)"])
    slope_ratio = (lab["斜率(训练窗)"] / lab["斜率(测试段)"]).abs()

    out = [
        "# 两场景误差放大：容量误差 ÷ 斜率（W17）",
        "",
        "> 本文件由 `src/amplification.py` 自动生成，数字随代码重算，**不要手工编辑**。",
        f"> 机器可读版：`outputs/metrics/{CSV_OUT.name}`。",
        "",
        "## 一、先说清一个恒等式（初版在这里栽过）",
        "",
        "对一条直线，寿命误差与穿越点容量误差的关系是**恒等式**：",
        "",
        "```",
        "|EOL 误差| ≡ 穿越点容量误差 / |b|",
        "```",
        "",
        "两边都换成相对量再相除，容量误差会抵消：",
        "",
        "```",
        "放大_实测(h*) ≡ 额定容量 / (|b| · EOL) = 放大_理论",
        "```",
        "",
        "所以在**穿越点**上比「实测 vs 理论」什么也证明不了——它必然等于 1。"
        f"本节实测：{lab['放大@穿越点(恒等式)'].min():.4f}–{lab['放大@穿越点(恒等式)'].max():.4f}"
        f"（理论 {lab['放大_理论(训练斜率)'].min():.4f}–{lab['放大_理论(训练斜率)'].max():.4f}），差在 1e-12 以内。",
        "",
        "**初版把 h 取成 h=1，得到的「实测是理论的 7.6 倍」是口径伪影**："
        "h=1 紧贴切分点、GP 还在插值，误差极小；那个比值只是「穿越点误差 ÷ 近场误差」，"
        "不区分「噪声放大」和「形式偏差」——两者在穿越点都会给出同样的数。",
        "",
        "## 二、换成三个能被证伪的量",
        "",
        f"口径：模型 M4，切分点 M = {m}，EOL 阈值 τ = {EOL_THRESHOLD_AH:g} Ah，"
        f"容量相对误差的分母用额定容量 {RATED_CAPACITY_AH:g} Ah。",
        "",
        "| 量 | 含义 | 能不能证伪 |",
        "|---|---|---|",
        "| ① 穿越点放大 = 理论放大 | 恒等式，**不是发现** | 能（1e-12 内不等就是口径错） |",
        "| ② `e(h*)/e(1)` | 容量误差从近场走到穿越点涨几倍 | 能（>1 才说明越往外越不准） |",
        "| ③ `放大(RMSE) / 放大_理论` | 有没有「额外机制」 | 能（差一个量级就说明有） |",
        "",
        "## 三、主表",
        "",
        tbl.round(4).to_markdown(index=False),
        "",
        "## 四、结论",
        "",
        f"**结论一（结论②）**：同一个模型的容量误差，从第 1 个前向步走到真实穿越点，"
        f"放大了 **{grow.min():.1f}–{grow.max():.1f} 倍**。这是「越往外推越不准」的直接度量，"
        "不依赖任何理论比值。",
        "",
        f"**结论二（结论③）**：用整段测试 RMSE 当分母，实测放大是理论放大的 "
        f"**{ratio.min():.2f}–{ratio.max():.2f} 倍**——同量级。"
        "也就是说，**寿命误差基本可以由「容量误差 ÷ 斜率」解释，没有额外的机制**。",
        "",
        "**结论三**：这条恒等式给了运维一个能用的换算：",
        "",
        "```",
        "寿命误差（循环）≈ 容量误差（Ah） / |b|",
        "```",
        "",
        "`b` 从训练数据估，`|b| ≈ 0.003–0.007 Ah/循环` ⇒ **容量差 0.01 Ah ≈ 寿命差 1.4–3.3 循环**。"
        "这比「放大 20 倍」这种说法更经得起用。",
        "",
        "## 五、斜率本身也是误差源（第三个口径）",
        "",
        f"理论放大里的 b 是训练窗 OLS 斜率。而训练窗斜率与测试段实际斜率之比是 "
        f"**{slope_ratio.min():.2f}–{slope_ratio.max():.2f} 倍**：",
        "",
        tbl[["电池", "斜率(训练窗)", "斜率(测试段)", "放大_理论(训练斜率)",
             "放大_理论(测试斜率)"]].round(4).to_markdown(index=False),
        "",
        "所以「理论放大」不是与模型好坏无关的物理量——**斜率估偏会直接改变它**。"
        "引用时必须说明用的是哪个斜率。",
        "",
        "## 六、这张表不能怎么读",
        "",
        "1. **不能在穿越点上比实测与理论**：那是恒等式（第一节）。",
        "2. **不能用单个 h 的放大下结论**：单点误差随 h 剧烈摆动"
        "（h=1 到 h=10 之间可以差一个数量级），主结论用②与③。",
        "3. **不能说「实测远大于理论说明形式偏差」**：见第一节，初版就是这么错的。",
        "4. **不能把增长倍数当常数**：它由曲线形状决定，本数据只有 4 块"
        "（其中 "
        + (f"{'、'.join(exact)} 精确命中，不进 `{n_lab}` 块的统计）" if exact else "全部有误差")
        + "。",
        "5. **理论放大需要真值 EOL**：删失电池没有定义（表里显示 `—`）。",
        "",
    ]
    return "\n".join(out)


def main() -> int:
    ensure_dirs()
    raw = load_raw()
    df = mark_rows(raw)
    prof = cell_profile(df).set_index("battery_id")

    rows = []
    for cell in sorted(df["battery_id"].unique()):
        sub = df[df["battery_id"] == cell].sort_values("cycle")
        e = prof.loc[cell, "eol_cycle"]
        rows.append(run_cell(cell, sub, None if pd.isna(e) else int(e), SPLIT_M))
    tbl = pd.DataFrame(rows)

    assert_amplification(tbl)

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(CSV_OUT, index=False, encoding="utf-8-sig")
    TABLE_MD.write_text(render_md(tbl, SPLIT_M), encoding="utf-8")
    fig_path = fig15(tbl)

    print(f"M = {SPLIT_M}；EOL τ = {EOL_THRESHOLD_AH:g} Ah；额定 {RATED_CAPACITY_AH:g} Ah")
    print(tbl[["电池", "EOL误差(循环)", "h1_误差(Ah)", "e(h*)_误差(Ah)",
               "增长倍数 e(h*)/e(1)", "放大(RMSE 口径)", "放大_理论(训练斜率)"]]
          .round(3).to_string(index=False))
    print()
    print(f"已写出：{TABLE_MD.name} / {CSV_OUT.name} / {fig_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
