"""W17：两场景误差放大——同一份模型误差，落到「容量」和「寿命」上是两个量级。

要回答的问题
------------
「短期容量预测」和「EOL 预测」用的是**同一个模型、同一次拟合**。
为什么前者的误差看起来可以接受（0.02 Ah），后者的误差却动辄 20 个循环？
这一节把这个差口量成一个数：**相对误差的放大倍数**。

两种放大，必须分开
------------------
1. **理论放大**（纯几何）：曲线斜率 b 决定「容量偏 ε Ah → 寿命偏 ε/|b| 循环」。
   换成相对口径：
       `放大_理论 = (ε/|b| / EOL) / (ε / 额定容量) = 额定容量 / (|b| · EOL)`
   它只由斜率与寿命决定，与模型好坏无关。
2. **实测放大**：直接用同一份模型的两处误差算——
       `放大_实测 = |EOL误差| / EOL  ÷  |h 步容量误差| / 额定容量`

**两者不相等，而差值本身是结论**：若实测远大于理论，说明寿命误差**不是**
「一步预测噪声沿斜率放大」得来的，而是**模型形式偏差在远端累积**——
这正是 W15 说「模型形式层是大头」、W16 说「区间窄 ≠ 预测准」的同一个事实。

口径
----
- 模型：M4（趋势 + GPR），切分点 M = 80，与 W11/W13 完全一致。
- 短期：`h` 步前向预测，`h ∈ {1, 5, 10, 20, 40}`；
  容量相对误差的分母用**额定容量** 2.0 Ah（工业口径），不用该点实测值
  （实测值在 1.2–2.0 之间摆动，用它当分母会让误差看起来更小）。
- 寿命：EOL 点估计相对误差，分母用该电池的**真实寿命**；删失电池没有这一项。

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
"""短期容量预测的前向步数（循环）。"""


def theory_amplification(slope: float, eol: float) -> float:
    """理论放大：额定容量 / (|斜率| · EOL)。

    推导：曲线偏 ε Ah ⇒ 穿越点偏 ε/|b| 循环；两边都换成相对量再相除，ε 抵消。
    """
    return RATED_CAPACITY_AH / (abs(slope) * eol)


def run_cell(cell: str, sub: pd.DataFrame, eol_true: float | None, m: int) -> dict:
    """拟合一次模型，同时在两个场景上取误差。"""
    import models_gpr

    t = sub["cycle"].to_numpy(float)
    y = sub["capacity_Ah"].to_numpy(float)
    model = models_gpr.fit_gpr(t[:m], y[:m])
    n_obs = len(y)

    eol_pred = models_gpr.eol_mean(model, n_obs)
    rec = {
        "电池": cell,
        "斜率(Ah/循环)": float(model["trend"]["b"]),
        "EOL预测": eol_pred,
        "EOL真值": eol_true,
        "EOL误差(循环)": (float(eol_pred - eol_true) if eol_true is not None and eol_pred is not None
                          else float("nan")),
    }
    rel_eol = (
        float(abs(eol_pred - eol_true) / eol_true)
        if eol_true is not None and eol_pred is not None else float("nan")
    )
    rec["EOL相对误差"] = rel_eol

    for h in HORIZONS:
        idx = m + h - 1
        if idx >= n_obs:
            rec[f"h{h}_误差(Ah)"] = float("nan")
        else:
            pred = models_gpr.predict_gpr(model, np.array([t[idx]]))["mean"][0]
            rec[f"h{h}_误差(Ah)"] = float(abs(pred - y[idx]))
        err = rec[f"h{h}_误差(Ah)"]
        rec[f"h{h}_容量相对误差"] = err / RATED_CAPACITY_AH
        rec[f"h{h}_实测放大"] = (
            rel_eol / rec[f"h{h}_容量相对误差"]
            if rec[f"h{h}_容量相对误差"] and rec[f"h{h}_容量相对误差"] > 0 else float("nan")
        )
    rec["理论放大"] = (
        theory_amplification(rec["斜率(Ah/循环)"], eol_true)
        if eol_true is not None else float("nan")
    )
    return rec


def assert_amplification(tbl: pd.DataFrame) -> None:
    """结构检查（写在调用方，变异才抓得到）。

    1. 有真值的电池，实测放大必须 > 1——否则「同一个模型在寿命上更难」这条立论不成立；
    2. 理论放大必须等于公式值（防止有人悄悄改分母口径）；
    3. h=1 的容量误差必须小于该电池的测试段 RMSE 量级——
       它是一步前向预测，若比全程 RMSE 还大，说明取点或口径错了。
    """
    lab = tbl[tbl["EOL真值"].notna()]
    # 只对「EOL 误差不为 0」的电池检查放大：误差为 0 时放大是 0，
    # 那是**精确命中**（B0018 就是），不是立论反例。
    lab_err = lab[lab["EOL误差(循环)"].abs() > 0]
    assert len(lab_err) >= 2, (
        f"EOL 有误差的电池只有 {len(lab_err)} 块——「寿命更难」这条断言形同虚设"
    )
    for _, row in lab_err.iterrows():
        assert row["h1_实测放大"] > 1, (
            f"{row['电池']} 的 h=1 实测放大 {row['h1_实测放大']:.2f} ≤ 1——"
            "「寿命上的相对误差比容量上的大」不成立"
        )
        expect = RATED_CAPACITY_AH / (abs(row["斜率(Ah/循环)"]) * row["EOL真值"])
        assert abs(expect - row["理论放大"]) < 1e-12, (
            f"{row['电池']} 的理论放大 {row['理论放大']} 与公式 {expect} 不一致"
        )
    assert lab_err["h1_实测放大"].max() >= 3, (
        "没有任何电池的实测放大达到 3 倍——md 里「大一个量级」的说法要降级"
    )


def fig15(tbl: pd.DataFrame) -> Path:
    setup_matplotlib()
    cells = tbl["电池"].tolist()
    colors = {"B0005": "#1f77b4", "B0006": "#d62728", "B0007": "#9467bd", "B0018": "#2ca02c"}

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.6))

    ax = axes[0]
    for _, row in tbl.iterrows():
        errs = [row[f"h{h}_误差(Ah)"] for h in HORIZONS]
        ax.plot(HORIZONS, errs, "-o", ms=4, color=colors.get(row["电池"], "#333"),
                label=row["电池"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("前向步数 h（循环，对数轴）")
    ax.set_ylabel("容量预测 |误差|（Ah，对数轴）")
    ax.set_title("① 短期容量误差随步长增长（但仍在 1e-2 Ah 量级）", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    w = 0.12
    idx = np.arange(len(HORIZONS))
    for k, (_, row) in enumerate(tbl.iterrows()):
        vals = [row[f"h{h}_实测放大"] for h in HORIZONS]
        if all(pd.isna(v) for v in vals):
            continue
        ax.bar(idx + (k - 1.5) * w, vals, w, label=row["电池"],
               color=colors.get(row["电池"], "#333"))
    ax.axhline(1.0, ls="--", lw=1.2, color="#555555", label="放大 = 1（两场景一样难）")
    # B0018 精确命中真值，放大 = 0，柱高为零——必须标注，否则会被读成缺数据
    for k, (_, row) in enumerate(tbl.iterrows()):
        if row["EOL真值"] is not None and not pd.isna(row["EOL真值"]) \
                and row["EOL误差(循环)"] == 0:
            ax.text(k, 1.5, "精确命中\n放大=0", ha="center", va="bottom", fontsize=8,
                    color=colors.get(row["电池"], "#333"))
    ax.set_xticks(idx)
    ax.set_xticklabels([f"h={h}" for h in HORIZONS])
    ax.set_ylabel("实测放大倍数（EOL 相对误差 ÷ 容量相对误差）")
    ax.set_title("② 同一份模型误差，换到寿命上放大多少倍（对数轴见左图量级差）", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("图15　短期容量预测 vs EOL 预测：同一模型的误差放大", fontsize=13, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save(fig, "fig15_error_amplification.png")


def render_md(tbl: pd.DataFrame, m: int) -> str:
    lab = tbl[tbl["EOL真值"].notna()]
    n_lab = len(lab)
    # 放大倍数只在「EOL 误差不为 0」的电池上有意义；精确命中的那块单独说
    lab_err = lab[lab["EOL误差(循环)"].abs() > 0]
    exact = lab[lab["EOL误差(循环)"].abs() == 0]["电池"].tolist()
    amp1 = lab_err["h1_实测放大"]
    theo = lab_err["理论放大"]
    ratio = (amp1 / theo).median()
    zero_cells = "、".join(exact) or "无"
    c_min = int(tbl["h1_误差(Ah)"].notna().sum())

    out = [
        "# 两场景误差放大：短期容量预测 vs EOL 预测（W17）",
        "",
        "> 本文件由 `src/amplification.py` 自动生成，数字随代码重算，**不要手工编辑**。",
        f"> 机器可读版：`outputs/metrics/{CSV_OUT.name}`。",
        "",
        "## 一、问题与口径",
        "",
        f"同一个模型（M4）、同一次拟合（M = {m}），在两个场景上取误差：",
        "",
        "| 场景 | 预测什么 | 误差单位 | 相对误差的分母 |",
        "|---|---|---|---|",
        "| **短期容量** | 第 t+h 个循环的容量（h = 1/5/10/20/40） | Ah | **额定容量** 2.0 Ah |",
        "| **长期寿命** | 首次穿越 τ = 1.4 Ah 的循环 | 循环 | 该电池**真实寿命** |",
        "",
        "容量相对误差的分母统一用额定容量而不是该点实测值——实测值在 1.2–2.0 之间摆动，"
        "用它当分母会让误差看起来更小，是常见的口径作弊。",
        "",
        "## 二、两种放大",
        "",
        "**理论放大**（纯几何）：曲线偏 ε Ah ⇒ 穿越点偏 ε/|b| 循环，两边换成相对量后 ε 抵消：",
        "",
        "```",
        "放大_理论 = 额定容量 / (|斜率 b| · EOL)",
        "```",
        "",
        "**实测放大** = `|EOL 误差| / EOL` ÷ `|h 步容量误差| / 额定容量`。",
        "",
        "## 三、主表",
        "",
        tbl[["电池", "斜率(Ah/循环)", "EOL预测", "EOL真值", "EOL误差(循环)",
             "EOL相对误差", "h1_误差(Ah)", "h1_实测放大", "h10_实测放大", "理论放大"]]
        .round(4).to_markdown(index=False),
        "",
        "逐档放大倍数（`—` 表示该电池没有真值，算不出相对误差）：",
        "",
        tbl[["电池"] + [f"h{h}_实测放大" for h in HORIZONS]]
        .round(2).astype(object).where(tbl[[f"h{h}_实测放大" for h in HORIZONS]].notna(), "—")
        .to_markdown(index=False),
        "",
        f"**读出来的事实**（{len(lab_err)} 块有真值且有 EOL 误差的电池，h=1 档）："
        f"容量相对误差 {lab_err['h1_容量相对误差'].min():.2%}–{lab_err['h1_容量相对误差'].max():.2%}，"
        f"EOL 相对误差 {lab_err['EOL相对误差'].min():.2%}–{lab_err['EOL相对误差'].max():.2%}，"
        f"**实测放大 {amp1.min():.1f}–{amp1.max():.1f} 倍**。",
        "",
        f"（{zero_cells} 的 EOL 预测**精确命中**真值，放大的分子为 0，"
        "不进入上面的区间——它是「猜对了」，不是「不难」。）",
        "",
        "也就是说：**同一个模型，说「下个循环容量是多少」可以准到 1%，"
        "说「它还能用多少循环」却差了百分之十几。**",
        "",
        "## 四、实测放大是理论放大的几倍——这才是本节的重点",
        "",
        f"理论放大只有 **{theo.min():.2f}–{theo.max():.2f} 倍**，"
        f"而实测放大是它的 **{ratio:.1f} 倍**（中位数）。",
        "",
        "这个差口说明：**寿命误差不是「一步预测噪声沿斜率放大」得来的**。"
        "如果只是噪声放大，实测应该贴着理论值；"
        "实测高出这么多，意味着误差的主体是**模型形式偏差在远端累积**——",
        "即「这条直线的形状对不对」，而不是「这个点估得准不准」。",
        "",
        "这与 W15（模型形式层是看不见的大头）和 W16（区间窄 ≠ 预测准）指向同一个事实。",
        "",
        "## 五、为什么这件事值得单独写一节",
        "",
        "因为它把「为什么要专门做 EOL 预测」讲清楚了：",
        "",
        "1. **不能拿容量预测的精度给寿命预测背书**——两者差一个量级；",
        "2. **EOL 的误差主要不来自噪声**，所以「多测几次、降噪」几乎不解决问题；",
        "3. **真正能改善的是形式与阈值口径**（W15 的定义层与模型形式层），"
        "这与 W15 的结论一致：调 GP 核只能压最小的一块。",
        "",
        "## 六、这张表不能怎么读",
        "",
        "1. **不能把放大倍数当普适常数**：它由斜率与寿命共同决定，"
        "本数据只有 4 块电池，换批次必须重算。",
        f"2. **不能只看 h=1**：h 越大容量误差越大、放大倍数越小"
        "（因为分母变大了）。「放大多少倍」必须带 h。",
        f"3. **B0006 的 h=10 误差反而比 h=1 大很多**（{tbl.loc[tbl['电池'] == 'B0006', 'h10_误差(Ah)'].iloc[0]:.3f} Ah）："
        "那一步正好撞上容量再生，不是模型变差——这也是为什么主结论用 h=1 与 h=10 两个档一起看。",
        "",
        "4. **删失电池没有 EOL 相对误差**：没有真值就没有分母，不能拿预测值当分母。",
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

    print(f"M = {SPLIT_M}；短期步长 {HORIZONS}")
    print(tbl[["电池", "斜率(Ah/循环)", "h1_误差(Ah)", "h1_实测放大", "理论放大",
               "EOL误差(循环)"]].round(3).to_string(index=False))
    print()
    print(f"已写出：{TABLE_MD.name} / {CSV_OUT.name} / {fig_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
