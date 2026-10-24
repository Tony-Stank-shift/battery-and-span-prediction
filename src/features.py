"""W8：早期特征工程 + 未来信息泄漏证明。

两个不同的任务，别混为一谈
--------------------------
本项目里「预测」有两种口径，用的数据结构和特征完全不同：

| | 任务 A：个体内轨迹外推（主线） | 任务 B：跨个体早期预测 |
|---|---|---|
| 做法 | 用某块电池前 M 个循环拟合，外推它后面的容量/RUL | 用某块电池前 K 个循环的统计量，预测它的寿命 |
| `cycle` 的角色 | **回归的自变量（时间轴）**，必用 | **禁止作特征**：它是个体内时间，跨个体不可迁移 |
| 谁用 | M1 线性 / M2 双指数 / M4 趋势+GPR | M3 随机森林（反例实验） |
| 样本量 | 每块电池一条曲线 | **名义 4 块，实际可用标签 3 块** |

**这张表是对 `preprocess.FEATURE_EXCLUDE` 里 `cycle` 那条的补充说明**：
那条禁令针对的是任务 B（跨个体迁移）；在任务 A 里 `cycle` 是 x 轴，不是「特征」。
两者都写清楚，否则报告里「禁止用 cycle」和「模型输入含 cycle」会互相打脸。

「因果」在这里的确切含义
------------------------
每块电池只算**一个**特征向量，时刻固定 t = K，输入数据先截断到 cycle ≤ K。
所以只要「先截断、再算」这个顺序成立，任何特征都不可能碰到未来数据。
泄漏检测就是去**证明这个顺序真的生效**：把 t>K 的行扰动掉，特征必须一格不变。

⚠️ 任务 B 名义 4 块电池，实际只有 3 个标签
-------------------------------------------
B0007 在 τ = 1.4 Ah 下**删失**（观察期内没到达 EOL），它没有 `eol_cycle`，
所以相关性只能在 3 块上算。n = 3 时 df = 1，Pearson r 在 α=0.05 双侧的临界值是
**|r| = 0.997**——除了三个点几乎完全单调，什么都算不上显著；
而「三个点几乎完全单调」恰恰是排序巧合的同义词（W5 已被这个坑咬过一次：
「σ 最大的两块恰好是再生最多的两块」，p≈0.08，是 n=4 时的巧合）。

所以本模块**不做特征筛选**，只做三件事：
1. 构造**因果**特征（每个特征只用 t ≤ K 的数据），逐条写清可得性；
2. **证明**没有未来信息泄漏——用「扰动未来 + 负控制」两段式实验，而不是嘴上保证；
3. 把与 EOL 的相关**如实列出并标注它没有推断意义**，供 W13 的 LOBOCV 用数据说话。

为什么这里没有图
----------------
3 个点的散点图看不出任何东西，画出来只会让读者以为看到了规律。
本节唯一有信息量的可视化是「K 敏感性」那张表，所以不出图。

运行
----
    python src/features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (  # noqa: E402
    EOL_THRESHOLD_AH,
    RANDOM_SEED,
    REGEN_MIN_JUMP_AH,
    TABLE_DIR,
    TREND_WINDOW,
    ensure_dirs,
)
from load_data import load_raw, split_by_battery  # noqa: E402
from preprocess import cell_profile, mark_rows  # noqa: E402

TABLE_MD = TABLE_DIR / "early_features.md"

K_MAIN = 50
"""任务 B 的主观测窗口：只用前 K 个循环。

取值理由：最短的 B0018 只有 132 循环，K=50 能留下 60% 的后段数据可供（将来的）验证；
同时 50 个点足够稳定地估出斜率（W5 实测 20 点窗口的斜率已很稳）。
**这是一个折中，不是最优值**，所以下面做了 K 敏感性扫描。"""

K_SWEEP = (20, 30, 40, 50, 60, 80, 100)
"""K 敏感性扫描。用来回答「相关方向是否依赖观测窗口长度」。"""

PERTURB_AH = 0.5
"""扰动未来数据的幅度。取 0.5 Ah（额定容量的 25%），远超任何真实波动——
只要特征用到了 t>K 的任何一行，值必然变。"""


# ---------------------------------------------------------------------------
# 因果特征（每条只用 cycle ≤ K 的数据）
# ---------------------------------------------------------------------------
def _linear(sub: pd.DataFrame) -> tuple[float, float]:
    """线性拟合，返回 (斜率 Ah/循环, R²)。

    与 `individual._linear` 数学上相同，但**服务对象不同**：那边是描述整条曲线，
    这边必须只在 cycle ≤ K 的数据上调用。故意各写一份，避免「改一处影响两处」。
    """
    x = sub["cycle"].to_numpy(float)
    y = sub["capacity_Ah"].to_numpy(float)
    if len(x) < 3:
        return float("nan"), float("nan")
    coef = np.polyfit(x, y, 1)
    pred = np.polyval(coef, x)
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(coef[0]), r2


def _curvature(sub: pd.DataFrame) -> float:
    """无量纲曲率 = c·(T/100)/|b|，只在 cycle ≤ K 上估。

    与 `individual._curvature_scaled` 的差别：那边用整条轨迹的 T，
    这边 T 就是 K。**同一个名字、不同的数据范围**，所以必须在各自文件里写清。
    """
    x = sub["cycle"].to_numpy(float) / 100.0
    y = sub["capacity_Ah"].to_numpy(float)
    if len(x) < 3:
        return float("nan")
    c2, c1, _ = np.polyfit(x, y, 2)
    if c1 == 0:
        return float("nan")
    return float(c2 * (x.max()) / abs(c1))


def _trailing_residual_std(sub: pd.DataFrame, window: int = TREND_WINDOW) -> float:
    """残差尺度：相对「尾部滚动中位数」的残差标准差。

    为什么用 trailing（`center=False`）而不是 W5/W6 的居中窗口？

    ⚠️ 先纠正一个我一开始写错的说法：**在本模块的设计里，居中窗口并不构成泄漏。**
    因为本模块对每块电池只算**一个**特征向量，时刻固定在 t = K，
    而输入已经被截断到 cycle ≤ K —— 居中窗口在第 t 行最多用到第 K 行，没有越过 K。
    （我原先写「用居中窗口就是泄漏」，那是把「面板式特征」的规则搬错了地方：
    只有当特征要逐行给出 (cell, t) 且 t 可以小于 K 时，居中窗口才会看到 t 之后的真实数据。）

    用 trailing 是**保守选择**，不是硬要求：它让「特征只依赖 ≤ 当前时刻」这件事
    在代码形态上一目了然，将来若真要做面板式特征不必改定义。
    代价是残差估计的方差略大（只用单侧）。
    """
    y = pd.Series(sub["capacity_Ah"].to_numpy(float))
    trend = y.rolling(window, min_periods=3, center=False).median()
    resid = (y - trend).to_numpy()
    cyc = sub["cycle"].to_numpy(int)
    # 前 window 个点的趋势只看到单侧历史，残差不可靠，剔除
    mask = cyc > window
    core = resid[mask]
    if core.size < 3:
        return float("nan")
    return float(np.nanstd(core))


def build_features_one(sub: pd.DataFrame, K: int) -> dict[str, float]:
    """单块电池的前 K 循环特征。**调用方保证传入的 sub 已截断到 cycle ≤ K。**"""
    sub = sub.sort_values("cycle").reset_index(drop=True)
    if int(sub["cycle"].max()) > K:
        raise ValueError(f"传入数据超过 K={K}，特征定义被破坏")
    cap = sub["capacity_Ah"].to_numpy(float)
    cap_first = float(cap[0])
    cap_k = float(cap[-1])
    slope, r2 = _linear(sub)
    tail = sub[sub["cycle"] > K - 20]
    slope_tail, _ = _linear(tail) if len(tail) >= 3 else (float("nan"), float("nan"))
    delta = sub["capacity_Ah"].diff()
    regen = delta[delta > REGEN_MIN_JUMP_AH]

    return {
        "cap_first": cap_first,
        "cap_at_K": cap_k,
        "drop_ratio_to_K": (cap_first - cap_k) / cap_first,
        "slope_to_K_Ah_per_100": slope * 100.0,
        "slope_tail20_Ah_per_100": slope_tail * 100.0,
        "linear_r2_to_K": r2,
        "curvature_to_K": _curvature(sub),
        "resid_std_trailing": _trailing_residual_std(sub),
        "n_regen_to_K": float(len(regen)),
        "max_regen_to_K": float(regen.max()) if len(regen) else 0.0,
    }


FEATURE_DOC: dict[str, str] = {
    "cap_first": "第 1 循环容量。**过去**信息；用了它就等于承认「新电池必须先测一次」，部署前提写进报告",
    "cap_at_K": "第 K 循环容量，即观测窗口末端的水平",
    "drop_ratio_to_K": "(首循环 − 第 K 循环)/首循环，窗口内相对衰减",
    "slope_to_K_Ah_per_100": "1..K 的 OLS 斜率 ×100，窗口内平均衰减速率",
    "slope_tail20_Ah_per_100": "末尾 20 个循环的 OLS 斜率 ×100，看最近是否加速/放缓",
    "linear_r2_to_K": "1..K 的线性拟合 R²，形状是否接近直线",
    "curvature_to_K": "1..K 的无量纲曲率（正=下弯变缓），窗口内的形状",
    "resid_std_trailing": (
        f"相对**尾部**滚动中位数（窗口 {TREND_WINDOW}）的残差标准差，逐循环波动尺度"
    ),
    "n_regen_to_K": f"1..K 内单步回升 > {REGEN_MIN_JUMP_AH} Ah 的次数",
    "max_regen_to_K": "1..K 内最大单步回升幅度（无回升记 0）",
}

# 每条特征「为什么在 t=K 时刻可得」的一句话论证
AVAILABILITY: dict[str, str] = {
    "cap_first": "用第 1 行",
    "cap_at_K": "用第 K 行",
    "drop_ratio_to_K": "用第 1 与第 K 行",
    "slope_to_K_Ah_per_100": "用 1..K 全部行",
    "slope_tail20_Ah_per_100": "用 K−19..K",
    "linear_r2_to_K": "用 1..K 全部行",
    "curvature_to_K": "用 1..K 全部行",
    "resid_std_trailing": "用 1..K 全部行，但滚动中位数**只看过去**（center=False）",
    "n_regen_to_K": "用 1..K 的差分",
    "max_regen_to_K": "用 1..K 的差分",
}


def build_features(df: pd.DataFrame, K: int) -> pd.DataFrame:
    """四块电池各自的前 K 循环特征。"""
    rows = []
    for cell, sub in split_by_battery(df).items():
        head = sub[sub["cycle"] <= K]
        if len(head) < K:
            raise ValueError(f"{cell} 只有 {len(head)} 个循环，不足 K={K}")
        feats = build_features_one(head, K)
        rows.append({"电池": cell, **feats})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 泄漏证明：扰动未来 + 负控制
# ---------------------------------------------------------------------------
def _perturb_future(df: pd.DataFrame, K: int, seed: int) -> pd.DataFrame:
    """把 cycle > K 的容量改成随机值（幅度 PERTURB_AH）。

    注意三点：
    1. 只动 `capacity_Ah`，不动 cycle —— 否则「行数/位置」也变了，测不出纯粹的信息泄漏；
    2. 随机而不是固定加常数 —— 避免「恰好抵消」这种偶然；
    3. `is_regen` / `cap_delta` 等派生列会随之重算，因为它们本来就该跟着容量走。
    """
    rng = np.random.default_rng(seed)
    out = df[["battery_id", "cycle", "capacity_Ah", "ambient_temp_C"]].copy()
    mask = out["cycle"] > K
    n = int(mask.sum())
    if n == 0:
        raise ValueError("没有可扰动的未来行，K 设得太大")
    out.loc[mask, "capacity_Ah"] = (
        out.loc[mask, "capacity_Ah"].to_numpy(float)
        + rng.uniform(-PERTURB_AH, PERTURB_AH, size=n)
    )
    # 派生列（cap_delta / is_regen / cell_censored …）必须跟着重算，
    # 否则扰动只改了容量却留下旧的派生列，检测会假阳性。
    return mark_rows(out)


def _leaky_control(df: pd.DataFrame, K: int) -> pd.DataFrame:
    """**故意泄漏**的对照特征：整条轨迹的最小容量与首次到达 EOL 的循环。

    它们用到了 t>K 的行（甚至用到整条轨迹），所以任何一个合格的泄漏检测都应该抓到。
    如果检测抓不到它们，说明检测本身是空的（vacuous），必须先修检测。
    """
    rows = []
    for cell, sub in split_by_battery(df).items():
        sub = sub.sort_values("cycle")
        cap = sub["capacity_Ah"].to_numpy(float)
        cyc = sub["cycle"].to_numpy(int)
        hit = np.flatnonzero(cap <= EOL_THRESHOLD_AH)
        rows.append(
            {
                "电池": cell,
                "LEAK_cap_min_full": float(cap.min()),
                "LEAK_eol_cycle_full": float(cyc[hit[0]]) if len(hit)
                else float(cyc[-1] + 1),
            }
        )
    return pd.DataFrame(rows)


def leakage_test(df: pd.DataFrame, K: int) -> dict:
    """两段式泄漏检测。

    1. **正实验**：真实特征集在「未来被扰动」后必须**逐个数值不变**；
    2. **负控制**：故意泄漏的对照特征集必须**发生变化**——否则说明检测没有分辨力。

    只做 1 不做 2 的检测，等价于「我保证没泄漏」。
    """
    base = build_features(df, K)
    perturbed = _perturb_future(df, K, RANDOM_SEED)

    # --- 正实验 ---
    pert = build_features(perturbed, K)
    if not base["电池"].tolist() == pert["电池"].tolist():
        raise AssertionError("扰动前后电池顺序不一致")
    changed = []
    for col in base.columns:
        if col == "电池":
            continue
        a = base[col].to_numpy(float)
        b = pert[col].to_numpy(float)
        bad = ~np.isclose(a, b, rtol=0, atol=0, equal_nan=True)
        if bad.any():
            for i in np.flatnonzero(bad):
                changed.append((col, base["电池"].iloc[i], float(a[i]), float(b[i])))
    if changed:
        raise AssertionError(
            "以下特征用到了 t>K 的数据（未来信息泄漏）：\n"
            + "\n".join(f"  {c} @{cell}: {x} -> {y}" for c, cell, x, y in changed)
        )

    # --- 负控制 ---
    leaky_base = _leaky_control(df, K)
    leaky_pert = _leaky_control(perturbed, K)
    caught = []
    control_cols = [c for c in leaky_base.columns if c != "电池"]
    assert control_cols, "负控制特征集为空，检测无意义"
    for col in control_cols:
        a = leaky_base[col].to_numpy(float)
        b = leaky_pert[col].to_numpy(float)
        if (~np.isclose(a, b, rtol=0, atol=0, equal_nan=True)).any():
            caught.append(col)
    if len(caught) != len(control_cols):
        raise AssertionError(
            f"负控制失败：本该被抓到的泄漏特征没被抓到（{set(control_cols) - set(caught)}），"
            "说明这个检测本身抓不到泄漏"
        )

    return {
        "K": K,
        "n_features": base.shape[1] - 1,
        "n_cells": len(base),
        "n_future_rows": int((df["cycle"] > K).sum()),
        "changed": changed,
        "control_caught": caught,
    }


# ---------------------------------------------------------------------------
# 与 EOL 的相关（描述性，明确无推断意义）
# ---------------------------------------------------------------------------
def r_critical(n: int, alpha: float = 0.05) -> float:
    """n 个样本、双侧 α 下 Pearson r 的临界值 |r|。"""
    from scipy import stats

    dfree = n - 2
    t_crit = float(stats.t.ppf(1 - alpha / 2, dfree))
    return t_crit / np.sqrt(t_crit**2 + dfree)


def correlation_table(feat: pd.DataFrame, prof: pd.DataFrame) -> pd.DataFrame:
    """每个因果特征与 EOL 循环的 Pearson r，并标注它离「显著」还有多远。

    ⚠️ n = 4。这张表**不做筛选依据**，只用于如实交代「我们的数据量根本支撑不了
    基于相关的特征选择」。W13 的 LOBOCV 才是判据。
    """
    from scipy import stats

    eol = prof.set_index("battery_id").reindex(feat["电池"])["eol_cycle"]
    eol_vals = eol.astype(float).to_numpy()
    censored = eol.isna().to_numpy()
    n_eff = int((~censored).sum())
    r_crit = r_critical(n_eff)

    rows = []
    for col in feat.columns:
        if col == "电池":
            continue
        x = feat[col].to_numpy(float)
        keep = ~censored & ~np.isnan(x)
        if keep.sum() < 3:
            rows.append({"特征": col, "n": int(keep.sum()), "r": "—",
                         "判读": "有效样本 < 3，无法计算"})
            continue
        r, _p = stats.pearsonr(x[keep], eol_vals[keep])
        # 竖线必须转义成 `\|`：markdown 表格里裸的 | 会被当成列分隔符。
        # 首版就是 `|r| < 0.997` 把整张表撕成了两半。
        if abs(r) >= r_crit:
            verdict = f"`\\|r\\|` ≥ {r_crit:.3f}，达到 n={int(keep.sum())} 的显著门槛"
        else:
            verdict = f"`\\|r\\|` < {r_crit:.3f}，**不显著**"
        rows.append(
            {
                "特征": col,
                "n": int(keep.sum()),
                "r": f"{r:+.3f}",
                "判读": verdict,
            }
        )
    return pd.DataFrame(rows)


def k_sensitivity(df: pd.DataFrame, prof: pd.DataFrame) -> pd.DataFrame:
    """换观测窗口 K，看每个特征的相关系数符号是否稳定。

    这是本模块唯一有点信息量的诊断：**方向**（正/负）如果随 K 翻号，
    说明这个特征承载的是噪声，不是趋势。n=4 下连这个都只是弱证据。
    """
    cols = None
    table = {}
    for K in K_SWEEP:
        feat = build_features(df, K)
        eol = prof.set_index("battery_id").reindex(feat["电池"])["eol_cycle"].astype(float)
        keep = eol.notna().to_numpy()
        vals = {}
        for col in feat.columns:
            if col == "电池":
                continue
            x = feat[col].to_numpy(float)[keep]
            y = eol.to_numpy()[keep]
            if np.std(x) == 0 or len(x) < 3:
                vals[col] = float("nan")
            else:
                vals[col] = float(np.corrcoef(x, y)[0, 1])
        table[f"K={K}"] = vals
    out = pd.DataFrame(table).reset_index().rename(columns={"index": "特征"})
    out["符号是否稳定"] = out[list(table)].apply(
        lambda row: "稳定" if len({np.sign(v) for v in row if not np.isnan(v)}) <= 1 else "**翻号**",
        axis=1,
    )
    return out


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------
def render_md(
    feat: pd.DataFrame,
    corr: pd.DataFrame,
    ksens: pd.DataFrame,
    leak: dict,
    n_cycle_full: dict[str, int],
    prof: pd.DataFrame,
) -> str:
    n_eff = int(corr["n"].max())
    r_crit = r_critical(n_eff)
    censored = prof.loc[prof["censored"], "battery_id"].tolist()
    n_total = len(prof)
    n_sig = int(corr["判读"].str.contains("达到").sum())
    flip = ksens.loc[ksens["符号是否稳定"] == "**翻号**", "特征"].tolist()
    stable = ksens.loc[ksens["符号是否稳定"] == "稳定", "特征"].tolist()

    out = [
        "# 早期特征与未来信息泄漏证明（W8）",
        "",
        "> 本文件由 `src/features.py` 自动生成，数字随代码重算，**不要手工编辑**。",
        "",
        "## 一、先分清两个任务",
        "",
        "本项目里「预测」有两种口径，特征规则完全不同：",
        "",
        "| | 任务 A：个体内轨迹外推（主线） | 任务 B：跨个体早期预测（本节） |",
        "|---|---|---|",
        "| 做法 | 用前 M 个循环拟合，外推后面的容量 / RUL | 用前 K 个循环的统计量，预测寿命 |",
        "| `cycle` | **回归自变量（时间轴）**，必用 | **禁止作特征**（个体内时间，跨个体不可迁移） |",
        "| 谁用 | M1 / M2 / M4 | M3 随机森林（反例实验） |",
        f"| 样本量 | 每块电池一条曲线 | **名义 {n_total} 块，可用标签 {n_eff} 块** |",
        "",
        "`preprocess.FEATURE_EXCLUDE` 里「`cycle` 不得作特征」那条针对的是**任务 B**；"
        "任务 A 里 `cycle` 是 x 轴而不是特征。这两句必须同时写，"
        "否则报告里「禁止用 cycle」和「模型输入含 cycle」会互相打脸。",
        "",
        "## 二、特征表（K = "
        f"{K_MAIN} 循环，只用 cycle ≤ K 的数据）",
        "",
        feat.round(5).to_markdown(index=False),
        "",
        "| 特征 | 可得性：为什么在 t = K 时刻就能算出来 |",
        "|---|---|",
    ]
    for col, why in FEATURE_DOC.items():
        out.append(f"| `{col}` | {why}（{AVAILABILITY[col]}） |")
    cur = feat["curvature_to_K"].to_numpy(float)
    out += [
        "",
        "**一条要提前说的特征缺陷**："
        f"K={K_MAIN} 时无量纲曲率跨 {np.nanmin(cur):.2f} 到 {np.nanmax(cur):+.2f}，"
        "同一个指标在四块电池上差了两个数量级，这本身就说明它不稳。"
        "原因是分母是二次拟合的一次项系数 |b|，而前 50 个循环里容量几乎还没开始掉，"
        "b 接近 0 时把小误差放大。"
        "第五节会看到它在 K 敏感性里翻号，就是同一个不稳的另一面——"
        "**所以 W13 不把它当主力特征**，只留作「形状」的辅助说明。",
        "",
        "## 三、泄漏证明：不靠保证，靠两段实验",
        "",
        f"把 **t > K 的 {leak['n_future_rows']} 行容量随机扰动 ±{PERTURB_AH} Ah**"
        f"（额定的 ±{PERTURB_AH / 2.0:.0%}，远超真实波动），重算全部特征：",
        "",
        f"- **正实验**：{leak['n_features']} 个因果特征 × {leak['n_cells']} 块电池，"
        "在扰动前后**逐个完全相同**（`rtol=0, atol=0` 严格判等）→ 没有一个特征碰到未来数据。",
        f"- **负控制**：故意泄漏的 `{'`、`'.join(leak['control_caught'])}` "
        "（用整条轨迹算的）**全部被抓到** → 说明这个检测真的能发现泄漏，不是空转。",
        "",
        "只做正实验、不做负控制的检测，等价于「我保证没泄漏」——"
        "这句话在报告里没有价值，能被复现的断言才有。",
        "",
        "顺带纠正一个我一开始写错的说法：**居中滚动窗口在本设计里并不构成泄漏。**"
        f"因为每块电池只算**一个**特征向量、时刻固定在 t = {K_MAIN}，"
        f"输入已截断到 cycle ≤ {K_MAIN}——居中窗口在第 t 行最多用到第 {K_MAIN} 行，没有越过。"
        "只有「逐行给出 (cell, t) 特征且 t 可以小于 K」的面板式设计，居中窗口才会看到未来。"
        "本模块用 trailing（`center=False`）是**保守选择**：让「只依赖 ≤ 当前时刻」"
        "在代码形态上一目了然，代价是残差估计方差略大。",
        "",
        "## 四、与 EOL 的相关：如实列出，并说明它没有推断意义",
        "",
        "先把样本量说清楚："
        f"名义上有 {n_total} 块电池，但 **{'、'.join(censored)} 在 "
        f"τ={EOL_THRESHOLD_AH:g} Ah 下删失**（观察期内没到达 EOL），它没有 `eol_cycle`，"
        f"所以下面每一行的 **n 都只有 {n_eff}**。这不是可以「补一个寿命」凑成 {n_total} 的——"
        "把删失记成观测长度会系统性低估寿命（D2 已禁止）。",
        "",
        corr.to_markdown(index=False),
        "",
        f"**n = {n_eff}，df = {n_eff - 2}，α = 0.05 双侧的临界 |r| = {r_crit:.3f}。**"
        f"也就是说，{n_eff} 个点必须几乎完全单调才「显著」，"
        "而那不是「有关联」，那是**几个排序值的巧合**"
        "（W5 已被这个坑咬过一次：σ 最大的两块恰好是再生最多的两块，p≈0.08）。",
        "",
        f"本表里达到这个门槛的特征有 **{n_sig} 个**。"
        "**即使有，也不作为筛选依据**——"
        f"理由不是「我们不信任统计」，而是 n={n_eff} 时任何基于相关的选择都无法与事后挑选区分开。",
        "",
        "## 五、换一个观测窗口 K，方向还稳吗",
        "",
        ksens.round(3).to_markdown(index=False),
        "",
        f"- 符号**稳定**的特征：{'、'.join(f'`{c}`' for c in stable) if stable else '无'}；",
        f"- 符号**随 K 翻号**的特征：{'、'.join(f'`{c}`' for c in flip) if flip else '无'}。",
        "",
        ("翻号说明这个特征承载的是窗口长度的效应，而不是与寿命的稳定关联，"
         "**不能拿它当「越 X 越短命」这类结论的依据**。"
         if flip else
         "没有特征翻号，这是个弱证据（n=4 下符号稳定也可能是巧合）。"),
        "",
        "## 六、这一节刻意没做的事",
        "",
        f"- **没有做特征筛选**：有效样本只有 {n_eff} 个，筛选等于挑好看的结果。"
        "选哪些特征交给 W13 的 LOBOCV 判。",
        f"- **没有画散点图**：{n_eff} 个点的散点图只会让人以为看到了规律。",
        "- **没有用 `cell_censored` / `cap_min` / `eol_cycle` 这类整轨迹量**："
        "它们是标签或标签的变形，W3 已登记在 `PROFILE_EXCLUDE`。",
        "",
        "## 七、观测长度（供下游参考）",
        "",
        "| 电池 | 总循环 | 前 K 占比 |",
        "|---|---|---|",
    ]
    for cell, n in n_cycle_full.items():
        out.append(f"| {cell} | {n} | {K_MAIN / n:.0%} |")
    out += [
        "",
        f"最短的 {min(n_cycle_full, key=n_cycle_full.get)} 只有 "
        f"{min(n_cycle_full.values())} 循环，所以 K 不能超过约 100，"
        "否则最短个体连一个窗口都给不出。",
        "",
    ]
    return "\n".join(out)


def main() -> int:
    ensure_dirs()

    raw = load_raw()
    df = mark_rows(raw)
    prof = cell_profile(df)
    series = split_by_battery(df)
    n_cycle_full = {c: int(len(s)) for c, s in series.items()}

    # --- 断言 1：特征构建前，输入数据的行数必须真的够 K ---
    for cell, sub in series.items():
        assert len(sub) >= K_MAIN, f"{cell} 观测长度 {len(sub)} < K={K_MAIN}"

    # --- 断言 2：泄漏证明 ---
    leak = leakage_test(df, K_MAIN)

    # --- 断言 3：特征里不许出现 FEATURE_EXCLUDE 的列名 ---
    from preprocess import FEATURE_EXCLUDE

    feat = build_features(df, K_MAIN)
    forbidden = set(FEATURE_EXCLUDE) & set(feat.columns)
    assert not forbidden, f"特征集里混进了被禁止作特征的列：{sorted(forbidden)}"

    # --- 断言 4：每个 K 都能跑通（K 敏感性表不能有空洞）---
    corr = correlation_table(feat, prof)
    ksens = k_sensitivity(df, prof)
    assert list(ksens.columns) == ["特征", *[f"K={K}" for K in K_SWEEP], "符号是否稳定"]

    md = render_md(feat, corr, ksens, leak, n_cycle_full, prof)
    TABLE_MD.write_text(md, encoding="utf-8")

    r_crit = r_critical(int(corr["n"].max()))
    print(f"K = {K_MAIN}；特征 {feat.shape[1] - 1} 个 × {len(feat)} 块电池")
    print()
    print(feat.round(5).to_string(index=False))
    print()
    print("泄漏检测：")
    print(f"  正实验：{leak['n_features']} 个因果特征 × {leak['n_cells']} 块，"
          f"扰动未来 {leak['n_future_rows']} 行后逐个不变 ✅")
    print(f"  负控制：{leak['control_caught']} 全部被抓到 ✅")
    print()
    print(f"与 EOL 的相关（n={int(corr['n'].max())}，临界 |r|={r_crit:.3f}）：")
    print(corr.to_string(index=False))
    print()
    print("K 敏感性（相关符号）：")
    print(ksens.round(3).to_string(index=False))
    print()
    print(f"早期特征表已写出：{TABLE_MD.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
