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
    # 窗口必须是**完整且从严**的 1..K，两头都要卡（第四轮审查 F6）：
    # - 右端不足 K：算出来的其实是 K' < K 的特征；
    # - 左端不是 1（比如给了 cycle 11..50 共 40 行）：`cap_first` 会取到第 11 循环，
    #   `drop_ratio_to_K` 也不再是「首循环到第 K 循环」的衰减——而表头照样写着「第 1 循环」。
    # 初版只查了右端，左端截断能静默混过。
    cyc = sub["cycle"].to_numpy(int)
    if not np.array_equal(cyc, np.arange(1, K + 1)):
        raise ValueError(
            f"特征窗口必须是完整的 1..{K}，实际是 {int(cyc.min())}..{int(cyc.max())}"
            f"（共 {len(sub)} 行）——否则算出来的不是 K={K} 的特征"
        )
    cap = sub["capacity_Ah"].to_numpy(float)
    # 非有限值必须直接报错：`np.polyfit` 遇 NaN 会静默返回 NaN，
    # 而泄漏检测用 `equal_nan=True` 比较，NaN==NaN 会被判成「没变」——
    # 整条链路都发现不了。这是对抗输入测试抓到的第二类静默错误。
    if not np.all(np.isfinite(cap)):
        raise ValueError(
            f"特征输入里有 {int((~np.isfinite(cap)).sum())} 个非有限容量值，"
            "算出来的特征会是 NaN 而不会报错"
        )
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
    "resid_std_trailing": (
        f"用 2..K 作滚动历史，取第 {TREND_WINDOW + 1}..K 循环的残差（第 1 循环从不参与："
        "尾部窗口在它那里没有任何历史）"
    ),
    "n_regen_to_K": f"用 1..K 的差分（第 1 循环没有差分，天然不参与）",
    "max_regen_to_K": "用 1..K 的差分（同上，第 1 循环不参与）",
}

SCHEMA: tuple[str, ...] = tuple(FEATURE_DOC)
"""特征的顺序与名字的唯一出处。

加这条是为了让「特征集被人悄悄改掉」变成硬错误：
`build_features_one` 返回的 key 序列必须与它**逐项相等**。
没有它的时候，往特征里塞一列（或改名）不会有任何断言响。
"""


def build_features(
    df: pd.DataFrame,
    K: int,
    extra=None,
) -> pd.DataFrame:
    """四块电池各自的前 K 循环特征。

    参数 `extra` 只给泄漏检测用：一个 `(full_df, feat_df) -> feat_df` 的回调，
    在**同一条管线末端**追加列。它的存在是为了让负控制走**真正的特征管线**
    （第四轮审查 F5：初版负控制绕过了 `build_features`，于是它只证明了
    「比较器有分辨力」，证明不了「管线里的泄漏检测有分辨力」）。
    """
    rows = []
    for cell, sub in split_by_battery(df).items():
        head = sub[sub["cycle"] <= K]
        if len(head) < K:
            raise ValueError(f"{cell} 只有 {len(head)} 个循环，不足 K={K}")
        feats = build_features_one(head, K)
        rows.append({"电池": cell, **feats})
    out = pd.DataFrame(rows)
    if extra is not None:
        out = extra(df, out)
    # 输出必须是有限值：NaN 特征会让下游的分支静默走错
    # （`k_sensitivity` 的符号集合会跳过 NaN→空集→标「稳定」；
    #  泄漏检测若用 equal_nan=True 会把 NaN==NaN 判成「没变化」）。
    vals = out.drop(columns=["电池"]).to_numpy(float)
    if not np.all(np.isfinite(vals)):
        bad = int((~np.isfinite(vals)).sum())
        raise ValueError(f"特征输出里有 {bad} 个非有限值，下游会把它们当有效数字用")
    return out


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


def _leaky_control(df: pd.DataFrame) -> pd.DataFrame:
    """**故意泄漏**的对照特征，三条，分别对应三种不同的泄漏通道。

    第四轮审查（W8）指出初版只放了两条「整轨迹容量统计」，且它们自己算的——
    和正实验走的是**不同代码路径**，所以负控制证明不了正实验那条路的检测力。
    现在三条对齐三个通道：

    | 通道 | 对照特征 | 只有哪条实验能抓到 |
    |---|---|---|
    | 未来**容量值** | `LEAK_cap_min_full` | 扰动臂（改容量） |
    | 未来**行是否存在** | `LEAK_n_cycles_full` | 删行臂（删掉 t>K 的行） |
    | 未来**标签** | `LEAK_eol_cycle_full` | 两者都能 |

    `LEAK_n_cycles_full` 是审查亲自补出来的漏洞：它只是一个「行数」，
    扰动容量时**完全不变**，所以只做扰动臂的检测会放过它——
    而它会直接把 168 写进 md。
    """
    from config import is_at_eol as _is_eol  # 判定口径仍走唯一入口（D17）

    rows = []
    for cell, sub in split_by_battery(df).items():
        sub = sub.sort_values("cycle")
        cap = sub["capacity_Ah"].to_numpy(float)
        cyc = sub["cycle"].to_numpy(int)
        hit = np.flatnonzero(_is_eol(cap))
        rows.append(
            {
                "电池": cell,
                "LEAK_cap_min_full": float(cap.min()),
                "LEAK_n_cycles_full": float(len(sub)),
                "LEAK_eol_cycle_full": float(cyc[hit[0]]) if len(hit)
                else float(cyc[-1] + 1),
            }
        )
    return pd.DataFrame(rows)


def _leaky_extra(full: pd.DataFrame, feat: pd.DataFrame) -> pd.DataFrame:
    """给 `build_features(..., extra=...)` 用的**故意泄漏**列。

    形状是 `(全量 df, 已算好的特征帧) -> 特征帧`，所以它走的是**同一条管线末端**，
    与真实特征共享 `build_features` 的返回路径——这样负控制才能证明
    「管线里的泄漏检测有分辨力」，而不只是「比较器有分辨力」（审查 F5）。
    """
    leak = _leaky_control(full)
    return feat.merge(leak, on="电池", how="left")


def leakage_test(df: pd.DataFrame, K: int) -> dict:
    """**三臂**泄漏检测：两条正实验 + 一条负控制。

    正实验（真实特征集必须**逐位不变**）：
    - **A 扰动臂**：把 t>K 的容量随机扰动 ±PERTURB_AH → 抓「用了未来容量值」；
    - **B 删行臂**：把 t>K 的行**整行删掉**再算 → 抓「用了未来行的存在性/元数据」
      （行数、最大循环号、未来时刻的温度统计……）。
      **这一臂是第四轮审查补的**：初版只有 A，而 A 对元数据**完全无感**——
      一个 `n_cycles = len(全表)` 的变异能顺利通过 A、通过白名单断言、还能把 168 写进 md。

    负控制：三条故意泄漏的对照特征，必须**被对应的臂抓到**；
    抓不到说明那一臂没有分辨力，必须先修检测。

    只做正实验、不做负控制的检测，等价于「我保证没泄漏」。
    """
    base = build_features(df, K)

    # --- 正实验 A：扰动未来容量 ---
    perturbed = _perturb_future(df, K, RANDOM_SEED)
    pert = build_features(perturbed, K)
    changed_a = _diff_frames(base, pert)

    # --- 正实验 B：删掉未来行 ---
    # 注意：这里不能只删容量列，必须整行删——否则「行数」这种元数据仍是未来的。
    kept = df[df["cycle"] <= K].copy()
    deleted = build_features(kept, K)
    changed_b = _diff_frames(base, deleted)

    # --- 负控制：三个通道各一条，且**都走 build_features 的 extra 钩子** ---
    leaky_base = build_features(df, K, extra=_leaky_extra)
    leaky_pert = build_features(perturbed, K, extra=_leaky_extra)
    leaky_del = build_features(kept, K, extra=_leaky_extra)
    control_cols = [c for c in leaky_base.columns if c.startswith("LEAK_")]
    assert control_cols, "负控制特征集为空，检测无意义"
    caught_a = _cols_hit(control_cols, leaky_base, leaky_pert)
    caught_b = _cols_hit(control_cols, leaky_base, leaky_del)

    if changed_a:
        raise AssertionError(
            "以下特征用到了 t>K 的**容量值**（扰动臂发现）：\n"
            + "\n".join(f"  {d}" for d in changed_a)
        )
    if changed_b:
        raise AssertionError(
            "以下特征用到了 t>K 的**行/元数据**（删行臂发现）：\n"
            + "\n".join(f"  {d}" for d in changed_b)
        )
    # 每条对照特征必须至少被一条臂抓到；且两个通道各自都要有臂负责
    missed = [c for c in control_cols if c not in caught_a and c not in caught_b]
    if missed:
        raise AssertionError(
            f"负控制失败：本该被抓到的泄漏特征没被抓到（{missed}）——这一臂没有分辨力"
        )
    if not caught_a:
        raise AssertionError("扰动臂一条对照特征都没抓到，等于没测")
    if not caught_b:
        raise AssertionError("删行臂一条对照特征都没抓到，等于没测")

    return {
        "K": K,
        "n_features": base.shape[1] - 1,
        "n_cells": len(base),
        "n_future_rows": int((df["cycle"] > K).sum()),
        "changed_a": changed_a,
        "changed_b": changed_b,
        "caught_a": caught_a,
        "caught_b": caught_b,
    }


def _cols_hit(cols: list[str], a: pd.DataFrame, b: pd.DataFrame) -> list[str]:
    """哪些列在两帧之间变了。`_diff_frames` 返回的是「列 @电池」字符串，这里按前缀归类。"""
    diffs = _diff_frames(a, b)
    return [c for c in cols if any(d.startswith(c + " @") for d in diffs)]


def _diff_frames(a: pd.DataFrame, b: pd.DataFrame) -> list[str]:
    """逐列逐行严格比对（rtol=0, atol=0），返回不一致的「列 @电池」清单。"""
    if a["电池"].tolist() != b["电池"].tolist():
        raise AssertionError("两帧的电池顺序不一致")
    out = []
    for col in a.columns:
        if col == "电池":
            continue
        x = a[col].to_numpy(float)
        y = b[col].to_numpy(float)
        bad = ~np.isclose(x, y, rtol=0, atol=0, equal_nan=True)
        for i in np.flatnonzero(bad):
            out.append(f"{col} @{a['电池'].iloc[i]}: {x[i]!r} -> {y[i]!r}")
    return out


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
        r, p = stats.pearsonr(x[keep], eol_vals[keep])
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
                "p": f"{p:.3f}",
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


GOLDEN: dict[str, dict[str, float]] = {
    # 在 K = 50 上把关键特征钉死。这不是「测自己」，是**回归护栏**：
    # 一旦有人改了 K_MAIN、改了再生阈值、把曲率换成斜率、或动坏了残差窗口，
    # 这里的数字立刻对不上（第四轮审查 F7 实测：K_MAIN→100、曲率→线性斜率、
    # REGEN 阈值→0 这三类改动当时**一条断言都拦不住**，还会照样改写 md）。
    # 课程方若更新数据集，这些值会全部 FAIL —— 与 sanity_check 的 EXPECTED_* 同一性质：
    # 这是特性不是 bug（README 已写明）。
    "B0005": {"cap_first": 1.8565, "cap_at_K": 1.7674, "curvature_to_K": -11.19506,
              "resid_std_trailing": 0.01828, "n_regen_to_K": 1.0},
    "B0006": {"cap_first": 2.0353, "cap_at_K": 1.7758, "curvature_to_K": 0.04384,
              "resid_std_trailing": 0.03990, "n_regen_to_K": 3.0},
    "B0007": {"cap_first": 1.8911, "cap_at_K": 1.8002, "curvature_to_K": -6.05846,
              "resid_std_trailing": 0.01438, "n_regen_to_K": 0.0},
    "B0018": {"cap_first": 1.8550, "cap_at_K": 1.6607, "curvature_to_K": 0.49851,
              "resid_std_trailing": 0.03498, "n_regen_to_K": 2.0},
}

GOLDEN_REL_TOL = 1e-4


def assert_golden(feat: pd.DataFrame) -> None:
    """把 K_MAIN 上的关键特征与 `GOLDEN` 比对。"""
    got = feat.set_index("电池")
    bad = []
    for cell, want in GOLDEN.items():
        if cell not in got.index:
            bad.append(f"{cell} 不在特征表里")
            continue
        for name, w in want.items():
            g = float(got.loc[cell, name])
            tol = GOLDEN_REL_TOL * max(1.0, abs(w))
            if abs(g - w) > tol:
                bad.append(f"{cell}.{name}: 期望 {w}，实际 {g}（容差 {tol:.1e}）")
    if bad:
        raise AssertionError(
            "golden 值对不上 —— K / 阈值 / 特征定义被改动过：\n"
            + "\n".join(f"  {x}" for x in bad)
        )


def curvature_profile(df: pd.DataFrame, ks=K_SWEEP) -> pd.DataFrame:
    """**原始曲率随 K 的剖面**——这是「曲率不稳定」的正确证据。

    第四轮审查 F2 指出：md 里原来拿「四块电池之间差多少」论证「不稳定」，
    那是**跨电池离散**，不是**同一块电池随观测窗口的漂移**，两者不是一回事。
    真正的证据是这张表：同一块电池，K 从 20 变到 100 时曲率怎么跑；
    再给出二次项系数 c2 的跨 K 范围，用来解释「它为什么跑」。
    """
    rows = []
    for cell, sub in split_by_battery(df).items():
        sub = sub.sort_values("cycle")
        rec = {"电池": cell}
        c2s = []
        for K in ks:
            head = sub[sub["cycle"] <= K]
            if len(head) < K:
                rec[f"K={K}"] = "—"
                continue
            rec[f"K={K}"] = f"{_curvature(head):.2f}"
            x = head["cycle"].to_numpy(float) / 100.0
            y = head["capacity_Ah"].to_numpy(float)
            c2s.append(float(np.polyfit(x, y, 2)[0]))
        rec["c2 跨 K 范围"] = f"[{min(c2s):+.3f}, {max(c2s):+.3f}]" if c2s else "—"
        rows.append(rec)
    return pd.DataFrame(rows)


def max_feasible_k(df: pd.DataFrame) -> int:
    """能构造出完整 1..K 窗口的最大 K（= 最短那条序列的长度）。"""
    return int(min(len(s) for s in split_by_battery(df).values()))


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------
def _assert_matches_individual(df: pd.DataFrame) -> None:
    """F11：`features._linear`/`_curvature` 与 `individual._linear`/`_curvature_scaled`
    数学上完全相同（审查实测逐位一致），但当时**没有任何断言保证这条等价**——
    改一处就会静默分叉。这里补上等价性检查。

    唯一的**有意差异**：本模块在两参数样本不足时返回 NaN（`len<3` 守卫），
    而 W5 那边用两点也能拟合。所以只在 len ≥ 3 上比对。
    """
    from individual import _curvature_scaled, _linear as ind_linear

    sub = split_by_battery(df)["B0005"]
    head = sub[sub["cycle"] <= K_MAIN].sort_values("cycle")
    s_feat, r2_feat = _linear(head)
    s_w5, r2_w5 = ind_linear(head)
    assert abs(s_feat - s_w5) <= 1e-15, f"线性斜率两份实现分叉：{s_feat!r} vs {s_w5!r}"
    assert abs(r2_feat - r2_w5) <= 1e-15, f"线性 R² 两份实现分叉：{r2_feat!r} vs {r2_w5!r}"
    a, b = _curvature(head), _curvature_scaled(head)
    assert abs(a - b) <= 1e-12 * max(1.0, abs(b)), (
        f"曲率两份实现分叉：features={a!r} individual={b!r}"
    )


def render_md(
    feat: pd.DataFrame,
    corr: pd.DataFrame,
    ksens: pd.DataFrame,
    leak: dict,
    n_cycle_full: dict[str, int],
    prof: pd.DataFrame,
    curv_tbl: pd.DataFrame,
    k_feasible: int,
) -> str:
    n_eff = int(corr["n"].max())
    r_crit = r_critical(n_eff)
    censored = prof.loc[prof["censored"], "battery_id"].tolist()
    n_total = len(prof)
    n_sig = int(corr["判读"].str.contains("达到").sum())
    min_p = float(corr["p"].astype(float).min())
    drop_ratio = dict(zip(feat["电池"], feat["drop_ratio_to_K"]))
    top_drop = max(drop_ratio, key=drop_ratio.get)
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
        "**一条要提前说的特征缺陷：`curvature_to_K` 随观测窗口剧烈漂移。**",
        "",
        "同一块电池、只改观测窗口 K，无量纲曲率是这样跑的：",
        "",
        curv_tbl.to_markdown(index=False),
        "",
        "这才是「不稳」的正确证据——**同一块电池随 K 漂移**，而不是「四块电池之间差多少」"
        "（后者是跨电池离散，两回事）。"
        "例如 B0005 从 K=40 的 "
        f"{curv_tbl.loc[curv_tbl['电池'] == 'B0005', 'K=40'].iloc[0]} 掉到 K=50 的 "
        f"{curv_tbl.loc[curv_tbl['电池'] == 'B0005', 'K=50'].iloc[0]}；"
        "二次项系数 c2 的跨 K 范围见末列，它连**符号**都不稳。",
        "",
        "⚠️ 我原来在这里写的机制是**编的**（第四轮审查 F2 逐条否证）："
        "「前 50 个循环容量几乎还没开始掉」——不成立，特征表里 `drop_ratio_to_K` 显示 "
        f"{top_drop} 已掉 {drop_ratio[top_drop] * 100:.2f}%、"
        f"B0018 已掉 {drop_ratio['B0018'] * 100:.2f}%；"
        "「b 接近 0」——也不成立，B0006/B0018 的 b 分别是 −0.607 和 −0.930，"
        "离 0 有 414 / 850 个标准误；"
        "「差了两个数量级」——按端点绝对值算只有 "
        f"{abs(np.nanmin(cur)) / abs(np.nanmax(cur)):.1f} 倍。"
        "**结论（不稳、不作主力）保留，但理由必须换成上面这张 K 剖面表。**"
        "W13 里它只作「形状」的辅助说明。",
        "",
        "## 三、泄漏证明：两条正实验 + 一条带对照的负控制",
        "",
        f"**正实验 A（扰动臂）**：把 t > K 的 {leak['n_future_rows']} 行容量随机扰动 "
        f"±{PERTURB_AH} Ah（额定的 ±{PERTURB_AH / 2.0:.0%}），重算全部特征。",
        "",
        f"**正实验 B（删行臂）**：把 t > K 的 {leak['n_future_rows']} 行**整行删掉**，重算全部特征。",
        "",
        f"两臂都必须让 {leak['n_features']} 个特征 × {leak['n_cells']} 块电池"
        "**逐个完全相同**（`rtol=0, atol=0` 严格判等）。",
        "",
        "⚠️ **为什么必须有第二臂**（第四轮审查 F1）：初版只有扰动臂，而它只能发现"
        "「用了未来的**容量值**」。如果某列用的是「未来**行是否存在**」这类元数据"
        "（行数、最大循环号……），扰动容量时它**一个数都不会变**——"
        "审查据此造了一个 `n_cycles = len(全表)` 的变异，它顺利通过扰动臂、通过列名白名单，"
        "最后把 168 写进了本文件。删行臂就是专门堵这个洞的。",
        "",
        "**负控制**（三条，分别对应三种泄漏通道，且**都走 `build_features` 的 `extra` 钩子**，"
        "与真实特征共享同一段返回路径）：",
        "",
        "| 对照特征 | 通道 | 被哪一臂抓到 |",
        "|---|---|---|",
    ]
    a_set, b_set = set(leak["caught_a"]), set(leak["caught_b"])
    for name, channel in (
        ("LEAK_cap_min_full", "未来容量值"),
        ("LEAK_n_cycles_full", "未来行是否存在"),
        ("LEAK_eol_cycle_full", "未来标签"),
    ):
        marks = []
        if name in a_set:
            marks.append("扰动臂")
        if name in b_set:
            marks.append("删行臂")
        out.append(f"| `{name}` | {channel} | {'、'.join(marks)} |")
    out += [
        "",
        "三条**全部被抓到**，且两臂各自都至少抓到一条 → 检测有分辨力，不是空转。",
        "",
        "（初版的负控制是自己另外算两列、**绕过** `build_features`，所以它只证明了"
        "「比较器有分辨力」，证明不了「管线里的泄漏检测有分辨力」——这是审查 F5。"
        "现在对照列由 `build_features(..., extra=...)` 在管线末端注入。）",
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
        "## 五、换一个观测窗口 K：**只作描述，不作推断**",
        "",
        ksens.round(3).to_markdown(index=False),
        "",
        f"⚠️ **n = {n_eff} 时，这一整节的符号比较都不能当证据用**（第四轮审查 F4）：",
        f"10 个特征的 p 值**全部 > 0.20**（最小的也有 "
        f"{min_p:.2f}）；而 n={n_eff} 时 Fisher z 的标准误 "
        f"`1/√(n−3)` 分母为负、根本无定义，r 的 95% 置信区间是**整个 [−1, 1]**。",
        "所以下面这两行只是**描述**，不是「哪些特征可信」的结论：",
        "",
        f"- 符号未变：{'、'.join(f'`{c}`' for c in stable) if stable else '无'}；",
        f"- 符号变过：{'、'.join(f'`{c}`' for c in flip) if flip else '无'}。",
        "",
        "而且「变号」里混着**近零值**（`resid_std_trailing` 在 K=30 是 0.032、K=40 是 −0.026），"
        "把一个几乎为 0 的相关系数从正变负叫作「窗口长度的效应」，是**给噪声安机制**——"
        "这正是 D19 第 3 条禁止的事。所以本节**不下**这类结论。",
        "",
        "另一处口径提醒：K=20 时 `slope_tail20_Ah_per_100` 与 "
        "`slope_to_K_Ah_per_100` 是**同一个量**（20 个点的「末尾 20」就是全部），"
        "所以 K=20 那一列的两个相关系数必然相同（都是 "
        f"{ksens.loc[ksens['特征'] == 'slope_to_K_Ah_per_100', 'K=20'].iloc[0]:.3f}），"
        "不是两条独立证据。",
        "",
        "## 六、这一节刻意没做的事",
        "",
        f"- **没有做特征筛选**：有效样本只有 {n_eff} 个，筛选等于挑好看的结果。"
        "选哪些特征交给 W13 的 LOBOCV 判。",
        f"- **没有画散点图**：{n_eff} 个点的散点图只会让人以为看到了规律。",
        "- **没有用 `cell_censored` / `cap_min` / `eol_cycle` 这类整轨迹量**："
        "它们是标签或标签的变形，W3 已登记在 `PROFILE_EXCLUDE`。",
        "",
        "## 七、观测长度与 K 的可行范围",
        "",
        "| 电池 | 总循环 | 前 K 占比 |",
        "|---|---|---|",
    ]
    for cell, n in n_cycle_full.items():
        out.append(f"| {cell} | {n} | {K_MAIN / n:.0%} |")
    out += [
        "",
        f"**K 的可行上限是 {k_feasible}**（= 最短序列 "
        f"{min(n_cycle_full, key=n_cycle_full.get)} 的长度），"
        f"K = {k_feasible + 1} 就会因为凑不出完整的 1..K 窗口而报错。",
        "",
        f"正式口径取 K = {K_MAIN} **不是**因为「再大就跑不动」——"
        f"{k_feasible} 都能跑。取 {K_MAIN} 是为了给最短的那块留出 "
        f"{min(n_cycle_full.values()) - K_MAIN} 个循环"
        f"（{1 - K_MAIN / min(n_cycle_full.values()):.0%}）做后段验证。"
        "（初版这里写「K 不能超过约 100，否则给不出窗口」，是错的——第四轮审查 F3。）",
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

    # --- 断言 2：泄漏证明（两条正实验 + 带对照的负控制）---
    leak = leakage_test(df, K_MAIN)

    # --- 断言 3：特征列名必须与 SCHEMA 逐项相等（F7：加列/改名/换序都必须炸）---
    from preprocess import FEATURE_EXCLUDE

    feat = build_features(df, K_MAIN)
    assert tuple(feat.columns[1:]) == SCHEMA, (
        f"特征列与 SCHEMA 不一致：\n  期望 {SCHEMA}\n  实际 {tuple(feat.columns[1:])}"
    )

    # --- 断言 4：不许出现 FEATURE_EXCLUDE 的列名 ---
    forbidden = set(FEATURE_EXCLUDE) & set(feat.columns)
    assert not forbidden, f"特征集里混进了被禁止作特征的列：{sorted(forbidden)}"

    # --- 断言 5：golden 值（F7：K / 再生阈值 / 曲率定义被改动时必须炸）---
    assert_golden(feat)

    # --- 断言 6：K 敏感性表不许有空洞（F7：全 NaN 曾被判「稳定」并照样写 md）---
    corr = correlation_table(feat, prof)
    ksens = k_sensitivity(df, prof)
    assert list(ksens.columns) == ["特征", *[f"K={K}" for K in K_SWEEP], "符号是否稳定"]
    k_vals = ksens[list(ksens.columns[1:-1])].apply(pd.to_numeric, errors="coerce")
    assert k_vals.notna().all().all(), "K 敏感性表里有 NaN —— 某个 K 没算出来"

    # --- 断言 7：与 W5 的形态函数等价（F11：两份实现必须同值）---
    _assert_matches_individual(df)

    curv_tbl = curvature_profile(df)
    k_feasible = max_feasible_k(df)
    md = render_md(feat, corr, ksens, leak, n_cycle_full, prof, curv_tbl, k_feasible)
    # F10：md 里出现 n=4 就是叙述与样本量脱节（B0007 删失后只有 3 个标签）
    assert "n=4" not in md and "n = 4" not in md, "md 里残留了 n=4 的说法"
    TABLE_MD.write_text(md, encoding="utf-8")

    r_crit = r_critical(int(corr["n"].max()))
    print(f"K = {K_MAIN}；特征 {feat.shape[1] - 1} 个 × {len(feat)} 块电池")
    print()
    print(feat.round(5).to_string(index=False))
    print()
    print("泄漏检测（两条正实验 + 带对照的负控制）：")
    print(f"  A 扰动臂：扰动未来 {leak['n_future_rows']} 行容量，"
          f"{leak['n_features']} 个特征 × {leak['n_cells']} 块逐个不变 ✅")
    print(f"  B 删行臂：删掉未来 {leak['n_future_rows']} 行后重算，同样逐个不变 ✅")
    print(f"  负控制：扰动臂抓到 {leak['caught_a']}；删行臂抓到 {leak['caught_b']} ✅")
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
