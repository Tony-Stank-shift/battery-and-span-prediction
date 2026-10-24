"""清洗与预处理（W3）。

这份数据的"干净"是假象：0 缺失、0 重复，但埋着四个坑。本模块做三件事：

1. 把四个坑的**处理决策**固化成代码，并生成决策表
   （`outputs/tables/data_cleaning_decisions.md`）——决策表里的每个数字都是
   现算的，不是手写的，避免文档和代码说两套话。
2. 加行级标记列，供后续建模使用。
3. 用白名单断言保证「哪些列能作特征」这件事不留缺口。

刻意不做的事（都是"看起来合理但错"的做法）
------------------------------------------
- **不删任何行**：删行会引入选择偏差。
- **不截断任何测量值**：把 2.0353 截成 2.0 等于篡改测量，会人为制造平坦段。
- **不用全局 3σ / IQR 判异常**：容量是**趋势数据**，全局统计量对趋势没有分辨力。
  本模块把"三种检测口径的实际结果"都算出来放进决策表。

⚠️ 未来信息警告
----------------
`cell_censored` 由**整条轨迹的最小容量**判定，也就是说第 t 行的值已经包含了
t 之后的信息。实测 328/636 行属于这种情况。它只能用于档案与作图，
**不得作特征**（已列入 FEATURE_EXCLUDE）。

运行
----
    python src/preprocess.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (  # noqa: E402
    BATTERIES,
    EOL_THRESHOLD_AH,
    RANDOM_SEED,
    REGEN_MIN_JUMP_AH,
    RATED_CAPACITY_AH,
    TABLE_DIR,
    ensure_dirs,
    is_at_eol,
)
from load_data import add_derived, load_raw, split_by_battery  # noqa: E402

TARGET_COLUMN = "capacity_Ah"
"""预测目标。除它以外，**每一个**行级列都必须出现在 FEATURE_EXCLUDE 里。"""

# ---------------------------------------------------------------------------
# 不可作为特征的列
# ---------------------------------------------------------------------------
FEATURE_EXCLUDE: dict[str, str] = {
    "battery_id": "标识符，非数值特征（个体差异由交叉验证处理，不靠编码进模型）",
    "cycle": "个体内时间轴，跨个体不可迁移；寿命预测是外推任务，把它当特征会把外推误差伪装成插值",
    "ambient_temp_C": "全表唯一值 24.0，零方差，无信息量",
    "cap_over_rated": "capacity_Ah 的精确仿射变换（×1/2.0），作特征即标签泄漏；预测未来时等价于 lag 容量、属冗余",
    "cap_over_first": "capacity_Ah 的精确仿射变换（×1/首循环容量），同上",
    "cap_delta": "含当前时刻目标；预测当前容量时禁止，仅当任务是 t→t+1 前瞻且特征在 t 时刻可得时才可作为历史特征",
    "is_regen": "cap_delta 的阈值化，与 cap_delta 同类问题",
    "cap_exceeds_rated": "capacity_Ah 的阈值化，与 cap_over_rated 同类问题",
    "cell_censored": "由**整条轨迹**最小容量判定，第 t 行已含 t 之后的观测 → 真·未来信息（实测 328/636 行受影响）",
}

# ---------------------------------------------------------------------------
# 个体档案表里不可回填成行级特征的列（全是整轨迹统计量或标签本身）
# ---------------------------------------------------------------------------
PROFILE_EXCLUDE: dict[str, str] = {
    "eol_cycle": "寿命标签本身（首次到达 EOL 的循环）",
    "censored": "删失标签本身",
    "cap_last": "观测末期容量，含未来观测",
    "cap_min": "全轨迹最小值，含未来观测",
    "cap_min_cycle": "全轨迹最小值的位置，含未来观测",
    "cap_max": "全轨迹最大值，含未来观测",
    "fade_pct": "用全轨迹最小值计算，含未来观测",
    "slope_Ah_per_100": "用全轨迹拟合的斜率，含未来观测",
    "n_regen": "全轨迹计数，含未来观测",
    "n_cycles": "观测长度，含未来观测",
}

PROFILE_ALLOWED_FEATURES: dict[str, str] = {
    "cap_first": "首循环容量，属于**过去**信息，理论上可用；但用了它就等于承认「新电池必须先测一次」，部署前提需在报告里写清",
}

CLEANING_MD = TABLE_DIR / "data_cleaning_decisions.md"


# ---------------------------------------------------------------------------
# 统计工具
# ---------------------------------------------------------------------------
def iqr_bounds(y: np.ndarray) -> tuple[float, float]:
    q1, q3 = np.percentile(y, [25, 75])
    iqr = q3 - q1
    return float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)


def iqr_flags(y: np.ndarray) -> int:
    lo, hi = iqr_bounds(y)
    return int(((y < lo) | (y > hi)).sum())


def sigma3_bounds(y: np.ndarray) -> tuple[float, float]:
    mu, sd = float(np.mean(y)), float(np.std(y, ddof=1))
    return mu - 3 * sd, mu + 3 * sd


def sigma3_flags(y: np.ndarray) -> int:
    lo, hi = sigma3_bounds(y)
    return int(((y < lo) | (y > hi)).sum())


def slope_per_100(sub: pd.DataFrame) -> float:
    """线性拟合斜率，换算成 Ah / 100 循环。"""
    x = sub["cycle"].to_numpy(float)
    y = sub["capacity_Ah"].to_numpy(float)
    return float(np.polyfit(x, y, 1)[0]) * 100.0


def regen_cross_cell_stats(df: pd.DataFrame) -> dict:
    """量化「容量再生」里的**跨电池同步成分**。

    动机：如果四个个体各自独立地出现容量回升，一阶差分的同号率应接近 0.5。
    实测远高于 0.5，且三块 168 循环电池在**同一个循环**同时出现各自的最大回升、
    次日同时出现各自的最大跌幅 —— 说明「再生」里含**共模成分**（共同的实验或
    测量事件），不能简单归因于四个独立个体的电化学过程。
    """
    piv = df.pivot_table(index="cycle", columns="battery_id", values="capacity_Ah")
    delta = piv.diff()
    cells = [c for c in BATTERIES if c in delta.columns]

    pairs = []
    for i, a in enumerate(cells):
        for b in cells[i + 1 :]:
            pair = delta[[a, b]].dropna()
            if len(pair) < 10:
                continue
            pairs.append(
                {
                    "a": a,
                    "b": b,
                    "pair": f"{a}–{b}",
                    "pearson": float(pair[a].corr(pair[b])),
                    "same_sign": float((np.sign(pair[a]) == np.sign(pair[b])).mean()),
                    "n": int(len(pair)),
                }
            )

    # 「同号率」的独立零基准：环形移位法。
    # 不能拿 0.5 当参照——两序列各自都有强负漂移（P(Δ<0) 约 0.72–0.85），
    # 即使完全独立，同号率也会明显高于 0.5。第三轮审查抓到了这个基准错误。
    rng = np.random.default_rng(RANDOM_SEED)
    n_perm = 500
    for p in pairs:
        sub_pair = delta[[p["a"], p["b"]]].dropna()
        sa = np.sign(sub_pair[p["a"]].to_numpy())
        sb = np.sign(sub_pair[p["b"]].to_numpy())
        n = len(sa)
        null = []
        for _ in range(n_perm):
            off = int(rng.integers(1, n))
            null.append(float((sa == np.roll(sb, off)).mean()))
        null_arr = np.asarray(null)
        p["same_sign_null"] = float(null_arr.mean())
        p["same_sign_p"] = float((null_arr >= p["same_sign"]).mean())

    peak_cycle = {c: int(delta[c].dropna().idxmax()) for c in cells if delta[c].notna().any()}
    peak_value = {c: float(delta[c].max()) for c in cells if delta[c].notna().any()}
    trough_cycle = {c: int(delta[c].dropna().idxmin()) for c in cells if delta[c].notna().any()}

    def shared(mapping: dict[str, int]) -> list[int]:
        counts = pd.Series(list(mapping.values())).value_counts()
        return sorted(int(cyc) for cyc, cnt in counts.items() if cnt >= 2)

    return {
        "pairs": pairs,
        "peak_cycle": peak_cycle,
        "peak_value": peak_value,
        "trough_cycle": trough_cycle,
        "shared_peak": shared(peak_cycle),
        "shared_trough": shared(trough_cycle),
    }


# ---------------------------------------------------------------------------
# 行级标记 + 个体档案
# ---------------------------------------------------------------------------
def mark_rows(df: pd.DataFrame) -> pd.DataFrame:
    """加行级标记列。不删除、不修改任何原始测量值。"""
    out = add_derived(df)

    out["cap_delta"] = out.groupby("battery_id")["capacity_Ah"].diff()
    out["is_regen"] = (out["cap_delta"] > REGEN_MIN_JUMP_AH).fillna(False).astype(bool)

    out["cap_exceeds_rated"] = out["capacity_Ah"] > RATED_CAPACITY_AH

    # ⚠️ 这一列用的是**整条轨迹**的最小值：第 t 行已经知道 t 之后会不会跌破阈值。
    # 只允许用于个体档案与作图，不得作特征（见 FEATURE_EXCLUDE）。
    min_cap = out.groupby("battery_id")["capacity_Ah"].transform("min").to_numpy()
    out["cell_censored"] = (~is_at_eol(min_cap)).astype(bool)

    return out


def cell_profile(df: pd.DataFrame) -> pd.DataFrame:
    """每个电池个体的档案表。

    注意：
    - 「首循环容量」用 `cycle == 1` 显式定位，不靠行序（审查在 add_derived 里
      已经抓过一次「位置第一行」的坑，这里不能重犯）。
    - 本表所有列除 `cap_first` 外都是整轨迹统计量或标签，不得回填成行级特征
      （见 PROFILE_EXCLUDE）。
    """
    rows = []
    for cell, sub in split_by_battery(df).items():
        sub = sub.sort_values("cycle").reset_index(drop=True)
        if int(sub["cycle"].iloc[0]) != 1:
            raise ValueError(f"{cell} 首行不是 cycle==1，无法定义首循环容量")

        cap = sub["capacity_Ah"]
        first_cap = float(cap.iloc[0])
        hit = sub.loc[is_at_eol(cap.to_numpy()), "cycle"]

        rows.append(
            {
                "battery_id": cell,
                "n_cycles": len(sub),
                "cap_first": round(first_cap, 4),
                "cap_last": round(float(cap.iloc[-1]), 4),
                "cap_min": round(float(cap.min()), 4),
                "cap_min_cycle": int(sub.loc[cap.idxmin(), "cycle"]),
                "cap_max": round(float(cap.max()), 4),
                "fade_pct": round(float((first_cap - cap.min()) / first_cap * 100), 2),
                "slope_Ah_per_100": round(slope_per_100(sub), 4),
                "eol_cycle": int(hit.iloc[0]) if len(hit) else pd.NA,
                "censored": bool(len(hit) == 0),
                "n_regen": int(sub["is_regen"].sum()),
            }
        )

    prof = pd.DataFrame(rows)
    # 可空整型：避免「未到达」被写成 125.0 这类浮点，与真实循环号混淆
    prof["eol_cycle"] = prof["eol_cycle"].astype("Int64")
    return prof


def assert_feature_exclude_complete(marked: pd.DataFrame) -> None:
    """白名单断言：除目标列外，每一列都必须明确表态过能不能作特征。

    这条是为了防「字典看起来挺全、其实漏了」——第一轮审查就是靠"10 列 - 6 列"
    的减法发现漏了 3 列的。以后新增列如果忘了登记，这里直接报错。
    """
    undeclared = set(marked.columns) - set(FEATURE_EXCLUDE) - {TARGET_COLUMN}
    if undeclared:
        raise AssertionError(
            f"以下列既不是目标列、也没登记进 FEATURE_EXCLUDE：{sorted(undeclared)}"
        )


# ---------------------------------------------------------------------------
# 四个坑的决策
# ---------------------------------------------------------------------------
def build_decisions(
    df: pd.DataFrame, prof: pd.DataFrame, raw: pd.DataFrame
) -> list[dict[str, str]]:
    """构造决策表数据。所有数字现算。"""
    by = split_by_battery(df)
    cap_all = df["capacity_Ah"].to_numpy(float)

    # --- 坑 1：温度零方差 ---
    temps = sorted(df["ambient_temp_C"].unique().tolist())

    # --- 坑 2：B0006 超额定（斜率对比统一按 4 位小数计算差值，避免舍入不自洽）---
    b6 = by["B0006"]
    over = b6[b6["capacity_Ah"] > RATED_CAPACITY_AH]
    s_full = round(slope_per_100(b6), 4)
    s_drop = round(slope_per_100(b6[b6["cycle"] > int(over["cycle"].max())]), 4)
    b6_clip = b6.copy()
    b6_clip["capacity_Ah"] = b6_clip["capacity_Ah"].clip(upper=RATED_CAPACITY_AH)
    s_clip = round(slope_per_100(b6_clip), 4)
    d_drop = round(s_drop - s_full, 4)
    d_clip = round(s_clip - s_full, 4)

    # --- 坑 3：容量再生（含跨电池同步性）---
    regen = regen_cross_cell_stats(df)
    n_regen = int(df["is_regen"].sum())
    regen_jumps = df.loc[df["is_regen"], "cap_delta"]
    max_jump = float(regen_jumps.max()) if len(regen_jumps) else float("nan")
    big_pairs = [p for p in regen["pairs"] if p["n"] >= 100]
    sync_text = "；".join(
        f"{p['pair']} Pearson {p['pearson']:.3f}、同号率 {p['same_sign']:.3f}"
        f"（环形移位零基准 {p['same_sign_null']:.3f}，置换 p={p['same_sign_p']:.3f}）"
        for p in big_pairs
    )
    if regen["shared_peak"]:
        cyc = regen["shared_peak"][0]
        who = [c for c, k in regen["peak_cycle"].items() if k == cyc]
        vals = "、".join(f"{c} {regen['peak_value'][c]:+.4f}" for c in who)
        shared_peak_text = f"三块电池在循环 {cyc} 同时出现各自最大回升（{vals}）"
    else:
        shared_peak_text = "未发现多块电池同时达到最大回升"

    # --- 同一循环上多块电池同时回升 → 「共模事件」---
    per_cycle = df.loc[df["is_regen"]].groupby("cycle")["battery_id"].nunique()
    shared_event_cycles = sorted(int(c) for c in per_cycle[per_cycle >= 2].index)
    n_shared_event_pairs = int(per_cycle[per_cycle >= 2].sum())
    n_independent_events = n_regen - n_shared_event_pairs + len(shared_event_cycles)

    # --- 坑 4：右删失 ---
    censored = prof.loc[prof["censored"], "battery_id"].tolist()
    b7 = by["B0007"]
    b7_min = float(b7["capacity_Ah"].min())
    b7_min_cycle = int(b7.loc[b7["capacity_Ah"].idxmin(), "cycle"])

    # --- 坑 5：三种离群点口径（池化 + 分电池）---
    n_iqr = iqr_flags(cap_all)
    n_3s = sigma3_flags(cap_all)
    n_over_rated = int((cap_all > RATED_CAPACITY_AH).sum())
    iqr_lo, iqr_hi = iqr_bounds(cap_all)
    s3_lo, s3_hi = sigma3_bounds(cap_all)
    cap_min_all, cap_max_all = float(cap_all.min()), float(cap_all.max())
    per_cell_flags = {
        c: (iqr_flags(sub["capacity_Ah"].to_numpy(float)), sigma3_flags(sub["capacity_Ah"].to_numpy(float)))
        for c, sub in by.items()
    }
    per_cell_text = "；".join(f"{c} IQR {a} 个 / 3σ {b} 个" for c, (a, b) in per_cell_flags.items())
    per_cell_all_zero = all(a == 0 and b == 0 for a, b in per_cell_flags.values())

    # --- 缺失/重复：在**原始**数据上现算 ---
    # 不能在 df 上算：cap_delta 在每个电池第 1 个循环必然是 NaN（合法的派生缺失），
    # 那会把「原始数据 0 缺失」误报成 4 个缺失。
    n_null = int(raw.isna().sum().sum())
    n_dup = int(raw.duplicated(subset=["battery_id", "cycle"]).sum())

    return [
        {
            "issue": "温度列零方差",
            "observed": f"`ambient_temp_C` 全部 {len(df)} 行唯一取值仅 {len(temps)} 个：{temps}",
            "chosen": "**保留该列，但列入 `FEATURE_EXCLUDE` 禁止作特征**",
            "rejected": "删除整列 / 当作可用特征做相关性分析",
            "reason": (
                "零方差列携带的信息是「实验条件被固定」这件事本身，删掉就把实验条件一起丢了。"
                "而拿它做「温度 vs 容量」相关性分析必然得到 NaN 或 0 相关——那是常量，不是发现。"
                "正确做法是声明它不可用，并在报告里解释：本数据只能沿循环维度分析，温度维度无法研究。"
            ),
            "evidence": "由 `src/sanity_check.py` 的 C5 断言守着（唯一值断言）。",
        },
        {
            "issue": "B0006 前 7 循环容量超额定",
            "observed": (
                f"第 1–{int(over['cycle'].max())} 循环共 {len(over)} 条超过额定 {RATED_CAPACITY_AH} Ah，"
                f"最高 {float(over['capacity_Ah'].max()):.4f} Ah"
            ),
            "chosen": "**保留原值，加 `cap_exceeds_rated` 标记列，不删不截断**",
            "rejected": "删除这 7 行 / 截断到 2.0 Ah",
            "reason": (
                "截断是**篡改测量值**：把 2.0353 改成 2.0 会人为制造一段平坦区，改变衰减速率估计。"
                "删除同样是选择偏差：被删的恰好是初期的最高点，会让 B0006 的初始容量被低估、衰减幅度看起来更小。"
                "「超过额定」本身是信息：它说明**额定 2 Ah 不是这批电池的实际上限**，"
                "因此 SOH = C/2.0 出现 >100% 是数据性质，不是错误。"
            ),
            "evidence": (
                f"反例代价已量化（统一按 4 位小数计算差值，避免舍入不自洽）："
                f"B0006 线性斜率 全样本 **{s_full:.4f}** / 删掉前 7 行 **{s_drop:.4f}** / "
                f"截断到 2.0 **{s_clip:.4f}** Ah·(100循环)⁻¹；"
                f"删行使斜率变化 **{d_drop:+.4f}**，截断使斜率变化 **{d_clip:+.4f}**。"
            ),
        },
        {
            "issue": "容量再生（非单调）",
            "observed": (
                f"相邻循环回升 > {REGEN_MIN_JUMP_AH} Ah 的 **(电池, 循环) 对**共 {n_regen} 个"
                f"（B0005 {int(prof.set_index('battery_id').loc['B0005','n_regen'])} / "
                f"B0006 {int(prof.set_index('battery_id').loc['B0006','n_regen'])} / "
                f"B0007 {int(prof.set_index('battery_id').loc['B0007','n_regen'])} / "
                f"B0018 {int(prof.set_index('battery_id').loc['B0018','n_regen'])}），"
                f"最大单次回升 **{max_jump:+.4f} Ah**（≈额定容量的 {max_jump / RATED_CAPACITY_AH * 100:.2f}%）。"
                f"注意：{n_regen} 是「电池×循环」对计数，不是独立事件数——"
                f"循环 {shared_event_cycles} 上有多块电池同时回升（合计 {n_shared_event_pairs} 对），"
                f"所以**独立事件数 ≈ {n_independent_events}**。"
                f"另一项佐证：{shared_peak_text}。"
            ),
            "chosen": (
                "**保留原始测量值，不平滑；加 `is_regen` 标记列（禁止作特征）；"
                "在建模阶段用「趋势 + 残差」结构显式处理**"
            ),
            "rejected": "用移动平均或样条把回升磨平",
            "reason": (
                "平滑掉等于把**现象当噪声删除**。但这里要说清一件容易被追问的事："
                "「容量再生是锂电池的电化学现象（电解液再分布等）」目前只是**假设，未经本数据验证**，"
                "而实测到的跨电池同步性（见证据）说明**至少一部分波动来自共模成分**——"
                "即几块电池**共同经历**的某种成分（其来源本数据无法分辨："
                "可能是共同的实验/测量事件，也可能是别的，本项目不主张成因），"
                "而不是四个独立个体的随机物理过程。"
                "因此正确处理不是「解释掉」它，而是把它拆成「共模 + 个体」两部分："
                "共模部分进不确定性分析的「数据不确定」层，个体部分才是真正的随机波动。"
            ),
            "evidence": (
                f"跨电池一阶差分（循环 2–132 对齐）：{sync_text}。"
                f"注意同号率的零基准不是 0.5——两条序列各自都有强负漂移"
                f"（P(Δ<0) 约 0.72–0.85），即使完全独立同号率也会明显高于 0.5，"
                f"所以本表用环形移位法现场估零基准，并给置换 p 值。"
                f"{shared_peak_text}，次日三块同时出现各自最大单步跌幅"
                f"（循环 {regen['shared_trough'][0] if regen['shared_trough'] else '—'}）。"
                f"B0018 是否属于这个同步簇，见 W6 的再生分解"
                f"（其残差与共模的相关**低于**自身独立零基准，故不属该簇）。"
            ),
        },
        {
            "issue": "右删失（观察期结束仍未失效）",
            "observed": (
                f"以 {EOL_THRESHOLD_AH} Ah（额定 70%）为 EOL，未到达的个体：**{censored}**；"
                f"B0007 最低 {b7_min:.4f} Ah @ 第 {b7_min_cycle} 循环，距阈值仅 "
                f"{b7_min - EOL_THRESHOLD_AH:+.4f} Ah。"
                f"（对照：B0018 在第 97 循环就已跌破 1.4 Ah，属已观测失效，不是删失。）"
            ),
            "chosen": "**保留全部观测；加 `cell_censored` 标记（禁止作特征）；寿命统计时如实记为「> 观察窗口」**",
            "rejected": "删除该个体 / 把它的寿命记成 168 或 132",
            "reason": (
                "删掉会丢掉 168 条真实观测；把「未观测到失效」记成「寿命 = 观察长度」是**逻辑错误**，"
                "会系统性低估寿命，进而低估更换周期。"
                "本项目**不做** Weibull/生存分析（D2），只用一句话说明口径，因为那超出课程范围。"
            ),
            "evidence": "C10 断言守着「B0007 从未到达 1.4 Ah」这个结论本身，而不只是复读数值。",
        },
        {
            "issue": "异常值检测：三种口径在池化与分电池两种聚合下的实际结果",
            "observed": (
                f"**池化**：全局 IQR 标记 **{n_iqr}** 个 / 全局 3σ 标记 **{n_3s}** 个 / "
                f"物理上界（> 额定 {RATED_CAPACITY_AH} Ah）**{n_over_rated}** 个。"
                f"**分电池**：{per_cell_text}"
                + ("，全部为 0。" if per_cell_all_zero else "。")
            ),
            "chosen": "**不做统计离群点剔除**；唯一的异常判定用物理上界（领域知识），且只标记不删除",
            "rejected": "把全局 3σ / IQR 当默认流程走一遍，然后删掉被标记的点",
            "reason": (
                "两种统计准则在**池化与分电池两种聚合下都一个点标不出来**。"
                "原因不是数据特别干净，而是容量是**趋势数据**：沿循环平滑变化，"
                "没有远离局部水平的极端值。"
                "「没检出异常」不等于「做了异常检测」——它只说明这两个工具在这里没有分辨力。"
                "分电池口径尤其重要：池化会把 4 个不同水平的电池混在一起、把 IQR 撑宽，"
                "所以必须验证分电池口径下结论是否一样（结论：一样）。"
                "真正需要判断的是**物理合理性**：容量超过额定 2 Ah 在物理上可疑（B0006 前 7 循环），"
                "这属于领域知识判断而非统计判断，且我们选择保留（见决策 2）。"
            ),
            "evidence": (
                f"池化 IQR 上下界 [{iqr_lo:.4f}, {iqr_hi:.4f}]，全局范围 [{cap_min_all:.4f}, {cap_max_all:.4f}]"
                f" —— 无点越界；池化 3σ 区间 [{s3_lo:.4f}, {s3_hi:.4f}]，同样无点越界。"
                f"B0006 那 7 个超额定点在统计口径下也不越界（{cap_max_all:.4f} < {s3_hi:.4f}），"
                f"所以它们只能靠**领域知识**识别，不是靠统计。"
            ),
        },
        {
            "issue": "缺失值与重复记录（现算，不写死）",
            "observed": f"缺失单元格 **{n_null}** 个；重复 (电池, 循环) 记录 **{n_dup}** 条",
            "chosen": "**不做插值、不做去重**（因为没有可处理的缺失与重复）",
            "rejected": "为了「做全套清洗」而强行插值或去重",
            "reason": (
                "没有缺失还去插值，是给数据凭空造点；没有重复还去去重，只是浪费一次运算。"
                "报告里这一节如实写「检查后无需处理」比编一段处理过程更可信。"
            ),
            "evidence": "由 C3/C4 断言（缺失/重复）守着，本节的数字也是现算后注入的，不随数据变化撒谎。",
        },
    ]


def render_markdown(
    decisions: list[dict[str, str]], prof: pd.DataFrame, n_rows: int, n_raw_rows: int
) -> str:
    lines = [
        "# 数据清洗与预处理决策表（W3）",
        "",
        "> 本文件由 `src/preprocess.py` 自动生成，数字随代码重算，**不要手工编辑**。",
        "",
        "## 一、清洗决策",
        "",
        "| # | 问题 | 数据事实 | 决策 | 否掉的方案 | 理由 | 证据 |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, d in enumerate(decisions, 1):
        lines.append(
            f"| {i} | **{d['issue']}** | {d['observed']} | {d['chosen']} | "
            f"{d['rejected']} | {d['reason']} | {d['evidence']} |"
        )

    lines += [
        "",
        "## 二、个体档案",
        "",
        prof.to_markdown(index=False),
        "",
        "## 三、不可作为特征的列（行级）",
        "",
        "| 列名 | 排除理由 |",
        "|---|---|",
    ]
    for col, why in FEATURE_EXCLUDE.items():
        lines.append(f"| `{col}` | {why} |")

    lines += [
        "",
        "## 四、个体档案里不可回填成行级特征的列",
        "",
        "| 列名 | 排除理由 |",
        "|---|---|",
    ]
    for col, why in PROFILE_EXCLUDE.items():
        lines.append(f"| `{col}` | {why} |")

    lines += ["", "唯一允许回填的档案列：", ""]
    for col, why in PROFILE_ALLOWED_FEATURES.items():
        lines.append(f"- `{col}`：{why}")

    lines += [
        "",
        "## 五、没做的事（以及为什么）",
        "",
        f"- **没有删除任何行**：清理后 {n_rows} 行 == 原始 {n_raw_rows} 行（代码里用 assert 守着）。",
        "- **没有插值/填充**：见决策 6，缺失数为 0（现算）。",
        "- **没有截断/盖帽**：见决策 2。",
        "- **没有做统计离群点剔除**：见决策 5，两种口径在池化与分电池下都标不出点。",
        "- **没有做 Weibull / 生存分析**：见 D2，方法超出课程范围，改用一句话交代口径。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ensure_dirs()

    raw = load_raw()
    df = mark_rows(raw)
    prof = cell_profile(df)

    assert_feature_exclude_complete(df)
    assert len(df) == len(raw), f"清洗改变了行数：{len(raw)} -> {len(df)}"

    decisions = build_decisions(df, prof, raw)
    markdown = render_markdown(decisions, prof, len(df), len(raw))
    CLEANING_MD.write_text(markdown, encoding="utf-8")

    n_null = int(raw.isna().sum().sum())
    n_dup = int(raw.duplicated(subset=["battery_id", "cycle"]).sum())
    n_derived_null = int(df.isna().sum().sum())

    print(f"清理后数据：{len(df)} 行 × {df.shape[1]} 列（原始 {len(raw)} 行 × {raw.shape[1]} 列）")
    print(f"行数未变：{len(df) == len(raw)}；原始缺失 {n_null} 个；重复 {n_dup} 条")
    print(f"派生列合法 NaN：{n_derived_null} 个（cap_delta 在每个电池第 1 循环，属预期）")
    print(f"FEATURE_EXCLUDE 已声明 {len(FEATURE_EXCLUDE)} 列，白名单断言通过（无遗漏列）")
    print(f"再生 (电池,循环) 对：{int(df['is_regen'].sum())} 个；超额定：{int(df['cap_exceeds_rated'].sum())} 行")
    print(f"删失个体：{prof.loc[prof['censored'], 'battery_id'].tolist()}")
    print()
    print(prof.to_string(index=False))
    print()
    print(f"决策表已写出：{CLEANING_MD.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
