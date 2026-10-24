"""清洗与预处理（W3）。

这份数据的"干净"是假象：0 缺失、0 重复，但埋着四个坑。本模块做两件事：

1. 把四个坑的**处理决策**固化成代码，并生成决策表
   （`outputs/tables/data_cleaning_decisions.md`）——决策表里的每个数字都是
   现算的，不是手写的，避免文档和代码说两套话。
2. 加行级标记列，供后续建模使用。

刻意不做的事（都是"看起来合理但错"的做法）
------------------------------------------
- **不删任何行**：删行会引入选择偏差。
- **不截断任何测量值**：把 2.0353 截成 2.0 等于篡改测量，会人为制造平坦段。
- **不用全局 3σ / IQR 判异常**：容量是**趋势数据**。全局统计量会把"晚期真实
  的低容量"当成异常点。正确做法是在**残差域**（相对局部趋势的偏差）检测。
  本模块把"错误做法的后果"量化出来放进决策表，作为证据。

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
    REGEN_MIN_JUMP_AH,
    RATED_CAPACITY_AH,
    TABLE_DIR,
    ensure_dirs,
)
from load_data import add_derived, load_raw, split_by_battery  # noqa: E402

# ---------------------------------------------------------------------------
# 明确不可作为特征的列（原因见每一行注释）
# ---------------------------------------------------------------------------
FEATURE_EXCLUDE: dict[str, str] = {
    "battery_id": "标识符，非数值特征（个体差异由交叉验证处理，不靠编码进模型）",
    "cycle": "时间索引，不是可泛化的特征；模型用它作为自变量时要声明这是外推任务",
    "ambient_temp_C": "全表唯一值 24.0，零方差，无信息量",
    "cap_over_rated": "capacity_Ah 的精确仿射变换（×1/2.0），作特征即标签泄漏",
    "cap_over_first": "capacity_Ah 的精确仿射变换（×1/首循环容量），作特征即标签泄漏",
    "cap_delta": "由当前容量差分得到，含当前与上一时刻信息，作特征需先确认因果方向",
}

CLEANING_MD = TABLE_DIR / "data_cleaning_decisions.md"


# ---------------------------------------------------------------------------
# 错误做法的量化（用来在决策表里做反例证据）
# ---------------------------------------------------------------------------
def iqr_flags(y: np.ndarray) -> int:
    """全局 IQR 准则标记的"异常点"个数（在本数据上会误判，仅用于反例）。"""
    q1, q3 = np.percentile(y, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((y < lo) | (y > hi)).sum())


def sigma3_flags(y: np.ndarray) -> int:
    """全局 3σ 准则标记的"异常点"个数（同上，仅用于反例）。"""
    mu, sd = float(np.mean(y)), float(np.std(y, ddof=1))
    return int((np.abs(y - mu) > 3 * sd).sum())


def slope_per_100(sub: pd.DataFrame) -> float:
    """线性拟合斜率，换算成 Ah / 100 循环。"""
    x = sub["cycle"].to_numpy(float)
    y = sub["capacity_Ah"].to_numpy(float)
    beta = float(np.polyfit(x, y, 1)[0])
    return beta * 100.0


# ---------------------------------------------------------------------------
# 行级标记 + 个体档案
# ---------------------------------------------------------------------------
def mark_rows(df: pd.DataFrame) -> pd.DataFrame:
    """加行级标记列。不删除、不修改任何原始测量值。"""
    out = add_derived(df)

    out["cap_delta"] = out.groupby("battery_id")["capacity_Ah"].diff()
    out["is_regen"] = out["cap_delta"] > REGEN_MIN_JUMP_AH
    out["is_regen"] = out["is_regen"].fillna(False).astype(bool)

    out["cap_exceeds_rated"] = out["capacity_Ah"] > RATED_CAPACITY_AH

    # 删失标记：该电池在整个观察期内是否从未跌破 EOL 阈值
    never_crossed = (
        out.groupby("battery_id")["capacity_Ah"].transform("min") > EOL_THRESHOLD_AH
    )
    out["cell_censored"] = never_crossed.astype(bool)

    return out


def cell_profile(df: pd.DataFrame) -> pd.DataFrame:
    """每个电池个体的档案表（W5「个体差异五维量化」会用到这里的一部分）。"""
    rows = []
    for cell, sub in split_by_battery(df).items():
        cap = sub["capacity_Ah"]
        imin = cap.idxmin()
        crossed = sub[cap <= EOL_THRESHOLD_AH]
        rows.append(
            {
                "battery_id": cell,
                "n_cycles": len(sub),
                "cap_first": round(float(cap.iloc[0]), 4),
                "cap_last": round(float(cap.iloc[-1]), 4),
                "cap_min": round(float(cap.min()), 4),
                "cap_min_cycle": int(sub.loc[imin, "cycle"]),
                "cap_max": round(float(cap.max()), 4),
                "fade_pct": round(float((cap.iloc[0] - cap.min()) / cap.iloc[0] * 100), 2),
                "slope_Ah_per_100": round(slope_per_100(sub), 4),
                "eol_cycle_at_1p4": int(crossed["cycle"].iloc[0]) if len(crossed) else pd.NA,
                "censored": bool(len(crossed) == 0),
                "n_regen": int(sub["is_regen"].sum()),
            }
        )
    prof = pd.DataFrame(rows)
    # 可空整型：避免「未到达」被写成 125.0 这类浮点，与真实循环号混淆
    prof["eol_cycle_at_1p4"] = prof["eol_cycle_at_1p4"].astype("Int64")
    return prof


# ---------------------------------------------------------------------------
# 四个坑的决策
# ---------------------------------------------------------------------------
def build_decisions(df: pd.DataFrame, prof: pd.DataFrame) -> list[dict[str, str]]:
    """构造决策表数据。所有数字现算。"""
    by = split_by_battery(df)
    cap_all = df["capacity_Ah"].to_numpy(float)

    # --- 坑 1：温度零方差 ---
    temps = sorted(df["ambient_temp_C"].unique().tolist())

    # --- 坑 2：B0006 超额定 ---
    b6 = by["B0006"]
    over = b6[b6["capacity_Ah"] > RATED_CAPACITY_AH]
    slope_full = slope_per_100(b6)
    slope_drop = slope_per_100(b6[b6["cycle"] > int(over["cycle"].max())])
    b6_clip = b6.copy()
    b6_clip["capacity_Ah"] = b6_clip["capacity_Ah"].clip(upper=RATED_CAPACITY_AH)
    slope_clip = slope_per_100(b6_clip)

    # --- 坑 3：容量再生 ---
    n_regen = int(df["is_regen"].sum())
    regen_jumps = df.loc[df["is_regen"], "cap_delta"]
    max_jump = float(regen_jumps.max()) if len(regen_jumps) else float("nan")

    # --- 坑 4：右删失 ---
    censored = prof.loc[prof["censored"], "battery_id"].tolist()
    b7_min = float(by["B0007"]["capacity_Ah"].min())

    # --- 统计离群点检测：三种口径在本数据上的实际结果 ---
    # 注意：这三个数字是**现算的**。写这一段时我原本的叙事（"IQR 会把晚期
    # 真实低容量误判为异常"）被这些数字直接否证了——实测 IQR 与 3σ 都标记 0 个点。
    # 结论因此改写：不是"这两个工具会把好数据误杀"，而是"它们在这份数据上没有分辨力"。
    n_iqr = iqr_flags(cap_all)
    n_3s = sigma3_flags(cap_all)
    n_over_rated = int((cap_all > RATED_CAPACITY_AH).sum())
    q1, q3 = np.percentile(cap_all, [25, 75])
    iqr_lo = float(q1 - 1.5 * (q3 - q1))
    mu = float(np.mean(cap_all))
    sd = float(np.std(cap_all, ddof=1))
    sigma_lo, sigma_hi = mu - 3 * sd, mu + 3 * sd
    cap_min_all = float(cap_all.min())
    cap_max_all = float(cap_all.max())

    return [
        {
            "issue": "温度列零方差",
            "observed": f"`ambient_temp_C` 全部 {len(df)} 行唯一取值仅 {len(temps)} 个：{temps}",
            "chosen": "**保留该列，但列入 `FEATURE_EXCLUDE` 禁止作特征**",
            "rejected": "删除整列 / 当作可用特征做相关性分析",
            "reason": (
                "零方差列携带的信息是「实验条件被固定」这件事本身，删掉就把这个实验条件一起丢了。"
                "而拿它做「温度 vs 容量」相关性分析必然得到 NaN 或 0 相关——那是常量，不是发现。"
                "正确做法是声明它不可用，并在报告里解释：本数据只能沿循环维度做分析，温度维度无法研究。"
            ),
            "evidence": "决策表由 `src/sanity_check.py` 的 C5 断言守着（唯一值断言）。",
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
                "「超过额定」这件事本身是信息：它说明**额定 2 Ah 不是这批电池的实际上限**，"
                "因此 SOH = C/2.0 出现 >100% 是数据性质，不是错误。"
            ),
            "evidence": (
                f"反例代价已量化：B0006 线性斜率 全样本 **{slope_full:.4f}** / "
                f"删掉前 7 行 **{slope_drop:.4f}** / 截断到 2.0 **{slope_clip:.4f}** Ah·(100循环)⁻¹。"
                f"删行使斜率变化 {slope_drop - slope_full:+.4f}，截断使斜率变化 {slope_clip - slope_full:+.4f}。"
            ),
        },
        {
            "issue": "容量再生（非单调）",
            "observed": (
                f"相邻循环回升 > {REGEN_MIN_JUMP_AH} Ah 的事件共 {n_regen} 次，"
                f"最大单次回升 {max_jump:+.4f} Ah；分电池："
                + "、".join(f"{r.battery_id} {r.n_regen} 次" for r in prof.itertuples())
            ),
            "chosen": "**保留原始测量值，不平滑；加 `is_regen` 标记列；在建模阶段用「趋势 + 残差」结构显式处理**",
            "rejected": "用移动平均或样条把回升磨平",
            "reason": (
                "平滑掉等于把**物理现象当噪声删除**。锂电池在低倍率循环中确实会出现容量回升"
                "（电解液再分布等原因），把它磨平会系统性低估不确定性，"
                "让第 6 章的残差图看起来漂亮但理由是假的。"
                "正确做法是承认它、标出来，并让不确定性分析把它算进「数据不确定」那一层。"
            ),
            "evidence": f"最大回升 {max_jump:.4f} Ah ≈ 额定容量的 {max_jump / RATED_CAPACITY_AH * 100:.2f}%，不可忽略。",
        },
        {
            "issue": "右删失（观察期结束仍未失效）",
            "observed": (
                f"以 {EOL_THRESHOLD_AH} Ah（额定 70%）为 EOL，未跌破的个体：{censored}；"
                f"其中 B0007 最低 {b7_min:.4f} Ah @ 第 "
                f"{int(by['B0007'].loc[by['B0007']['capacity_Ah'].idxmin(), 'cycle'])} 循环，"
                f"距阈值仅 {b7_min - EOL_THRESHOLD_AH:.4f} Ah"
            ),
            "chosen": "**保留全部观测；加 `cell_censored` 标记；寿命统计时如实记为「> 观察窗口」**",
            "rejected": "删除该个体 / 把它的寿命记为 168 或 132",
            "reason": (
                "删掉会丢掉 168 条真实观测；把「未观测到失效」记成「寿命 = 观察长度」是**逻辑错误**，"
                "会系统性低估寿命，进而低估更换周期。"
                "本项目**不做** Weibull/生存分析（D2），只用一句话说明口径，因为那是超出课程范围的方法。"
            ),
            "evidence": "C10 断言守着「B0007 从未穿越 1.4 Ah」这个结论本身，而不只是复读数值。",
        },
        {
            "issue": "异常值检测：三种口径的实际结果",
            "observed": (
                f"全局 IQR（1.5×IQR）：标记 **{n_iqr}** 个；"
                f"全局 3σ：标记 **{n_3s}** 个；"
                f"物理上界（> 额定 {RATED_CAPACITY_AH} Ah）：**{n_over_rated}** 个"
            ),
            "chosen": "**不做统计离群点剔除**；唯一的异常判定用物理上界（领域知识），且只标记不删除",
            "rejected": "把全局 3σ / IQR 当默认流程走一遍，然后删掉被标记的点",
            "reason": (
                "两种统计准则在这份数据上**一个点都标不出来**。原因不是数据特别干净，"
                "而是容量是**趋势数据**：沿循环平滑变化，没有远离整体分布的极端值。"
                "所以「没检出异常」不等于「做了异常检测」——它只说明这两个工具在这里没有分辨力。"
                "真正需要判断的是**物理合理性**：容量超过额定 2 Ah 在物理上可疑（B0006 前 7 循环），"
                "这属于领域知识判断而非统计判断，且我们选择保留（见决策 2）。"
                "报告里要如实写：**我们不声称做过统计离群点剔除，因为在这份数据上它没有意义。**"
            ),
            "evidence": (
                f"IQR 下界 {iqr_lo:.4f} Ah < 全局最小值 {cap_min_all:.4f} Ah，无点越界；"
                f"3σ 区间 [{sigma_lo:.4f}, {sigma_hi:.4f}] 完全覆盖全局范围 "
                f"[{cap_min_all:.4f}, {cap_max_all:.4f}]，同样无点越界。"
                f"两者都判 0 个，说明它们对本数据没有分辨力，而不是本数据「没有异常」。"
            ),
        },
    ]


def render_markdown(decisions: list[dict[str, str]], prof: pd.DataFrame) -> str:
    lines = [
        "# 数据清洗与预处理决策表（W3）",
        "",
        "> 本文件由 `src/preprocess.py` 自动生成，数字随代码重算，不要手工编辑。",
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
        "## 三、不可作为特征的列",
        "",
        "| 列名 | 排除理由 |",
        "|---|---|",
    ]
    for col, why in FEATURE_EXCLUDE.items():
        lines.append(f"| `{col}` | {why} |")

    lines += [
        "",
        "## 四、没做的事（以及为什么）",
        "",
        "- **没有删除任何行**：0 缺失、0 重复，没有删除的理由；有理由的删除都会引入选择偏差。",
        "- **没有插值/填充**：不存在缺失，人为插值只会引入伪数据。",
        "- **没有截断/盖帽**：见决策 2。",
        "- **没有做 Weibull / 生存分析**：见 D2，方法超出课程范围，改用一句话交代口径。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ensure_dirs()

    raw = load_raw()
    df = mark_rows(raw)
    prof = cell_profile(df)
    decisions = build_decisions(df, prof)

    markdown = render_markdown(decisions, prof)
    CLEANING_MD.write_text(markdown, encoding="utf-8")

    print(f"清理后数据：{len(df)} 行 × {df.shape[1]} 列（原始 {raw.shape[1]} 列，未删除任何行）")
    print(f"新增标记列：cap_delta, is_regen, cap_exceeds_rated, cap_over_rated, cap_over_first, cell_censored")
    print(f"再生事件标记：{int(df['is_regen'].sum())} 次")
    print(f"超额定标记：{int(df['cap_exceeds_rated'].sum())} 行")
    print(f"删失个体：{prof.loc[prof['censored'], 'battery_id'].tolist()}")
    print()
    print(prof.to_string(index=False))
    print()
    print(f"决策表已写出：{CLEANING_MD.relative_to(CLEANING_MD.parents[2])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
