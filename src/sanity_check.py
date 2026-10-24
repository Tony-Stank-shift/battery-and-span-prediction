"""数据体检：把这份数据的每一条事实做成**可复现的断言**。

动机：这份 CSV 表面上"很干净"（0 缺失、0 重复），但里面埋着若干个会直接
毁掉分析的坑（温度零方差、容量超额定、容量再生、删失个体）。这些坑不能靠
肉眼记，必须靠断言守住——换个人跑一遍，结论必须一样。

运行
----
    python src/sanity_check.py

退出码
------
    0 = 全部断言通过
    1 = 至少一条断言失败（并打印是哪条）
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
)
from load_data import EXPECTED_COLUMNS, add_derived, load_raw, split_by_battery  # noqa: E402

# 期望值——全部来自对原始 CSV 的实测，不是猜的
EXP_RECORDS = 636
EXP_CYCLES = {"B0005": 168, "B0006": 168, "B0007": 168, "B0018": 132}
EXP_AMBIENT_TEMPS = [24.0]
EXP_OVERRATED_CYCLES = 7          # B0006 第 1–7 循环超过额定 2.0 Ah
EXP_OVERRATED_MAX = 2.0353
EXP_REGEN_EVENTS = {"B0005": 2, "B0006": 4, "B0007": 1, "B0018": 5}
EXP_REGEN_MAX_JUMP = 0.1519
EXP_EOL_AT_1_4 = {"B0005": 125, "B0006": 109, "B0007": None, "B0018": 97}
EXP_B0007_MIN = 1.4005

_results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, bool(ok), detail))


def regen_events(sub: pd.DataFrame) -> tuple[list[int], list[float]]:
    """返回 (发生回升的循环号列表, 回升幅度列表)。"""
    cap = sub["capacity_Ah"].to_numpy()
    cyc = sub["cycle"].to_numpy()
    delta = np.diff(cap)
    idx = np.where(delta > REGEN_MIN_JUMP_AH)[0]
    return [int(cyc[i + 1]) for i in idx], [float(delta[i]) for i in idx]


def eol_cycle(sub: pd.DataFrame, tau: float) -> int | None:
    """容量首次跌破 tau 的循环号；从未跌破则返回 None（右删失）。"""
    hit = sub[sub["capacity_Ah"] <= tau]
    return int(hit["cycle"].iloc[0]) if len(hit) else None


def run() -> None:
    raw = load_raw()
    df = add_derived(raw)
    by = split_by_battery(df)

    # ---- C1 规模 ----
    check(
        "C1 数据规模 = 636 行 × 4 列",
        raw.shape == (EXP_RECORDS, len(EXPECTED_COLUMNS)),
        f"实际 {raw.shape[0]} 行 × {raw.shape[1]} 列",
    )

    # ---- C2 列名 ----
    check(
        "C2 列名与数据说明一致",
        list(raw.columns) == EXPECTED_COLUMNS,
        f"实际 {list(raw.columns)}",
    )

    # ---- C3 缺失值 ----
    n_null = int(raw.isna().sum().sum())
    check("C3 无缺失值", n_null == 0, f"缺失单元格 {n_null} 个")

    # ---- C4 重复记录 ----
    n_dup = int(raw.duplicated(subset=["battery_id", "cycle"]).sum())
    check("C4 无重复 (电池, 循环) 记录", n_dup == 0, f"重复 {n_dup} 条")

    # ---- C5 温度零方差（最大的坑）----
    temps = sorted(raw["ambient_temp_C"].unique().tolist())
    check(
        "C5 ambient_temp_C 是零方差列（唯一值仅 1 个）",
        len(temps) == len(EXP_AMBIENT_TEMPS) and np.allclose(temps, EXP_AMBIENT_TEMPS),
        f"唯一取值 {temps} → 不能当特征，任何「温度 vs 容量」相关性分析都会退化",
    )

    # ---- C6 个体与循环数 ----
    got_cycles = {b: int(len(sub)) for b, sub in by.items()}
    check(
        "C6 4 个电池个体，循环数 168/168/168/132",
        list(by.keys()) == BATTERIES and got_cycles == EXP_CYCLES,
        f"实际 {got_cycles}",
    )

    # ---- C7 循环号连续无断档 ----
    gaps = {}
    for b, sub in by.items():
        cyc = sub["cycle"].tolist()
        if cyc != list(range(1, len(cyc) + 1)):
            gaps[b] = "有断档"
    check("C7 各电池循环号 1..n 连续无断档", not gaps, f"异常 {gaps}" if gaps else "全部连续")

    # ---- C8 容量超额定（B0006 前 7 循环）----
    b6 = by["B0006"]
    over = b6[b6["capacity_Ah"] > 2.0]
    check(
        "C8 B0006 前 7 循环容量超过额定 2.0 Ah",
        len(over) == EXP_OVERRATED_CYCLES
        and over["cycle"].tolist() == list(range(1, EXP_OVERRATED_CYCLES + 1))
        and np.isclose(over["capacity_Ah"].max(), EXP_OVERRATED_MAX, atol=1e-6),
        f"{len(over)} 条，最高 {over['capacity_Ah'].max():.4f} Ah（物理上可疑，需在清洗章节说明）",
    )

    # ---- C9 容量再生事件 ----
    got_events = {b: len(regen_events(sub)[0]) for b, sub in by.items()}
    all_jumps = [j for sub in by.values() for j in regen_events(sub)[1]]
    check(
        "C9 容量再生事件数 = 2/4/1/5，最大回升 0.1519 Ah",
        got_events == EXP_REGEN_EVENTS
        and np.isclose(max(all_jumps), EXP_REGEN_MAX_JUMP, atol=1e-4),
        f"实际 {got_events}，最大回升 {max(all_jumps):.4f} Ah（是物理现象，不是噪声）",
    )

    # ---- C10 EOL 与删失 ----
    got_eol = {b: eol_cycle(sub, EOL_THRESHOLD_AH) for b, sub in by.items()}
    b7_min = float(by["B0007"]["capacity_Ah"].min())
    check(
        "C10 EOL(1.4 Ah) = 125/109/未到达/97，B0007 为右删失",
        got_eol == EXP_EOL_AT_1_4 and np.isclose(b7_min, EXP_B0007_MIN, atol=1e-6),
        f"实际 {got_eol}；B0007 最低 {b7_min:.4f} Ah，距阈值仅差 {b7_min - EOL_THRESHOLD_AH:.4f} Ah",
    )


def report() -> int:
    width = max(len(n) for n, _, _ in _results) + 2
    print("=" * 78)
    print("数据体检报告")
    print("=" * 78)
    for name, ok, detail in _results:
        flag = "PASS" if ok else "FAIL"
        print(f"[{flag}] {name:<{width}} {detail}")
    n_fail = sum(1 for _, ok, _ in _results if not ok)
    print("-" * 78)
    print(f"共 {len(_results)} 条断言：{len(_results) - n_fail} 通过，{n_fail} 失败")
    print("=" * 78)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    run()
    sys.exit(report())
