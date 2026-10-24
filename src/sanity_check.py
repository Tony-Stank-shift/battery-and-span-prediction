"""数据体检：把这份数据的每一条事实做成**可复现的断言**。

动机
----
这份 CSV 表面上"很干净"（0 缺失、0 重复），但里面埋着若干个会直接毁掉分析的坑
（温度零方差、容量超额定、容量再生、删失个体）。这些坑不能靠肉眼记，必须靠断言
守住——换个人跑一遍，结论必须一样。

最后一个：
    EOL 判定统一走 `config.is_at_eol()`，本文件不再硬编码比较符。
    **本文件有意绑定 τ=1.4（NASA 惯例）与 D4 的 τ 扫描解耦**：
    早期版本有过一条 C12 断言“config 的 EOL 阈值必须等于 1.4”，第一轮审查指出
    那条会把数据健康信号和调参信号混在一起——W7 一旦要扫 τ，最自然的实现就是改
    EOL_RATIO，而 C12 会让整条管线在第 1 步就中止。所以 C12 已删除，
    W7 的扫描必须**显式传 tau 参数**，不许改常量。

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
    DATA_RAW,
    REGEN_MIN_JUMP_AH,
    RATED_CAPACITY_AH,
    is_at_eol,
)
from load_data import EXPECTED_COLUMNS, add_derived, load_raw, split_by_battery  # noqa: E402

# ---------------------------------------------------------------------------
# 期望值：全部来自对原始 CSV 的实测，不是猜的
# ---------------------------------------------------------------------------
EXP_RECORDS = 636
EXP_CYCLES = {"B0005": 168, "B0006": 168, "B0007": 168, "B0018": 132}
EXP_AMBIENT_TEMP = 24.0
EXP_OVERRATED_N = 7          # B0006 第 1–7 循环超过额定容量
EXP_OVERRATED_MAX = 2.0353
EXP_REGEN_EVENTS = {"B0005": 2, "B0006": 4, "B0007": 1, "B0018": 5}
EXP_REGEN_MAX_JUMP = 0.1519

EOL_REF_AH = 1.4             # 体检绑定的阈值，有意与 config 解耦（见文件头）
EXP_EOL_AT_REF = {"B0005": 125, "B0006": 109, "B0007": None, "B0018": 97}
EXP_B0007_MIN = 1.4005

EXPECTED_HEADER = ",".join(EXPECTED_COLUMNS)

# 容差：数据只记录到 4 位小数，任何真实改动都会差 >= 1e-4。
# 因此用 1e-9 + rtol=0 做"精确比较"是安全的，而且比 np.isclose 的默认
# rtol=1e-5 严格得多（默认 rtol 会让 atol 参数基本失效）。
TOL = 1e-9

_results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, bool(ok), detail))


def close(a: float, b: float) -> bool:
    """显式关掉 rtol，避免默认 rtol=1e-5 悄悄放大容差。"""
    return bool(np.isclose(a, b, rtol=0.0, atol=TOL))


def regen_events(sub: pd.DataFrame) -> tuple[list[int], list[float]]:
    """返回 (发生回升的循环号列表, 回升幅度列表)。"""
    cap = sub["capacity_Ah"].to_numpy()
    cyc = sub["cycle"].to_numpy()
    delta = np.diff(cap)
    idx = np.where(delta > REGEN_MIN_JUMP_AH)[0]
    return [int(cyc[i + 1]) for i in idx], [float(delta[i]) for i in idx]


def eol_cycle(sub: pd.DataFrame, tau: float) -> int | None:
    """容量首次到达 tau 的循环号；从未达到则返回 None（右删失）。

    判定口径由 `config.is_at_eol()` 统一提供，本函数不硬编码比较符——
    第一轮审查发现过「开关是死代码、调用处各自硬编码」的问题。
    """
    hit = sub[is_at_eol(sub["capacity_Ah"].to_numpy(float), tau)]
    return int(hit["cycle"].iloc[0]) if len(hit) else None


def run() -> None:
    raw = load_raw()
    df = add_derived(raw)
    by = split_by_battery(df)

    # ---- C1 记录数（列数交给 C2 对文件头断言，避免恒真）----
    check(
        f"C1 数据记录数 = {EXP_RECORDS}",
        len(raw) == EXP_RECORDS,
        f"实际 {len(raw)} 行",
    )

    # ---- C2 原始文件表头（同时验证 BOM 已剥掉）----
    header = DATA_RAW.read_text(encoding="utf-8-sig").splitlines()[0].strip()
    check(
        "C2 原始文件表头与数据说明一致（含 BOM 处理）",
        header == EXPECTED_HEADER,
        f"实际 '{header}'",
    )

    # ---- C3 缺失值 ----
    n_null = int(raw.isna().sum().sum())
    check("C3 无缺失值", n_null == 0, f"缺失单元格 {n_null} 个")

    # ---- C4 重复记录 ----
    n_dup = int(raw.duplicated(subset=["battery_id", "cycle"]).sum())
    check("C4 无重复 (电池, 循环) 记录", n_dup == 0, f"重复 {n_dup} 条")

    # ---- C5 温度零方差（最大的坑）----
    temps = sorted(raw["ambient_temp_C"].unique().tolist())
    temp_ok = len(temps) == 1 and close(temps[0], EXP_AMBIENT_TEMP)
    check(
        "C5 ambient_temp_C 是零方差列（唯一值仅 1 个）",
        temp_ok,
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

    # ---- C8 容量超额定（用 config 常量，不硬编码 2.0）----
    b6 = by["B0006"]
    over = b6[b6["capacity_Ah"] > RATED_CAPACITY_AH]
    check(
        f"C8 B0006 前 {EXP_OVERRATED_N} 循环容量超过额定 {RATED_CAPACITY_AH} Ah",
        len(over) == EXP_OVERRATED_N
        and over["cycle"].tolist() == list(range(1, EXP_OVERRATED_N + 1))
        and close(float(over["capacity_Ah"].max()), EXP_OVERRATED_MAX),
        f"{len(over)} 条，最高 {over['capacity_Ah'].max():.4f} Ah（物理上可疑，需在清洗章节说明）",
    )

    # ---- C9 容量再生事件（空列表也要能正常报 FAIL 而不是崩）----
    got_events = {b: len(regen_events(sub)[0]) for b, sub in by.items()}
    all_jumps = [j for sub in by.values() for j in regen_events(sub)[1]]
    max_jump = max(all_jumps, default=float("nan"))
    check(
        "C9 容量再生事件数 = 2/4/1/5，最大回升 0.1519 Ah",
        got_events == EXP_REGEN_EVENTS and close(max_jump, EXP_REGEN_MAX_JUMP),
        f"实际 {got_events}，最大回升 {max_jump:.4f} Ah（是物理现象，不是噪声）",
    )

    # ---- C10 EOL 与删失：断言"结论"，而不只是复读数值 ----
    got_eol = {b: eol_cycle(sub, EOL_REF_AH) for b, sub in by.items()}
    b7_min = float(by["B0007"]["capacity_Ah"].min())
    never_crossed = b7_min > EOL_REF_AH
    check(
        "C10 EOL(1.4 Ah) = 125/109/未到达/97，B0007 为右删失",
        got_eol == EXP_EOL_AT_REF
        and close(b7_min, EXP_B0007_MIN)
        and never_crossed,
        f"实际 {got_eol}；B0007 最低 {b7_min:.4f} Ah，距阈值差 {b7_min - EOL_REF_AH:.4f} Ah",
    )

    # ---- C11 切分不丢行（防 split_by_battery 静默丢电池）----
    total = sum(len(s) for s in by.values())
    check(
        "C11 按电池切分后总行数守恒",
        total == len(raw),
        f"切分后 {total} 行 vs 原始 {len(raw)} 行",
    )

    # C12 已删除：原「config 阈值必须等于 1.4」的断言会让 W7 的 τ 扫描
    # 在改 EOL_RATIO 时把整条管线掐死（数据健康信号 vs 调参信号混在一起）。


def report() -> int:
    if not _results:
        print("未运行 run()，没有任何断言结果")
        return 1

    print("=" * 78)
    print("数据体检报告")
    print("=" * 78)
    for name, ok, detail in _results:
        flag = "PASS" if ok else "FAIL"
        # 不做定宽补齐：中文是双宽字符，按 len() 补齐一定错位
        print(f"[{flag}] {name} | {detail}")
    n_fail = sum(1 for _, ok, _ in _results if not ok)
    print("-" * 78)
    print(f"共 {len(_results)} 条断言：{len(_results) - n_fail} 通过，{n_fail} 失败")
    print("=" * 78)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    run()
    sys.exit(report())
