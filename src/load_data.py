"""数据读入与基础派生列。

原则：本模块只做"读进来 + 加列"，不做任何清洗判断。
清洗决策集中在 preprocess.py（W3），这样"数据长什么样"和"我们怎么处理它"是分开的。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import BATTERIES, DATA_RAW, RATED_CAPACITY_AH  # noqa: E402

EXPECTED_COLUMNS = ["battery_id", "cycle", "capacity_Ah", "ambient_temp_C"]


def load_raw(path: Path | str | None = None) -> pd.DataFrame:
    """读入原始 CSV，做类型规整，按 (电池, 循环) 排序。

    Returns
    -------
    DataFrame
        列：battery_id(str), cycle(int), capacity_Ah(float), ambient_temp_C(float)
    """
    path = Path(path) if path is not None else DATA_RAW
    if not path.exists():
        raise FileNotFoundError(f"找不到数据文件：{path}")

    df = pd.read_csv(path)

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"数据缺少列：{missing}；实际列为 {list(df.columns)}")

    df = df[EXPECTED_COLUMNS].copy()
    df["battery_id"] = df["battery_id"].astype(str).str.strip()
    df["cycle"] = df["cycle"].astype(int)
    df["capacity_Ah"] = df["capacity_Ah"].astype(float)
    df["ambient_temp_C"] = df["ambient_temp_C"].astype(float)

    return df.sort_values(["battery_id", "cycle"]).reset_index(drop=True)


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """加入派生列。

    新增列：
    - soh          : State of Health = 当前容量 / 额定容量（0–1 之间的相对健康度）
    - soh_vs_first : 相对于该电池首个循环容量的健康度（消除个体初始容量差异）
    - is_cell_first_cycle : 是否是该电池的第 1 个循环
    """
    out = df.copy()
    out["soh"] = out["capacity_Ah"] / RATED_CAPACITY_AH

    first_cap = out.groupby("battery_id")["capacity_Ah"].transform("first")
    out["soh_vs_first"] = out["capacity_Ah"] / first_cap

    out["is_cell_first_cycle"] = out["cycle"] == 1
    return out


def split_by_battery(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """按电池个体切分，返回有序字典（顺序固定为 BATTERIES）。"""
    return {
        b: df[df["battery_id"] == b].sort_values("cycle").reset_index(drop=True)
        for b in BATTERIES
        if (df["battery_id"] == b).any()
    }


if __name__ == "__main__":
    raw = load_raw()
    print(f"读入 {len(raw)} 行 × {raw.shape[1]} 列")
    print(raw.head(3).to_string(index=False))
    print()
    print("按电池分组：")
    for b, sub in split_by_battery(raw).items():
        print(f"  {b}: {len(sub)} 个循环（{sub['cycle'].min()}–{sub['cycle'].max()}）")
