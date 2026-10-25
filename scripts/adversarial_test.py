"""对抗输入测试：用病态数据喂核心函数，检查它们是**报错**还是**静默给错答案**。

为什么单独一个脚本
------------------
`mutation_test.py` 注入的是**代码错误**（改坏实现）；这个脚本注入的是**数据错误**
（NaN、空数组、常数、严格递增、单点）。两类问题不一样：

- 代码错误 → 断言该抓到；
- 病态数据 → 函数**要么给出可验证的正确答案，要么明确报错**。
  「静默算出一个数」是最坏结果，因为它会被下游当真的用。

这个脚本真的抓到过两个静默错误（都是在已提交的代码里）：

1. `prefix_min_mask` 遇到 NaN 后，`np.minimum.accumulate` 的最小值永远是 NaN，
   之后所有点都不算「新低」→ `eol_at_exact` **静默返回「删失」**。
   实测 cap = [1.8, nan, 1.7, 1.6, 1.5]、τ=1.6 的正确答案是第 4 循环，旧实现返回 None。
2. `features.build_features_one` 遇 NaN 会静默产出 NaN 特征，而 `leakage_test` 用
   `equal_nan=True` 比较，NaN == NaN 被判成「没变化」——泄漏检测也发现不了。

两处都已改成显式 `ValueError`。

用法
----
    python scripts/adversarial_test.py      # 全部通过才返回 0
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import features as F  # noqa: E402
import sensitivity as S  # noqa: E402

results: list[tuple[str, bool, str]] = []


def check(name: str, fn, expect) -> None:
    """跑 fn()，把返回值 / 异常与期望比对。"""
    try:
        got = fn()
    except Exception as exc:  # noqa: BLE001 —— 这里就是要区分异常类型
        got = f"{type(exc).__name__}"
    ok = got == expect if not callable(expect) else expect(got)
    results.append((name, ok, f"期望 {expect!r}，实际 {got!r}"))


def expect_raises(kind: str):
    return lambda got: got == kind


def main() -> int:
    # ------------------------------------------------------------------
    # A. 病态容量序列：正确答案是**可手算的**，逐一钉住
    # ------------------------------------------------------------------
    dec = np.array([2.0, 1.9, 1.8, 1.7, 1.6])          # 严格递减
    flat = np.array([1.5] * 5)                          # 常数
    inc = np.array([1.5, 1.6, 1.7, 1.8, 1.9])          # 严格递增
    dup = np.array([1.8, 1.7, 1.7, 1.7, 1.6])          # 带重复值
    one = np.array([1.5])                               # 单点

    check("严格递减 / τ=1.6 → 第 5 循环", lambda: S.eol_at_exact(dec, 1.6), 5)
    check("严格递减 / τ=1.5 → 删失", lambda: S.eol_at_exact(dec, 1.5), None)
    check("常数 / τ=1.5 → 第 1 循环（首点即到达）",
          lambda: S.eol_at_exact(flat, 1.5), 1)
    check("常数 / τ=1.4 → 删失", lambda: S.eol_at_exact(flat, 1.4), None)
    check("严格递增 / τ=1.6 → 第 1 循环（首点已 ≤ 1.6）",
          lambda: S.eol_at_exact(inc, 1.6), 1)
    check("严格递增 / τ=1.4 → 删失（首点 1.5 > 1.4）",
          lambda: S.eol_at_exact(inc, 1.4), None)
    check("带重复值 / τ=1.75 → 第 2 循环（第 2/3/4 都是 1.7，取首次）",
          lambda: S.eol_at_exact(dup, 1.75), 2)
    check("带重复值 / 前缀最小点数 = 3（重复值只留第一次）",
          lambda: int(S.prefix_min_mask(dup).sum()), 3)
    check("单点 / τ=1.5 → 第 1 循环", lambda: S.eol_at_exact(one, 1.5), 1)
    check("单点 / τ=1.4 → 删失", lambda: S.eol_at_exact(one, 1.4), None)

    # 两种实现在病态数据上也要一致（这是本项目最关键的一致性约束）
    for tag, arr in (("递减", dec), ("常数", flat), ("递增", inc), ("重复", dup)):
        for tau in (1.4, 1.5, 1.6, 1.75, 2.0):
            check(
                f"{tag} / τ={tau}：朴素与断点实现一致",
                lambda a=arr, t=tau: S.eol_at(a, t) == S.eol_at_exact(a, t),
                True,
            )

    # ------------------------------------------------------------------
    # B. 非有限输入：**必须报错**，不许静默给答案
    # ------------------------------------------------------------------
    nan_arr = np.array([1.8, np.nan, 1.7, 1.6, 1.5])
    inf_arr = np.array([1.8, np.inf, 1.7, 1.6, 1.5])
    all_nan = np.array([np.nan] * 5)

    check("含 NaN → prefix_min_mask 报 ValueError",
          lambda: S.prefix_min_mask(nan_arr), expect_raises("ValueError"))
    check("含 NaN → eol_at_exact 报 ValueError",
          lambda: S.eol_at_exact(nan_arr, 1.6), expect_raises("ValueError"))
    check("含 NaN → eol_at 报 ValueError",
          lambda: S.eol_at(nan_arr, 1.6), expect_raises("ValueError"))
    check("含 NaN → censored_below 报 ValueError",
          lambda: S.censored_below(nan_arr), expect_raises("ValueError"))
    check("含 inf → eol_at_exact 报 ValueError",
          lambda: S.eol_at_exact(inf_arr, 1.6), expect_raises("ValueError"))
    check("全 NaN → 报 ValueError",
          lambda: S.eol_at_exact(all_nan, 1.6), expect_raises("ValueError"))
    check("空数组 → 报 ValueError",
          lambda: S.eol_at_exact(np.array([]), 1.6), expect_raises("ValueError"))

    # ------------------------------------------------------------------
    # C. 特征侧的对抗输入
    # ------------------------------------------------------------------
    def make_sub(cap: np.ndarray, cell: str = "X") -> pd.DataFrame:
        return pd.DataFrame(
            {
                "battery_id": cell,
                "cycle": np.arange(1, len(cap) + 1),
                "capacity_Ah": cap,
            }
        )

    good = make_sub(np.linspace(1.9, 1.5, 50))
    check("正常输入（恰好 50 行）→ 返回 10 个特征",
          lambda: len(F.build_features_one(good, 50)), 10)
    check("含 NaN 的特征输入 → 报 ValueError",
          lambda: F.build_features_one(make_sub(np.r_[np.linspace(1.9, 1.5, 49), np.nan]), 50),
          expect_raises("ValueError"))
    check("只给 30 行却要 K=50 → 必须报错（否则算出来的是 K=30 的特征）",
          lambda: F.build_features_one(make_sub(np.linspace(1.9, 1.5, 30)), 50),
          expect_raises("ValueError"))
    check("给了 60 行却要 K=50 → 必须报错（不许悄悄截断）",
          lambda: F.build_features_one(make_sub(np.linspace(1.9, 1.5, 60)), 50),
          expect_raises("ValueError"))

    # ------------------------------------------------------------------
    # 汇总
    # ------------------------------------------------------------------
    n_ok = sum(1 for _, ok, _ in results if ok)
    for name, ok, detail in results:
        if not ok:
            print(f"[FAIL] {name}\n        {detail}")
    print()
    print(f"{n_ok}/{len(results)} 条对抗输入检查通过")
    if n_ok != len(results):
        print("有病态输入被静默接受了 —— 先修防护，再加功能。")
        return 1
    print("所有病态输入要么被正确回答、要么被明确拒绝；没有静默给错答案。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
