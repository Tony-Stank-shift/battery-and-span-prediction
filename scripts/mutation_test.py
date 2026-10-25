"""变异测试：证明 W7/W8 的断言真的能失败。

一个从不失败的断言等于没有断言。做法是手工注入已知错误，
看对应的断言是否**按预期炸掉**。每次注入都必须让管线以非零退出码结束。

当前 **11 条**：W7 六条 + W8 四条 + 一条回归用例（W8-R1）。

它真的抓到过东西（不只是验证代码）：
1. 最初我给 W8 写的变异是「把 trailing 滚动窗口换成居中窗口」，预期泄漏检测会炸——
   **没炸**。追下去才发现：错的是我的叙述，不是检测（每块电池只算一个 t=K 的特征、
   输入先截断到 cycle ≤ K，居中窗口越不过 K）。该结论现在固化成 W8-R1：
   判据不是「炸」，而是「**不炸 且 数值确实不同**」——它本来就是口径问题，不是泄漏。
2. 第四轮独立审查用「往特征里塞全表行数」这种**元数据**变异，绕过了当时只扰动能量的
   泄漏检测，并把 168 写进了 md。补上删行臂后，W8-M3 现在会被删行臂抓到。

用法
----
    python scripts/mutation_test.py      # 11/11 通过才说明这些断言不是空的
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def run_case(name: str, target, mutate, expect_in: str) -> bool:
    """注入变异后跑 target.main()，要求它抛 AssertionError 且信息含 expect_in。"""
    saved = getattr(target, mutate["attr"])
    try:
        setattr(target, mutate["attr"], mutate["new"])
        buf = io.StringIO()
        with redirect_stdout(buf):
            target.main()
    except AssertionError as exc:
        msg = str(exc)
        ok = expect_in in msg
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        print(f"        期望信息含 {expect_in!r}")
        print(f"        实际信息：{msg.splitlines()[0][:110]}")
        return ok
    except Exception as exc:  # 不是断言失败，说明变异打在别处
        print(f"[FAIL] {name}：抛出的是 {type(exc).__name__} 而不是 AssertionError")
        print(f"        {exc}")
        return False
    else:
        print(f"[FAIL] {name}：注入错误后竟然跑通了 —— 这条断言是空的")
        return False
    finally:
        setattr(target, mutate["attr"], saved)


def main() -> int:
    import sensitivity
    import features

    _orig_build = features.build_features  # W8 三个变异都以原函数为基底
    _orig_cap_series = sensitivity.cap_series  # W7-M4 也要
    results = []

    # --- W7 变异 1：EOL 断点实现退回「永远返回首个前缀最小点」---
    def broken_exact(cap, tau):
        import numpy as np

        idx = np.flatnonzero(sensitivity.prefix_min_mask(cap))
        hit = sensitivity.is_at_eol(cap[idx], tau)
        return int(idx[0]) + 1 if hit.any() else None

    results.append(
        run_case(
            "W7-M1 断点实现退化成 idx[0]（首版真实 bug）",
            sensitivity,
            {"attr": "eol_at_exact", "new": broken_exact},
            "朴素=",
        )
    )

    # --- W7 变异 2：扫描不再显式传 tau，偷偷用 config 常量（D17 最怕的事）---
    # 注意：这条断言原本写的是「默认 τ 与显式 MAIN_TAU 一致」，
    # 但 MAIN_TAU 就是 EOL_THRESHOLD_AH，两边表达式完全相同 → 恒真，
    # 第五轮审查 F14 抓到了。断言已换成「临时改 config 常量，扫描结果必须不变」，
    # 所以变异也要换成对应的错误：忽略传入的 tau。
    def ignores_tau(cap, tau):
        import numpy as np

        idx = np.flatnonzero(sensitivity.prefix_min_mask(cap))
        hit = sensitivity.is_at_eol(cap[idx])  # ← 故意不传 tau，用 config 默认值
        return int(idx[hit][0]) + 1 if hit.any() else None

    results.append(
        run_case(
            "W7-M2 断点实现忽略传入的 τ（改用 config 常量）",
            sensitivity,
            {"attr": "eol_at_exact", "new": ignores_tau},
            "τ 没有被显式传入",
        )
    )

    # --- W7 变异 3：删失边界错位（用 < 代替 ≤ 的效果）---
    def shifted_bound(cap):
        import numpy as np

        return float(np.nextafter(cap.min(), np.inf))

    results.append(
        run_case(
            "W7-M3 删失边界挪高一格（边界判定口径被改）",
            sensitivity,
            {"attr": "censored_below", "new": shifted_bound},
            "仍判为到达",
        )
    )

    # --- W8 变异 4：去掉「先截断再算」这个顺序（真·泄漏）---
    # 注意：把 trailing 换成 center=True **不是**泄漏（截断后居中窗口最多用到第 K 行）。
    # 真正的泄漏是「在完整数据上算特征、再去取结果」，所以这里直接打掉截断。
    def leaky_build(df, K, extra=None):
        import numpy as np

        out = _orig_build(df, K, extra=extra)
        for i, cell in enumerate(out["电池"]):
            sub = df[df["battery_id"] == cell].sort_values("cycle")
            # 用**整条轨迹**的尾部滚动残差（含 t>K 的行）覆盖掉因果版本
            y = sub["capacity_Ah"].reset_index(drop=True)
            resid = (y - y.rolling(11, min_periods=3).median()).to_numpy()
            out.loc[i, "resid_std_trailing"] = float(np.nanstd(resid[11:]))
        return out

    results.append(
        run_case(
            "W8-M1 残差特征改用整条轨迹算（含 t>K 的行）",
            features,
            {"attr": "build_features", "new": leaky_build},
            "容量值",
        )
    )

    # --- W8 变异 5：把整轨迹最小容量混进特征（老坑重演）---
    def leaky_build2(df, K, extra=None):
        import numpy as np

        out = _orig_build(df, K, extra=extra)
        mins = df.groupby("battery_id")["capacity_Ah"].min()
        out["cap_min_full"] = out["电池"].map(mins).to_numpy(float)
        return out

    results.append(
        run_case(
            "W8-M2 特征里混入整轨迹最小容量",
            features,
            {"attr": "build_features", "new": leaky_build2},
            "容量值",
        )
    )

    # --- W7 变异 4：容量被整体挪动 → 断言 1（跨模块口径核对）必须发现 ---
    # 守的是「W7 算出的 EOL 与 W3 个体档案一致」这条跨模块约束。
    def shifted_series(df):
        out = {}
        for cell, (cyc, cap) in _orig_cap_series(df).items():
            out[cell] = (cyc, cap - 0.05)  # 容量整体下移，EOL 会提前
        return out

    results.append(
        run_case(
            "W7-M4 容量序列整体下移 0.05 Ah（跨模块 EOL 不再一致）",
            sensitivity,
            {"attr": "cap_series", "new": shifted_series},
            "与 W3 档案不一致",
        )
    )

    # --- W7 变异 5：断点表里的循环号错一格 → 断言 5 必须发现 ---
    _orig_breakpoints = sensitivity.breakpoints

    def off_by_one_breakpoints(cap):
        w, c = _orig_breakpoints(cap)
        return w, (c + 1).astype(int)

    results.append(
        run_case(
            "W7-M5 断点表循环号整体 +1（断点表与实测不符）",
            sensitivity,
            {"attr": "breakpoints", "new": off_by_one_breakpoints},
            "处朴素实现不符",
        )
    )

    # --- W7 变异 6：某处偷偷改了 config 常量 → 断言 8 的 tripwire 必须响 ---
    _orig_range_table = sensitivity.range_table

    def mutating_range_table(series):
        import config as cfg

        cfg.EOL_RATIO = 0.5  # 就是 D17 明令禁止的那种写法
        return _orig_range_table(series)

    results.append(
        run_case(
            "W7-M6 扫描过程顺手改 config.EOL_RATIO（D17 明令禁止）",
            sensitivity,
            {"attr": "range_table", "new": mutating_range_table},
            "扫描过程改动了",
        )
    )

    # --- W8 变异 3：特征里混进「未来行数」这种元数据 ---
    # 这条最关键：它对**容量扰动完全无感**，只有删行臂能抓到。
    # 第四轮审查正是用这个形状的变异绕过了初版检测并把 168 写进了 md。
    def metadata_leak_build(df, K, extra=None):
        out = _orig_build(df, K, extra=extra)
        out["obs_len"] = df.groupby("battery_id").size().reindex(out["电池"]).to_numpy(float)
        return out

    results.append(
        run_case(
            "W8-M3 特征里混进「全表行数」元数据（只有删行臂能抓）",
            features,
            {"attr": "build_features", "new": metadata_leak_build},
            "行/元数据",
        )
    )

    # --- W8 变异 4：特征里混进被禁列名（cycle）---
    def forbidden_col_build(df, K, extra=None):
        out = _orig_build(df, K, extra=extra)
        out["cycle"] = df.groupby("battery_id")["cycle"].max().reindex(out["电池"]).to_numpy()
        return out

    results.append(
        run_case(
            "W8-M4 特征里混进 FEATURE_EXCLUDE 里的 cycle",
            features,
            {"attr": "build_features", "new": forbidden_col_build},
            "行/元数据",
        )
    )

    # --- W8 回归用例：center=True 在「先截断」下**不**泄漏（应通过泄漏检测，但值确实变了）---
    # 审查 F9 指出历史叙述里这条只有注释、没有可复现的变异。这里固化成用例：
    # 判据不是「炸掉」，而是「不炸 且 数值不同」——正好说明它是口径问题、不是泄漏问题。
    def centered_resid(sub, window=features.TREND_WINDOW):
        import numpy as np
        import pandas as pd

        y = pd.Series(sub["capacity_Ah"].to_numpy(float))
        resid = (y - y.rolling(window, min_periods=3, center=True).median()).to_numpy()
        cyc = sub["cycle"].to_numpy(int)
        core = resid[cyc > window]
        return float(np.nanstd(core))

    saved = features._trailing_residual_std
    try:
        features._trailing_residual_std = centered_resid
        try:
            features.leakage_test(features.load_raw().pipe(features.mark_rows), features.K_MAIN)
            passed = True
        except AssertionError:
            passed = False
        import numpy as np
        df_ = features.load_raw().pipe(features.mark_rows)
        head = features.split_by_battery(df_)["B0005"]
        head = head[head["cycle"] <= features.K_MAIN]
        v_center = centered_resid(head)
        v_trail = saved(head)
        differ = not np.isclose(v_center, v_trail, rtol=0, atol=0)
    finally:
        features._trailing_residual_std = saved
    results.append(bool(passed and differ))
    print(f"[{'PASS' if passed and differ else 'FAIL'}] "
          "W8-R1 center=True 不泄漏、但值确实不同（口径问题，非泄漏）")
    print(f"        泄漏检测通过={passed}，数值不同={differ}"
          f"（居中 {v_center:.8f} / 尾部 {v_trail:.8f}）")

    print()
    n_ok = sum(results)
    print(f"{n_ok}/{len(results)} 个变异被断言抓住")
    if n_ok != len(results):
        print("有断言抓不住对应错误 —— 必须先修断言，再加功能。")
        return 1
    print("全部变异都被抓住：这些断言不是空的。")
    print()
    print("⚠️ 未被变异测试覆盖的断言（不是漏了，是覆盖不了）：")
    print("  · W7 断言 4「N_EOL 随 τ 单调不增」：这是 EOL 定义本身保证的恒等式——")
    print("    τ 变大时集合 {c: cap(c) ≤ τ} 只会变大，最小值只会更小。")
    print("    任何数据、任何实现都违反不了，所以它**没有验证力**，只是回归护栏。")
    print("  · W7 断言 7「排序集合只有 1 个元素」：守的是**数据性质**（谁先坏），")
    print("    不是实现；要让它失败必须伪造一份不同的数据，那测的是数据不是代码。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
