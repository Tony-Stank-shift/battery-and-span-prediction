"""变异测试：证明 W7/W8 的断言真的能失败。

一个从不失败的断言等于没有断言。做法是手工注入已知错误，
看对应的断言是否**按预期炸掉**。每次注入都必须让管线以非零退出码结束。

它真的抓到过东西（不只是验证代码）：
最初我给 W8 写的变异是「把 trailing 滚动窗口换成居中窗口」，预期泄漏检测会炸——
结果**没炸**。顺着这条追下去才发现：错的是我的叙述，不是检测。
每块电池只算一个 t=K 的特征向量、输入先截断到 cycle ≤ K，
所以居中窗口在截断后的数据上根本越不过 K，「居中窗口就是泄漏」是我把
面板式特征的规则搬错了地方。变异已换成「先算后截断」（真泄漏），检测随即炸掉。

用法
----
    python scripts/mutation_test.py      # 5/5 通过才说明断言不是空的
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

    _orig_build = features.build_features  # W8 两个变异都以原函数为基底
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
    def leaky_build(df, K):
        import numpy as np

        out = _orig_build(df, K)
        for i, cell in enumerate(out["电池"]):
            sub = df[df["battery_id"] == cell].sort_values("cycle")
            # 用**整条轨迹**的尾部滚动残差（含 t>K 的行）覆盖掉因果版本
            y = sub["capacity_Ah"].reset_index(drop=True)
            resid = (y - y.rolling(11, min_periods=3).median()).to_numpy()
            out.loc[i, "resid_std_trailing"] = float(np.nanstd(resid[11:]))
        return out

    results.append(
        run_case(
            "W8-M1 特征改用整条轨迹算（先算后截断，真泄漏）",
            features,
            {"attr": "build_features", "new": leaky_build},
            "未来信息泄漏",
        )
    )

    # --- W8 变异 5：把整轨迹最小容量混进特征（老坑重演）---
    def leaky_build2(df, K):
        import numpy as np

        out = _orig_build(df, K)
        mins = df.groupby("battery_id")["capacity_Ah"].min()
        out["cap_min_full"] = out["电池"].map(mins).to_numpy(float)
        return out

    results.append(
        run_case(
            "W8-M2 特征里混入整轨迹最小容量",
            features,
            {"attr": "build_features", "new": leaky_build2},
            "未来信息泄漏",
        )
    )

    print()
    n_ok = sum(results)
    print(f"{n_ok}/{len(results)} 个变异被断言抓住")
    if n_ok != len(results):
        print("有断言抓不住对应错误 —— 必须先修断言，再加功能。")
        return 1
    print("全部变异都被抓住：这些断言不是空的。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
