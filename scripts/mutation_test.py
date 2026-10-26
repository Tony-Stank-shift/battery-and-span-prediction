"""变异测试：证明 W7/W8/W10 的断言真的能失败。

一个从不失败的断言等于没有断言。做法是手工注入已知错误，
看对应的断言是否**按预期炸掉**。每次注入都必须让管线以非零退出码结束。

当前 **38 条**：W7 六条 + W8 四条 + W9 一条 + W10 四条 + W11 两条 + W12 两条 +
W13 五条 + W15 四条 + W16 五条 + W17 两条 + W18 三条 + 一条回归用例（W8-R1）。

（跑得比早期慢很多：W16 的变异要真跑整条 t* 扫描，单条约 1 分钟。）

它真的抓到过东西（不只是验证代码）：
1. 最初我给 W8 写的变异是「把 trailing 滚动窗口换成居中窗口」，预期泄漏检测会炸——
   **没炸**。追下去才发现：错的是我的叙述，不是检测（每块电池只算一个 t=K 的特征、
   输入先截断到 cycle ≤ K，居中窗口越不过 K）。该结论现在固化成 W8-R1：
   判据不是「炸」，而是「**不炸 且 数值确实不同**」——它本来就是口径问题，不是泄漏。
2. 第四轮独立审查用「往特征里塞全表行数」这种**元数据**变异，绕过了当时只扰动能量的
   泄漏检测，并把 168 写进了 md。补上删行臂后，W8-M3 现在会被删行臂抓到。

用法
----
    python scripts/mutation_test.py      # 33/33 通过才说明这些断言不是空的
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
        lines = msg.splitlines()
        print(f"        实际信息：{lines[0][:110] if lines else '(空消息——断言没写说明)'}")
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
    import evaluate
    import compare
    import uncertainty
    import warning
    import amplification
    import calibration
    import strategy
    import models_ml
    import models_parametric
    import models_gpr

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

    # --- W10 变异 1：偷偷给 M3 加第二自变量（cycle²）---
    # 这会把「树没有趋势概念」的干净反例弄脏，而且断言的守护点必须在**调用方**：
    # fit_rf 自己内部的断言救不了「fit_rf 被整个换掉」。
    _orig_fit_rf = models_ml.fit_rf

    def two_feature_fit(t, y, **kw):
        import numpy as np
        from sklearn.ensemble import RandomForestRegressor

        t = np.asarray(t, float)
        est = RandomForestRegressor(
            n_estimators=kw.get("n_estimators", models_ml.N_TREES),
            max_depth=kw.get("max_depth"),
            min_samples_leaf=kw.get("min_samples_leaf", 1),
            bootstrap=kw.get("bootstrap", True),
            random_state=kw.get("seed", 42),
            n_jobs=1,
        )
        est.fit(np.column_stack([t, t**2]), np.asarray(y, float))
        return {
            "name": "M3 随机森林", "estimator": est, "features": ["cycle", "cycle^2"],
            "t_min": float(t.min()), "t_max": float(t.max()),
            "n_estimators": int(kw.get("n_estimators", models_ml.N_TREES)),
            "max_depth": kw.get("max_depth"), "bootstrap": True, "n_train": len(t),
        }

    results.append(
        run_case(
            "W10-M1 M3 偷偷多吃一个自变量 cycle²（反例不再干净）",
            models_ml,
            {"attr": "fit_rf", "new": two_feature_fit},
            "只许吃",
        )
    )

    # --- W10 变异 2：给外推段补一条线性趋势（"手动修一下"这种最危险的做法）---
    _orig_predict = models_ml.predict_rf

    def trended_predict(model, t):
        import numpy as np

        t = np.asarray(t, float)
        base = _orig_predict(model, t)
        return base - 0.001 * np.maximum(0.0, t - model["t_max"])

    results.append(
        run_case(
            "W10-M2 外推段被偷偷补一条线性趋势（毁掉「分段常数」这个事实）",
            models_ml,
            {"attr": "predict_rf", "new": trended_predict},
            "不是严格常数",
        )
    )

    # --- W10 变异 3：EOL 落在训练区间内，却不标「查表」---
    def silent_lookup_eol(model, n_obs, tau=None):
        import numpy as np

        from config import EOL_SEARCH_MULTIPLE, is_at_eol

        grid = np.arange(1, int(np.ceil(n_obs * EOL_SEARCH_MULTIPLE)) + 1)
        hit = np.asarray(is_at_eol(models_ml.predict_rf(model, grid), tau))
        idx = np.flatnonzero(hit)
        return (int(grid[idx[0]]), "") if len(idx) else (None, "")

    results.append(
        run_case(
            "W10-M3 训练区间内的「EOL」不标查表（把查表当预测）",
            models_ml,
            {"attr": "eol_rf", "new": silent_lookup_eol},
            "查表",
        )
    )

    # --- W10 变异 4：强制浅树，让「长到底 = 最后一个观测值」不成立 ---
    def shallow_fit(t, y, **kw):
        kw["max_depth"] = 3
        return _orig_fit_rf(t, y, **kw)

    results.append(
        run_case(
            "W10-M4 强制 max_depth=3（外推值不再等于最后一次观测）",
            models_ml,
            {"attr": "fit_rf", "new": shallow_fit},
            "最右叶子",
        )
    )


    # --- W19 变异 1：SUM 判据偷懒用 RSS 的半宽（最坏情况退化成乐观估计）---
    _orig_rep = strategy.replacement_table

    def lazy_sum(budget):
        out = _orig_rep(budget)
        mask = out["判据"] == "SUM 下界"
        # 把 SUM 的替换时刻改成和 RSS 一样
        for cell, g in budget.set_index("电池").iterrows():
            out.loc[mask & (out["电池"] == cell), "替换时刻"] = (
                float(g["点估计"]) - float(g["rss"])
            )
        return out

    results.append(
        run_case(
            "W19-M1「最坏情况」判据偷懒用 RSS 的半宽（SUM 判决不再保守）",
            strategy,
            {"attr": "replacement_table", "new": lazy_sum},
            "严格递减",
        )
    )

    # --- W19 变异 2：稳定时刻没算出来就填观测长度（默认值冒充结论）---
    _orig_acq = strategy.acquisition_table

    def fake_stable(budget):
        out = _orig_acq(budget)
        out["稳定时刻(95%)"] = out["观测长度"]
        return out

    results.append(
        run_case(
            "W19-M2 稳定时刻用观测长度冒充（没算出来也没人发现）",
            strategy,
            {"attr": "acquisition_table", "new": fake_stable},
            "等于观测长度",
        )
    )

    # --- W18 变异 1：PICP 反号（模拟区间上下界写反的后果）---
    _orig_picp = calibration.picp_table

    def swapped_intervals(points, sd_key, label):
        out = _orig_picp(points, sd_key, label)
        out["PICP"] = 1.0 - out["PICP"]
        return out

    results.append(
        run_case(
            "W18-M1 PICP 反号（模拟区间上下界互换）",
            calibration,
            {"attr": "picp_table", "new": swapped_intervals},
            "的 PICP 不随名义覆盖率单调",
        )
    )

    # --- W18 变异 2：把区间标准差放大 10 倍（覆盖冲满）---
    _orig_run_cell2 = calibration.run_cell

    def wide_intervals(cell, sub, m):
        d = _orig_run_cell2(cell, sub, m)
        d["sd_pred"] = d["sd_pred"] * 10.0
        return d

    results.append(
        run_case(
            "W18-M2 把区间标准差放大 10 倍（覆盖冲满）",
            calibration,
            {"attr": "run_cell", "new": wide_intervals},
            "区间实际覆盖",
        )
    )

    # --- W18 变异 3：诊断放大系数算成 0.5（「要多宽才够」说不通）---
    results.append(
        run_case(
            "W18-M3 诊断放大系数被算成 0.5",
            calibration,
            {"attr": "needed_scale", "new": lambda pooled, target=90.0: 0.5},
            "needed_scale 的算法错了",
        )
    )

    # --- W17 变异 1：近场误差被放大 30 倍（「越往外越不准」被抹平）---
    _orig_run_cell = amplification.run_cell

    def flat_growth(cell, sub, eol_true, m):
        rec = _orig_run_cell(cell, sub, eol_true, m)
        rec["h1_误差(Ah)"] *= 30.0
        e_star = rec.get("e(h*)_误差(Ah)", float("nan"))
        if rec["h1_误差(Ah)"] > 0 and e_star == e_star:  # NaN != NaN
            rec["增长倍数 e(h*)/e(1)"] = e_star / rec["h1_误差(Ah)"]
        return rec

    results.append(
        run_case(
            "W17-M1 近场误差被放大 30 倍（增长倍数掉到 1 以下）",
            amplification,
            {"attr": "run_cell", "new": flat_growth},
            "没有变大",
        )
    )

    # --- W17 变异 2：理论放大算错（恒等式必须当场炸）---
    _orig_theory = amplification.theory_amplification

    results.append(
        run_case(
            "W17-M2 理论放大公式漏掉额定容量（恒等式被破坏）",
            amplification,
            {"attr": "theory_amplification", "new": lambda slope, eol: 0.5 / (abs(slope) * eol)},
            "这是恒等式",
        )
    )

    # --- W16 变异 1：把区间上下界写反（置信度越高反而越窄）---
    _orig_scan = warning.scan_cell

    def swapped_cell(df, cell, seeds=warning.SEEDS, step=warning.GRID_STEP):
        out = _orig_scan(df, cell, seeds=seeds, step=step)
        for p in warning.LEVELS:
            for s in seeds:
                lo, hi = out[f"L{p}_s{s}"].copy(), out[f"U{p}_s{s}"].copy()
                out[f"L{p}_s{s}"], out[f"U{p}_s{s}"] = hi, lo
        return out

    results.append(
        run_case(
            "W16-M1 置信区间的上下界写反",
            warning,
            {"attr": "scan_cell", "new": swapped_cell},
            "有电池在网格内始终不稳定",
        )
    )

    # --- W16 变异 1b：四档置信度的顺序换反（单调性断言必须响，R9-11）---
    _orig_build = warning.build_tstar

    def swapped_levels(scans, truth, step5):
        t = _orig_build(scans, truth, step5)
        flipped = list(reversed(warning.LEVELS))
        t["置信度"] = t["置信度"].map(dict(zip(warning.LEVELS, flipped)))
        return t

    results.append(
        run_case(
            "W16-M1b 四档置信度的顺序被换反（t* 单调性必须被抓）",
            warning,
            {"attr": "build_tstar", "new": swapped_levels},
            "随置信度不单调",
        )
    )

    # --- W16 变异 2：命中时刻往后推 100 循环（预警晚于失效）---
    _orig_hit = warning.first_hit

    def late_hit(tbl, p, seed, truth):
        v = _orig_hit(tbl, p, seed, truth)
        return (None, v[1], v[2]) if v[0] is None else (int(v[0]) + 100, v[1], v[2])

    results.append(
        run_case(
            "W16-M2 命中时刻被推后 100 循环（预警晚于真实失效）",
            warning,
            {"attr": "first_hit", "new": late_hit},
            "块电池在失效前就命中",
        )
    )

    # --- W16 变异 3：命中判据永远返回 None（检查会变空）---
    results.append(
        run_case(
            "W16-M3 命中时刻永远算不出来（断言形同虚设）",
            warning,
            {"attr": "first_hit", "new": lambda tbl, p, seed, truth: (None, float("nan"), 0)},
            "这条断言形同虚设",
        )
    )

    # --- W15 变异 4：把模型形式层整层从合并里删掉（R9-10：原来的断言是恒真式）---
    _orig_combine2 = uncertainty.combine

    def drop_model_layer(tbl):
        out = _orig_combine2(tbl.copy())
        d, s = out["定义层半宽"], out["数据层半宽"]
        out["rss"] = (d**2 + s**2) ** 0.5
        out["sum"] = d + s
        return out

    results.append(
        run_case(
            "W15-M4 合并时把模型形式层整层漏掉",
            uncertainty,
            {"attr": "combine", "new": drop_model_layer},
            "这一层没进入合并结果",
        )
    )

    # --- W15 变异 1：RSS 换成算术平均（合并比单层还窄）---
    _orig_combine = uncertainty.combine

    def mean_combine(tbl):
        out = _orig_combine(tbl)
        out["rss"] = (out["定义层半宽"] + out["模型层半极差"] + out["数据层半宽"]) / 3.0
        return out

    results.append(
        run_case(
            "W15-M1 三层用算术平均合并（合并比单层还窄）",
            uncertainty,
            {"attr": "combine", "new": mean_combine},
            "这一层没进入合并结果",
        )
    )

    # --- W15 变异 2：SUM 换成取最小值（最坏情况比 RSS 还窄）---
    def min_combine(tbl):
        out = _orig_combine(tbl)
        out["sum"] = out[
            ["定义层半宽", "模型层半极差", "数据层半宽"]
        ].min(axis=1)
        return out

    results.append(
        run_case(
            "W15-M2 「最坏情况」用取最小代替相加",
            uncertainty,
            {"attr": "combine", "new": min_combine},
            "最坏情况合并比 RSS 还窄",
        )
    )

    # --- W15 变异 3：定义层算错（窗口取反），与 W7 发表的数字对不上 ---
    _orig_def = uncertainty.layer_definition

    def wrong_window(series):
        out = _orig_def(series)
        return {k: (v[0] + 1.0, v[1]) for k, v in out.items()}

    results.append(
        run_case(
            "W15-M3 定义层极差算错（与 W7 发表的数对不上）",
            uncertainty,
            {"attr": "layer_definition", "new": wrong_window},
            "与 W7 发表的",
        )
    )

    # --- W13 变异 1：某个模型的表少一块电池（"只比能比的"这种偷懒）---
    _orig_collect = compare.collect

    def missing_cell_collect(df, prof, m=models_ml.SPLIT_M):
        out = _orig_collect(df, prof, m)
        t = out["table"]
        keep = t[~((t["模型"] == "M3 随机森林") & (t["电池"] == "B0007"))]
        return {**out, "table": keep}

    results.append(
        run_case(
            "W13-M1 某个模型少一块电池（在比不同的电池）",
            compare,
            {"attr": "collect", "new": missing_cell_collect},
            "有模型的行数不等于电池数",
        )
    )

    # --- W13 变异 1b：四模型**一起**少一块电池（R8-03 抓到的真漏洞）---
    def drop_cell_everywhere(df, prof, m=models_ml.SPLIT_M):
        out = _orig_collect(df, prof, m)
        t = out["table"]
        # 只删表里的行，**不改 cells**：期望集合由 main() 从数据现取，
        # 所以这个「对称的偷懒」必须被抓住（R8-03）
        return {**out, "table": t[t["电池"] != "B0007"]}

    results.append(
        run_case(
            "W13-M1b 四模型一起少一块电池（最自然的偷懒，对称所以骗过对称性检查）",
            compare,
            {"attr": "collect", "new": drop_cell_everywhere},
            "电池集合与数据不一致",
        )
    )

    # --- W13 变异 2：把「—」当 0 塞进 EOL 误差（会让 M3 显示成完美）---
    _orig_states = compare.split_eol_states

    def zero_filled_states(tbl, expected_cells=None):
        import pandas as _pd

        out = _orig_states(tbl, expected_cells).copy()
        # 只把**非删失**行里的「—」填 0：这样第一道「删失不许有误差」不会先响，
        # 命中的就是「可比数」那条断言（两个变异各打各的靶）
        mask = (out["EOL误差"].astype(str) == "—") & (~out["真值删失"])
        out.loc[mask, "EOL误差"] = 0
        out["EOL误差数值"] = _pd.to_numeric(out["EOL误差"], errors="coerce")
        return out

    results.append(
        run_case(
            "W13-M2 把 EOL 的「—」当成 0（M3 会显得最准）",
            compare,
            {"attr": "split_eol_states", "new": zero_filled_states},
            "模型给不出 EOL，却带着一个误差数字",
        )
    )

    # --- W13 变异 2b：删失只认字符串，不认 NaN（R8-05 抓到的真漏洞）---
    def string_only_censor(tbl, expected_cells=None):
        import pandas as _pd

        out = _orig_states(tbl, expected_cells).copy()
        out["真值删失"] = out["EOL真值"].astype(str) == "删失"
        out.loc[out["真值删失"], "EOL误差"] = 3.0
        out["EOL误差数值"] = _pd.to_numeric(out["EOL误差"], errors="coerce")
        return out

    results.append(
        run_case(
            "W13-M2b 删失只按字符串判断（真值编码成 NaN 时会静默变成可比）",
            compare,
            {"attr": "split_eol_states", "new": string_only_censor},
            "有删失电池被算进了 EOL 误差",
        )
    )

    # --- W13 变异 3：合并时覆盖模型身份列（R7-6 那个 bug 的回归）---
    _orig_to_common = compare.to_common_schema

    def collapsed_schema(tbl, model_name=None):
        return _orig_to_common(tbl).assign(模型="M_all")

    results.append(
        run_case(
            "W13-M3 合并时把四个模型压成一个标签",
            compare,
            {"attr": "to_common_schema", "new": collapsed_schema},
            "模型集合不对",
        )
    )

    # --- W12 变异 1：LOBOCV 的折里混进被留出的个体（数据泄漏的典型形态）---
    def overlapping_folds(labels):
        ids = list(labels.index)
        return [(ids, hold) for hold in ids]

    results.append(
        run_case(
            "W12-M1 LOBOCV 训练集里混进被留出的个体",
            evaluate,
            {"attr": "lobocv_folds", "new": overlapping_folds},
            "折构造不对",
        )
    )

    # --- W12 变异 2：统一表静默补 NaN，而不是报错 ---
    def filling_schema(tbl, model_name=None):
        import numpy as np

        out = tbl.copy()
        for c in evaluate.COMMON_COLUMNS:
            if c not in out.columns:
                out[c] = np.nan
        if model_name is not None:
            out["模型"] = model_name
        return out

    results.append(
        run_case(
            "W12-M2 统一表缺列时静默补 NaN（掩盖列名分叉）",
            evaluate,
            {"attr": "to_common_schema", "new": filling_schema},
            "没有报错",
        )
    )

    # --- W11 变异 1：把「趋势参数方差」从带宽里去掉，只留残差方差 ---
    # 这是 M4 存在理由的反面：去掉以后带宽不再随距离张开，
    # structure_checks 的第 5 条断言（远处趋势方差必须占大头）会当场失败。
    _orig_predict_gpr = models_gpr.predict_gpr
    _orig_fit_gpr = models_gpr.fit_gpr

    def no_trend_var(model, t, with_trend_var=True):
        return _orig_predict_gpr(model, t, with_trend_var=False)

    results.append(
        run_case(
            "W11-M1 带宽里去掉趋势参数方差（只留残差噪声）",
            models_gpr,
            {"attr": "predict_gpr", "new": no_trend_var},
            "带宽主要由趋势项提供",
        )
    )

    # --- W11 变异 2：把噪声下限去掉，让 GP 逐点插值（第一版真踩过的坑）---
    def interpolating_fit(t, y, seed=models_gpr.GP_SEED, n_restarts=2):
        import numpy as np
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel

        mod = _orig_fit_gpr(t, y, seed=seed, n_restarts=n_restarts)
        resid = np.asarray(y, float) - models_gpr.trend_line(mod["trend"], t)
        gp = GaussianProcessRegressor(
            kernel=ConstantKernel(1.0, (1e-3, 1e3)) * RBF(10.0, (1e-1, 1e4)),
            alpha=1e-12, normalize_y=False, random_state=seed,
        )
        gp.fit(np.asarray(t, float).reshape(-1, 1), resid)
        return {**mod, "gp": gp, "length_scale": float(gp.kernel_.k2.length_scale)}

    results.append(
        run_case(
            "W11-M2 去掉噪声下限，GP 变成逐点插值",
            models_gpr,
            {"attr": "fit_gpr", "new": interpolating_fit},
            "逐点插值",
        )
    )

    # --- W9 变异 5：把「外推单调」的判据区间改回 [1, 5N]（第六轮审查抓到的真 bug）---
    # 老口径把训练段内那点上翘也算进去，于是 B0005/B0007 被判成"外推不单调"→ 不给寿命。
    # 现在 render_md 里有一条断言「四块电池都该有寿命」，专门守这件事。
    _orig_fit_de = models_parametric.fit_double_exp

    def old_window_fit(t, y, p0=None):
        import numpy as np

        mod = _orig_fit_de(t, y, p0=p0)
        if not mod["converged"]:
            return mod
        p = mod["params"]
        probe = np.linspace(1.0, float(len(t) * 5), 2000)
        curve = models_parametric._double_exp(probe, p["a"], p["b"], p["c"], p["d"])
        deriv = (p["a"] * p["b"] * np.exp(p["b"] * probe)
                 + p["c"] * p["d"] * np.exp(p["d"] * probe))
        ok = bool(np.all(deriv < 0) and np.all(curve > 0))
        return {**mod, "monotone": ok, "valid": ok,
                "why": "" if ok else "外推段不严格递减/为正（旧口径：整条曲线）"}

    results.append(
        run_case(
            "W9-M5 外推单调判据退回整条曲线 [1,5N]（R6-2 的真 bug）",
            models_parametric,
            {"attr": "fit_double_exp", "new": old_window_fit},
            "闸门现在只挡外推段",
        )
    )

    # --- W8 回归用例：center=True 在「先截断」下**不**泄漏（应通过泄漏检测，但值确实变了）---
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
