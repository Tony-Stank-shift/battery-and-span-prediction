"""一键复现入口。

当前包含 W1–W19 的 17 个步骤；W20 会把它扩展成完整管线
（不确定性 → 图表 → 报告）。

注意：`scripts/mutation_test.py` 不在这里。它验证的是「断言会不会失败」，
不产出任何数据，属于自检工具，单独跑。

用法
----
    python scripts/run_all.py             # 只跑 17 步
    python scripts/run_all.py --check     # 跑完再与 outputs/MANIFEST.sha256 对账
    python scripts/run_all.py --check-only # 只对账，不重跑

约定
----
- 任何一步返回非零退出码，管线立刻中止并透传退出码。
- `--check` 对账失败返回 2（与「某一步失败」的退出码区分开）。
- 子进程继承 stdio（不捕获管道），这样报错信息能实时看到。
- 统一注入 MPLCONFIGDIR 与 PYTHONIOENCODING，保证换台机器也能跑。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from config import MPLCONFIG_DIR  # noqa: E402

STEPS: list[tuple[str, Path]] = [
    ("数据体检", ROOT / "src" / "sanity_check.py"),
    ("清洗与预处理", ROOT / "src" / "preprocess.py"),
    ("退化曲线与个体差异", ROOT / "src" / "explore.py"),
    ("个体差异五维量化", ROOT / "src" / "individual.py"),
    ("容量再生检测与分解", ROOT / "src" / "regeneration.py"),
    ("EOL 阈值敏感性", ROOT / "src" / "sensitivity.py"),
    ("早期特征与泄漏证明", ROOT / "src" / "features.py"),
    ("评测口径", ROOT / "src" / "evaluate.py"),
    ("M1 线性 / M2 双指数", ROOT / "src" / "models_parametric.py"),
    ("M3 随机森林（外推失效）", ROOT / "src" / "models_ml.py"),
    ("M4 趋势 + GPR", ROOT / "src" / "models_gpr.py"),
    ("四模型对比主表", ROOT / "src" / "compare.py"),
    ("三层不确定性", ROOT / "src" / "uncertainty.py"),
    ("最早可信预警时刻", ROOT / "src" / "warning.py"),
    ("两场景误差放大", ROOT / "src" / "amplification.py"),
    ("区间校准 PICP/MPIW", ROOT / "src" / "calibration.py"),
    ("维护策略", ROOT / "src" / "strategy.py"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description="一键重跑全部步骤")
    ap.add_argument("--check", action="store_true", help="跑完与产物清单对账")
    ap.add_argument("--check-only", action="store_true", help="只对账，不重跑")
    args = ap.parse_args()

    # stdout 被重定向到管道时 Python 默认块缓冲，会导致父进程的打印晚于子进程输出，
    # 日志顺序错乱。强制行缓冲。
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:  # 理论上不会发生（Python >= 3.7），保底
        pass

    import manifest

    if args.check_only:
        return 0 if manifest.check() == 0 else 2

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

    print(f"项目根目录：{ROOT}")
    print(f"解释器：{sys.executable}")

    timings: list[tuple[str, float]] = []
    for name, script in STEPS:
        rel = script.relative_to(ROOT)
        print()
        print("=" * 70)
        print(f"步骤：{name}  ({rel})")
        print("=" * 70)
        t0 = time.perf_counter()
        result = subprocess.run([sys.executable, str(script)], cwd=ROOT, env=env)
        elapsed = time.perf_counter() - t0
        timings.append((name, elapsed))

        if result.returncode != 0:
            print()
            print(f"[FAIL] {name} 退出码 {result.returncode}，管线中止")
            return result.returncode

    print()
    print("=" * 70)
    print(f"全部 {len(STEPS)} 个步骤通过（耗时 {sum(t for _, t in timings):.1f}s）")
    print("=" * 70)
    # 每步耗时：只打印，不写文件——写进 outputs/ 会破坏「逐字节一致」
    print()
    print("每步耗时：")
    for name, t in timings:
        print(f"  {t:6.1f}s  {name}")

    if args.check:
        print()
        print("—— 与产物清单对账 ——")
        if manifest.check() != 0:
            print("[FAIL] 产物与 MANIFEST.sha256 不一致（退出码 2）")
            return 2
        print("[OK] 产物与提交版本逐字节一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
