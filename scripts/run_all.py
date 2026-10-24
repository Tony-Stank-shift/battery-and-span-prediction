"""一键复现入口。

当前阶段只包含 W1–W2 的步骤；W20 会把它扩展成完整管线
（清洗 → 探索 → 特征 → 建模 → 不确定性 → 图表 → 报告）。

用法
----
    python scripts/run_all.py

约定
----
- 任何一步返回非零退出码，管线立刻中止并透传退出码。
- 子进程继承 stdio（不捕获管道），这样报错信息能实时看到。
- 统一注入 MPLCONFIGDIR 与 PYTHONIOENCODING，保证换台机器也能跑。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import MPLCONFIG_DIR  # noqa: E402

STEPS: list[tuple[str, Path]] = [
    ("数据体检", ROOT / "src" / "sanity_check.py"),
    ("清洗与预处理", ROOT / "src" / "preprocess.py"),
]


def main() -> int:
    # stdout 被重定向到管道时 Python 默认块缓冲，会导致父进程的打印晚于子进程输出，
    # 日志顺序错乱。强制行缓冲。
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:  # 理论上不会发生（Python >= 3.7），保底
        pass

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

    print(f"项目根目录：{ROOT}")
    print(f"解释器：{sys.executable}")

    for name, script in STEPS:
        rel = script.relative_to(ROOT)
        print()
        print("=" * 70)
        print(f"步骤：{name}  ({rel})")
        print("=" * 70)
        result = subprocess.run([sys.executable, str(script)], cwd=ROOT, env=env)

        if result.returncode != 0:
            print()
            print(f"[FAIL] {name} 退出码 {result.returncode}，管线中止")
            return result.returncode

    print()
    print("=" * 70)
    print(f"全部 {len(STEPS)} 个步骤通过")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
