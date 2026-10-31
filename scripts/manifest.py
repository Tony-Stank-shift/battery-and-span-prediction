"""产物清单：生成 / 校验 `outputs/MANIFEST.sha256`。

W20 的验收标准是「清空 outputs 后一条命令全重建；**与提交版本无差异**」。
第二条需要一份**可对账的清单**——本脚本就是那把尺子。

用法
----
    python scripts/manifest.py            # 生成/覆盖清单
    python scripts/manifest.py --check    # 校验，不一致退出码 1

清单格式（纯文本，可 diff）
--------------------------
    # 生成环境（这部分**必须**记，否则「一致」是个假承诺）
    python 3.12.10 | Windows-11 | numpy 2.x | pandas 2.x | scikit-learn 1.9 | scipy 1.x | matplotlib 3.x
    # 计数
    files 40
    # 路径（相对仓库根）  完整 sha256  字节数
    outputs/figures/fig01_xxx.png   <64 hex>   123456

为什么记环境
------------
PNG 的字节对 matplotlib 版本敏感：同一个脚本、不同版本画出来的图可能不同。
所以清单只承诺「**同一环境下**逐字节一致」；跨环境必须重跑并重新生成清单。

刻意不做的事
------------
- **不把 `_tmp/` 纳入清单**：那是审查脚本与缓存，不是交付物。
- **不把清单自己纳入清单**：否则每次生成都会变。
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import sys

# Windows 默认控制台是 GBK，print 出 ✓/✗ 会抛 UnicodeEncodeError；
# 在入口处强制 UTF-8，保证 `run_all.py --check` 在任何控制台都不崩。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:  # pragma: no cover
    pass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
# **放在仓库根，不放在 outputs/ 里**：验收要求「清空 outputs 后重建再对账」，
# 清单若在 outputs/ 内会被一起删掉，那这个实验就没法成立。
MANIFEST = ROOT / "MANIFEST.sha256"

SCAN_DIRS = ("figures", "tables", "metrics")
"""参与对账的三个交付目录。`_tmp/` 明确排除。"""


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def env_line() -> str:
    """一行环境描述。缺包就写 `?`，不因为某个包没装就让清单生成失败。"""
    try:
        import matplotlib
        import numpy
        import pandas
        import scipy
        import sklearn

        return (
            f"python {sys.version.split()[0]} | {platform.platform()} | "
            f"numpy {numpy.__version__} | pandas {pandas.__version__} | "
            f"scikit-learn {sklearn.__version__} | scipy {scipy.__version__} | "
            f"matplotlib {matplotlib.__version__}"
        )
    except ImportError as exc:  # pragma: no cover - 只在环境残缺时触发
        return f"环境不完整（{exc}）"


def collect() -> dict[str, tuple[str, int]]:
    """相对路径 → (sha256, 字节数)。"""
    rows: dict[str, tuple[str, int]] = {}
    for d in SCAN_DIRS:
        base = OUTPUTS / d
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if not p.is_file():
                continue
            rows[p.relative_to(ROOT).as_posix()] = (sha256_of(p), p.stat().st_size)
    assert rows, "outputs/ 下没有任何产物——先跑 `python scripts/run_all.py`"
    return rows


def write() -> int:
    rows = collect()
    lines = [
        "# 由 scripts/manifest.py 生成，**不要手工编辑**",
        f"# {env_line()}",
        f"files {len(rows)}",
    ]
    for rel, (digest, size) in rows.items():
        lines.append(f"{rel}  {digest}  {size}")
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已写出 {MANIFEST.relative_to(ROOT)}：{len(rows)} 个产物")
    return 0


def parse() -> tuple[dict[str, str], str]:
    """读清单 → ({路径: sha256}, 环境行)。"""
    if not MANIFEST.exists():
        raise FileNotFoundError(
            f"缺 {MANIFEST.relative_to(ROOT)}——先跑 `python scripts/manifest.py`"
        )
    rows: dict[str, str] = {}
    env = ""
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if line.startswith("#"):
            # 环境行是唯一以 "# python " 开头的那条（不能拿第一条注释冒充）
            if line.startswith("# python ") or line.startswith("# 环境不完整"):
                env = line[2:]
            continue
        if line.startswith("files "):
            continue
        parts = line.split("  ")
        assert len(parts) == 3, f"清单行格式不对：{line!r}"
        rows[parts[0]] = parts[1]
    return rows, env


def check(verbose: bool = True) -> int:
    """对账。返回 0 表示一致，1 表示有差异（并打印差异清单）。"""
    want, env = parse()
    got = collect()
    missing = sorted(set(want) - set(got))
    extra = sorted(set(got) - set(want))
    changed = sorted(k for k in set(want) & set(got) if want[k] != got[k][0])

    ok = not (missing or extra or changed)
    if verbose:
        print(f"清单环境：{env}")
        print(f"比对：清单 {len(want)} 个，实际 {len(got)} 个")
        for tag, items in (("缺失", missing), ("多出", extra), ("内容变了", changed)):
            for k in items:
                print(f"  [{tag}] {k}")
        print("一致 ✓" if ok else "不一致 ✗")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="产物清单生成 / 校验")
    ap.add_argument("--check", action="store_true", help="校验而不是生成")
    args = ap.parse_args()
    return check() if args.check else write()


if __name__ == "__main__":
    sys.exit(main())
