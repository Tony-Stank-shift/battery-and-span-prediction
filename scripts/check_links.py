"""W24：链接与文档完整性检查。

要回答的问题
------------
1. **README 里的每个相对链接都能打开吗？**（W24 验收：附录链接能打开）
2. **docs/ 里引用的文件存在吗？**（跨文档引用最容易在改名后失效）
3. **README 提到的命令与文件真的存在吗？**
4. 外部链接是否可访问（可选，需要外网）。

为什么单独写
------------
「链接能打开」是那种**只有坏掉时才会被发现**的东西：写的时候都在，
改一次文件名就全断，而没人会去手点 30 个链接。

用法
----
    python scripts/check_links.py            # 只查本地（离线可用）
    python scripts/check_links.py --online   # 外加外部链接 HEAD 检查
"""

from __future__ import annotations

import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DOCS = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
CODE_PATH = re.compile(r"`(src/[\w_]+\.py|scripts/[\w_]+\.py|docs/[\w_.]+\.md|report/[^`]+)`")


def local_targets(text: str) -> list[str]:
    out = []
    for m in LINK.finditer(text):
        t = m.group(1).strip()
        if t.startswith(("http://", "https://", "mailto:", "#")):
            continue
        out.append(t.split("#")[0])
    return out


def check_file(path: Path) -> tuple[int, list[str]]:
    text = path.read_text(encoding="utf-8")
    problems: list[str] = []
    n = 0

    for target in local_targets(text):
        n += 1
        resolved = (path.parent / target).resolve()
        if not resolved.exists():
            problems.append(f"{path.name} -> {target}（不存在）")

    # 反引号里写的路径也要存在（README 里大量用这种写法）
    for m in CODE_PATH.finditer(text):
        p = m.group(1)
        if p.startswith(("src/", "scripts/", "docs/", "report/")):
            n += 1
            if not (ROOT / p).exists():
                problems.append(f"{path.name} 提到 `{p}`（不存在）")

    return n, problems


def check_url(url: str, timeout: int = 15) -> str | None:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "dsh-linkcheck"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return None if r.status < 400 else f"{url} -> HTTP {r.status}"
    except Exception as exc:  # noqa: BLE001 - 任何异常都算不可达，但要报出来
        return f"{url} -> {type(exc).__name__}"


def main() -> int:
    online = "--online" in sys.argv
    total, problems = 0, []
    for path in DOCS:
        if not path.exists():
            problems.append(f"缺文件：{path}")
            continue
        n, probs = check_file(path)
        total += n
        problems.extend(probs)

    urls: list[str] = []
    for path in DOCS:
        if path.exists():
            urls.extend(u for u in LINK.findall(path.read_text(encoding="utf-8"))
                        if u.startswith(("http://", "https://")))

    if online:
        for u in sorted(set(urls)):
            bad = check_url(u)
            if bad:
                problems.append(f"外链不可达：{bad}")

    print(f"检查 {len(DOCS)} 个文档，{total} 个本地引用"
          + (f"，{len(set(urls))} 个外链" if online else "（外链未查，加 --online）"))
    if problems:
        for p in problems:
            print("  [FAIL]", p)
        print(f"\n共 {len(problems)} 处问题")
        return 1
    print("全部可打开 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
