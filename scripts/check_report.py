"""W21 自检：报告与产物是否对得上。

为什么要有这个脚本
------------------
报告是唯一一份「散文里带着数字」的交付物，而本项目被审查抓到最多的问题就是
**数字算对了、据它写出的论述是编的**。所以这里做四件事：

1. **结构**：9 章齐全、图 ≥12、表 ≥5、目录是 Word 域、有页码；
2. **图表引用双向一致**：正文引用的图表必须存在（正向），
   存在的图表也必须在正文里被引用（反向——R14-14 就是反向漏网的孤儿图）；
3. **内容非空**：每张表至少有 2 行、每张表至少有一个数字，
   正文里不许残留 markdown 标记（R14-06）；
4. **数字可追**：正文里断言的每个关键数字，都要能在 `outputs/` 的产物里找到来源。

第 4 条只能抽样（正文有近百个数字），所以它不承诺「全覆盖」，
只承诺「登记的这些必须有出处」——这一点也写在输出里，不夸大。

用法
----
    python scripts/check_report.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document

ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / "report" / "20_题3_黄锦添_项目报告.docx"
OUT = ROOT / "outputs"

CHAPTERS = ("1. 引言", "2. 数据介绍", "3. 数据清洗与预处理", "4. 探索性分析与个体差异",
            "5. 模型构建与对比", "6. 不确定性量化、预警与维护策略", "7. 总结与结论",
            "8. 小组成员分工与个人承担", "9. 附录")


def main() -> int:
    assert DOCX.exists(), f"缺报告：{DOCX}（先跑 python scripts/build_report.py）"
    doc = Document(str(DOCX))
    paras = [p.text.strip() for p in doc.paragraphs]
    text = "\n".join(paras)
    fails: list[str] = []

    def need(cond: bool, msg: str) -> None:
        if not cond:
            fails.append(msg)

    # ---- 1. 结构 ----
    h1 = [p.text.strip() for p in doc.paragraphs if p.style.name == "Heading 1"]
    for ch in CHAPTERS:
        need(ch in h1, f"缺章：{ch}")
    fig_caps = [t for t in paras if re.match(r"^图 \d+　", t)]
    tab_caps = [t for t in paras if re.match(r"^表 \d+　", t)]
    need(len(fig_caps) >= 12, f"图题只有 {len(fig_caps)} 个，少于 12")
    need(len(tab_caps) >= 5, f"表题只有 {len(tab_caps)} 个，少于 5")
    need(len(doc.tables) == len(tab_caps), f"表题 {len(tab_caps)} vs 表格 {len(doc.tables)}")
    need(len(doc.inline_shapes) == len(fig_caps), f"图题 {len(fig_caps)} vs 图片 {len(doc.inline_shapes)}")

    # 目录：必须是**我们自己的** TOC 域，且不得残留模板的旧目录（w:sdt）
    xml = doc.element.xml
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    n_sdt = sum(1 for el in doc.element.body if el.tag == f"{ns}sdt")
    need(n_sdt == 0, f"成品里还有 {n_sdt} 个模板旧目录（w:sdt）——会出现两个目录")
    need('TOC \\o' in xml or 'TOC \\\\o' in xml or "TOC" in xml, "没有 TOC 域，目录不能 F9 更新")
    footer_xml = " ".join(s.footer.paragraphs[0]._p.xml for s in doc.sections)
    need("PAGE" in footer_xml, "页脚里没有 PAGE 域（目录的页码无处可对）")

    # ---- 2. 图表引用双向一致 ----
    fig_nums = {int(re.match(r"^图 (\d+)", t).group(1)) for t in fig_caps}
    tab_nums = {int(re.match(r"^表 (\d+)", t).group(1)) for t in tab_caps}
    cited_figs = {int(m) for m in re.findall(r"图 (\d+)", text)}
    cited_tabs = {int(m) for m in re.findall(r"表 (\d+)", text)}
    need(fig_nums == set(range(1, len(fig_nums) + 1)), f"图号有跳号：{sorted(fig_nums)}")
    need(tab_nums == set(range(1, len(tab_nums) + 1)), f"表号有跳号：{sorted(tab_nums)}")
    for n in sorted(cited_figs - fig_nums):
        need(False, f"正文引用了不存在的图 {n}")
    for n in sorted(cited_tabs - tab_nums):
        need(False, f"正文引用了不存在的表 {n}")
    # 反向：每张图表都要被正文提到（题注本身也算提到）
    for n in sorted(fig_nums):
        body_hits = len(re.findall(rf"图 {n}(?![　\d])", text))
        need(body_hits >= 1, f"图 {n} 只出现在题注里，正文没有引用（孤儿图）")
    for n in sorted(tab_nums):
        body_hits = len(re.findall(rf"表 {n}(?![　\d])", text))
        need(body_hits >= 1, f"表 {n} 只出现在题注里，正文没有引用（孤儿表）")

    # ---- 3. 内容非空 / 无 markdown 残留 ----
    for i, t in enumerate(doc.tables, 1):
        need(len(t.rows) >= 2, f"表 {i} 只有 {len(t.rows)} 行")
        cells = " ".join(c.text for r in t.rows[1:] for c in r.cells)
        need(re.search(r"\d", cells) is not None, f"表 {i} 的表体里一个数字都没有")
    # 附录里的代码段本身就是代码，含 markdown 字符是正常的：按字体排除
    code_paras = set()
    for par in doc.paragraphs:
        if par.runs and all((r.font.name or "") == "Consolas" for r in par.runs if r.text.strip()):
            code_paras.add(par.text.strip())   # paras 里存的是 strip 过的版本
    md_hits = [t for t in paras if ("**" in t or "`" in t) and t not in code_paras]
    need(not md_hits, f"正文里残留 markdown 标记 {len(md_hits)} 段，例如：{md_hits[:1]}")

    # ---- 4. 数字可追（抽样，不承诺全覆盖）----
    checks: list[tuple[str, str, bool]] = []

    def claim(desc: str, needle: str, source_md: str) -> None:
        src = (OUT / "tables" / source_md).read_text(encoding="utf-8")
        checks.append((f"{desc}（{source_md}）", needle, needle in text and needle in src))

    claim("容量再生最大回升", "0.1519", "regeneration_events.md")
    claim("B0007 最低容量", "1.4005", "strategy.md")
    claim("M1 的 EOL 估计", "146", "model_comparison.md")
    claim("随机森林训练 R² 上限", "0.9987", "model_rf.md")
    claim("90% 名义区间实际覆盖", "63%", "interval_calibration.md")
    claim("误差放大上限", "10.9", "error_amplification.md")
    claim("SUM 提前量比例下限", "26.3%", "strategy.md")
    claim("任务 B 零基准 MAE", "14.7", "task_b_lobocv.md")
    for desc, needle, ok in checks:
        print(("  [OK]   " if ok else "  [FAIL] ") + f"{desc} -> {needle!r}")
        need(ok, f"{desc}：正文或产物里找不到 {needle!r}")

    if fails:
        print()
        for f in fails:
            print("  [FAIL]", f)
        print(f"\n共 {len(fails)} 项不通过")
        return 1

    print()
    print(f"结构：{len(h1)} 个一级标题（目录 + 9 章）；图 {len(fig_caps)}；表 {len(tab_caps)}")
    print(f"引用：{len(cited_figs)} 个图号、{len(cited_tabs)} 个表号，正反向都一致（无孤儿）")
    print(f"内容：{len(doc.tables)} 张表都有数字；正文无 markdown 残留")
    print(f"数字：{len(checks)} 条抽样可追（正文约 {len(re.findall(r'[0-9]+', text))} 个数字 token，"
          f"**只抽样，不承诺全覆盖**）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
