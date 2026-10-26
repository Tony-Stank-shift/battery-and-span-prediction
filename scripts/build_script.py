"""W23：从 PPT 的备注栏生成逐字稿，并算时间预算。

为什么从备注生成
----------------
逐字稿和 PPT 是最容易「各改各的」的一对：改了 PPT 忘了改稿，或者反过来。
所以唯一出处是 `build_ppt.py` 里的 `notes()`——本脚本只负责读出来、算时长、排版。
要改词，改 `build_ppt.py` 再重跑两个脚本。

时长怎么算
----------
中文演讲的舒适语速约 **240–280 字/分钟**（本项目按 260 计）。
本脚本给出「按 260 字/分」的估算与「要压到 280 秒内，每秒最多说几个字」的余量，
并在最后列出**必须真人朗读核对的三个数**——这台机器不能替你念稿，所以不谎称"实测"。

用法
----
    python scripts/build_script.py
    # 产出 docs/W23_逐字稿.md
"""

from __future__ import annotations

import sys
from pathlib import Path

from pptx import Presentation

ROOT = Path(__file__).resolve().parents[1]
PPTX = ROOT / "report" / "20_题3_黄锦添.pptx"
MD = ROOT / "docs" / "W23_逐字稿.md"

LIMIT_S = 280          # 4:40
CHARS_PER_MIN = 260    # 舒适语速中值
FILLER = 0.08          # 停顿、切换、看图不语的时间占比


def main() -> int:
    assert PPTX.exists(), f"缺 PPT：{PPTX}（先跑 python scripts/build_ppt.py）"
    prs = Presentation(str(PPTX))

    slides = []
    for i, slide in enumerate(prs.slides, 1):
        text = slide.notes_slide.notes_text_frame.text.strip()
        assert text, f"第 {i} 页没有备注（逐字稿就靠它）"
        slides.append((i, text))

    total_chars = sum(len(t) for _, t in slides)
    speech_s = total_chars / CHARS_PER_MIN * 60
    total_s = speech_s * (1 + FILLER)
    mm, ss = divmod(int(round(total_s)), 60)

    lines = [
        "# W23 逐字稿与演示脚本",
        "",
        "> **本文件由 `scripts/build_script.py` 从 PPT 备注栏生成，不要手改。**",
        "> 要改词，改 `scripts/build_ppt.py` 里的 `notes()`，然后重跑：",
        "> `python scripts/build_ppt.py` → `python scripts/build_script.py`。",
        "",
        "## 一、时间预算",
        "",
        f"- 稿件总字数：**{total_chars}** 字",
        f"- 按 {CHARS_PER_MIN} 字/分估算的纯朗读时间：**{speech_s:.0f} 秒**",
        f"- 加上停顿与切页（{FILLER:.0%}）：**约 {total_s:.0f} 秒 = {mm} 分 {ss:02d} 秒**",
        f"- 硬上限：**{LIMIT_S} 秒（4 分 40 秒）**，余量 **{LIMIT_S - total_s:.0f} 秒**",
        "",
        f"> ⚠️ 这是**估算**，不是实测。语速因人而异，必须自己朗读计时一次：",
        f"> 若实测超过 {LIMIT_S} 秒，按第二节每页的字数砍——优先砍第 3、4 页",
        "> （方法与四模型对比可以各少说一句，结论页不能砍）。",
        "",
        "## 二、逐页稿件（每页一句核心结论 + 口播）",
        "",
    ]

    for num, text in slides:
        chars = len(text)
        sec = chars / CHARS_PER_MIN * 60
        lines += [
            f"### 第 {num} 页　（{chars} 字 ｜ 约 {sec:.0f} 秒）",
            "",
            text,
            "",
        ]

    lines += [
        "## 三、演示与代码环节（可选，若老师要求现场跑）",
        "",
        "**目标**：30 秒内证明「结论不是我编的，是代码算出来的」。",
        "",
        "```powershell",
        "cd battery-health-and-lifespan-prediction",
        "$env:PYTHONIOENCODING=\"utf-8\"; $env:MPLCONFIGDIR=\"outputs/_tmp/mplconfig\"",
        "",
        "# 1) 只对账，不重跑：3 秒内证明产物与提交版本逐字节一致",
        "python scripts/run_all.py --check-only",
        "",
        "# 2) 现场演示断言不是空的：把一处断言换成错误实现，看它炸",
        "python scripts/mutation_test.py",
        "```",
        "",
        "**口播（约 20 秒）**：",
        "",
        "> 这条命令不重跑任何东西，只做一件事：把 40 个产物的哈希和提交时的清单比一遍。",
        "> 一致就说明你看到的每一张图、每一个数，都是代码算出来的，没有人手改过。",
        "> 下面这条是变异测试——它不是「测试通过」，而是把断言换成肯定错误的实现，",
        "> 看断言会不会报警。42 条全部报警，说明这些断言不是摆设。",
        "",
        "## 四、录屏前检查清单",
        "",
        "- [ ] 打开演讲者视图（能看到备注，就是这份稿子）",
        "- [ ] 关闭微信 / 邮件通知",
        "- [ ] 字体已确认：微软雅黑（本机已内嵌，换机器要确认）",
        "- [ ] 逐字稿朗读计时一次，确认 ≤ 4:40",
        "- [ ] 视频分辨率 1080p，画面含人脸或声音至少一项（按课程要求）",
        "",
        "## 五、三个必须真人核对的数（本脚本给不了）",
        "",
        "1. **实测总时长**：读完一遍用秒表记，不是看上面的估算；",
        "2. **每页切换是否自然**：估算按字数，实际按语感，某页可能明显偏慢；",
        "3. **口误率**：第一次读通常会卡 2–3 处，第二次再录。",
        "",
    ]

    MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"已写出：{MD.relative_to(ROOT)}")
    print(f"总字数 {total_chars}；估算 {total_s:.0f}s / 上限 {LIMIT_S}s；余量 {LIMIT_S - total_s:.0f}s")
    assert total_s <= LIMIT_S, f"估算 {total_s:.0f}s 已超上限 {LIMIT_S}s，必须删稿"
    return 0


if __name__ == "__main__":
    sys.exit(main())
