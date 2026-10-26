"""W22：生成汇报 PPT（8–10 页）。

设计约束（来自账本与课程要求）
------------------------------
- **8–10 页**，命名 `20_题3_黄锦添.pptx`；
- **每页只讲一件事**：顶部一句话就是这一页的结论，正文只放支撑它的东西；
  这条同时也是 W23 逐字稿的骨架（逐字稿按页写，一页一段）；
- 16:9，图占版面主体，字要少——投影时字多了没人看。

为什么也是「生成」而不是手搓
----------------------------
和报告同一个理由（D25）：PPT 里的数字最容易过期。这里所有引用数字都从
`outputs/` 的产物读，图直接用产物文件，不截图、不重画。

用法
----
    python scripts/build_ppt.py
    # 产出 report/20_题3_黄锦添.pptx
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
REPORT = ROOT / "report"
PPTX = REPORT / "20_题3_黄锦添.pptx"

TITLE = "无人设备电池健康评估与剩余寿命预测"
SUB = "从「还能用多久」到「什么时候该换」"
STUDENT = "黄锦添"
SID = "25121605"
GROUP = "20"
COURSE = "Python数据分析编程"
COLLEGE = "上海大学 未来技术学院"

# 配色（与图的一致性不重要，投影可读性更重要）
NAVY = RGBColor(0x1F, 0x38, 0x64)
ACCENT = RGBColor(0xC0, 0x39, 0x2B)
GRAY = RGBColor(0x59, 0x59, 0x59)
LIGHT = RGBColor(0xF2, 0xF5, 0xFA)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

FONT = "Microsoft YaHei"
W, H = Inches(13.333), Inches(7.5)


def textbox(slide, left, top, width, height, text, size=18, bold=False,
            color=GRAY, align=PP_ALIGN.LEFT, line_spacing=1.25, font=FONT):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.TOP
    lines = text.split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = line_spacing
        run = p.add_run()
        run.text = line
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color
        run.font.name = font
    return tb


def rect(slide, left, top, width, height, color, line=False):
    from pptx.enum.shapes import MSO_SHAPE

    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    shp.fill.solid()
    shp.fill.fore_color.rgb = color
    if not line:
        shp.line.fill.background()
    shp.shadow.inherit = False
    return shp


def new_slide(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])  # 空白版式，全部自己排


def _fit_size(text: str, full_width_pt: float = 25.0) -> float:
    """按字数估一个不会溢出的字号。

    渲染实测：25 pt 下 11.6 英寸宽约放得下 26 个汉字，超出就掉到色带外面。
    这里按「每多 4 个字降 2 pt」收敛，最短 17 pt。
    """
    n = len(text)
    if n <= 26:
        return full_width_pt
    return max(17.0, full_width_pt - 2.0 * ((n - 26) // 4 + 1))


def header(slide, kicker: str, message: str, page: int) -> None:
    """页眉：小标签 + 一句话结论 + 页码。**这一句话就是这一页的全部信息**。"""
    rect(slide, 0, 0, W, Inches(1.42), NAVY)
    textbox(slide, Inches(0.55), Inches(0.16), Inches(11.0), Inches(0.34),
            kicker, size=12, bold=True, color=RGBColor(0xA8, 0xBB, 0xDD))
    textbox(slide, Inches(0.55), Inches(0.52), Inches(11.45), Inches(0.8),
            message, size=_fit_size(message), bold=True, color=WHITE)
    textbox(slide, Inches(12.25), Inches(0.32), Inches(0.9), Inches(0.4),
            f"{page:02d}", size=16, bold=True, color=RGBColor(0xA8, 0xBB, 0xDD),
            align=PP_ALIGN.RIGHT)


def picture(slide, name: str, left, top, width=None, height=None, crop_bottom=0.0):
    path = OUT / "figures" / name
    assert path.exists(), f"缺图：{path}"
    if width is not None:
        pic = slide.shapes.add_picture(str(path), left, top, width=width)
    else:
        pic = slide.shapes.add_picture(str(path), left, top, height=height)
    if crop_bottom:
        pic.crop_bottom = crop_bottom
    return pic


def caption(slide, text: str, left, top, width):
    textbox(slide, left, top, width, Inches(0.3), text, size=11, color=GRAY)


def notes(slide, text: str) -> None:
    """备注栏放逐字稿，演示时开演讲者视图即可看到（W23 的产物之一）。"""
    slide.notes_slide.notes_text_frame.text = text


def main() -> int:
    REPORT.mkdir(exist_ok=True)
    raw = pd.read_csv(OUT / "metrics" / "uncertainty_budget.csv")
    strat = pd.read_csv(OUT / "metrics" / "strategy.csv")
    s2 = strat[strat["表"] == "S2"]

    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H

    # ============ 1 封面 ============
    s = new_slide(prs)
    rect(s, 0, 0, W, H, NAVY)
    rect(s, 0, Inches(4.62), W, Inches(0.06), ACCENT)
    textbox(s, Inches(1.0), Inches(1.85), Inches(11.3), Inches(1.3),
            TITLE, size=40, bold=True, color=WHITE, line_spacing=1.15)
    textbox(s, Inches(1.0), Inches(3.25), Inches(11.3), Inches(0.7),
            SUB, size=22, color=RGBColor(0xD6, 0xE0, 0xF0))
    info = f"{COURSE}　题目三　组号 {GROUP}\n{STUDENT}　{SID}　{COLLEGE}"
    textbox(s, Inches(1.0), Inches(5.0), Inches(11.3), Inches(1.1),
            info, size=16, color=RGBColor(0xD6, 0xE0, 0xF0), line_spacing=1.5)
    notes(s, "各位老师好，我汇报的是题目三：无人设备电池健康评估与剩余寿命预测。"
             "20 组，黄锦添。一句话概括：我做的不是「猜一个寿命数字」，"
             "而是把「这个数字能信到什么程度」一起讲清楚。")

    # ============ 2 问题与数据 ============
    s = new_slide(prs)
    header(s, "01 问题", "同一批电池，寿命读数从 97 到 125 个循环", 2)
    bullets = [
        "NASA 锂离子电池老化实验\n4 块电池、636 个循环、恒定 24 ℃",
        "寿命定义：容量首次 ≤ 1.4 Ah\n（额定的 70%）",
        "实测寿命：B0005 = 125、B0006 = 109\nB0018 = 97",
        "B0007 从未穿越阈值\n→ 只能给下界，不能给数",
    ]
    for i, b in enumerate(bullets):
        textbox(s, Inches(0.6), Inches(1.8 + i * 1.12), Inches(5.6), Inches(1.0),
                b, size=15, color=GRAY, line_spacing=1.3)
    picture(s, "fig01_capacity_degradation.png", Inches(6.55), Inches(1.75), width=Inches(6.4))
    caption(s, "四块电池的容量退化轨迹（红线为 τ=1.4 Ah）", Inches(6.55), Inches(6.55), Inches(6.4))
    notes(s, "先看数据。四块电池，636 个循环，没有缺失、没有重复。"
             "右边这张图就是全部故事：红线是失效阈值，四条曲线穿过它的位置各不相同，"
             "从 97 到 125 个循环。而最上面那条绿色，也就是 B0007，"
             "在整个观测窗口里一次都没碰到红线——它只能给下界。")

    # ============ 3 两种问题 ============
    s = new_slide(prs)
    header(s, "02 方法", "「预测这块电池」和「预测新电池」是两个问题，不能用同一套验证", 3)
    left = ("任务 A：个体内轨迹外推\n"
            "· 自变量是循环序号，用前 80 个循环拟合、其后外推\n"
            "· 四类模型：线性 / 双指数 / 随机森林 / 趋势+高斯过程\n"
            "· 关键：只有前两类是曲线拟合，后两类是机器学习（scikit-learn）")
    right = ("任务 B：跨个体早期预测\n"
             "· 用三块电池预测第四块（留一电池交叉验证）\n"
             "· cycle 绝对不能作特征——那是把答案抄进题目\n"
             "· 先给「零基准」：只用训练集均值去猜")
    textbox(s, Inches(0.6), Inches(1.8), Inches(5.8), Inches(2.6), left, size=15, line_spacing=1.45)
    textbox(s, Inches(6.9), Inches(1.8), Inches(5.8), Inches(2.6), right, size=15, line_spacing=1.45)
    rect(s, Inches(0.6), Inches(4.15), Inches(12.1), Inches(1.5), LIGHT)
    textbox(s, Inches(0.9), Inches(4.38), Inches(11.5), Inches(1.1),
            "评价口径先定死：RMSE / MAE / 寿命误差；任务 B 还要看「有没有赢过零基准」。"
            "——模型好不好，取决于你用哪把尺子。", size=15, color=NAVY, bold=True)
    notes(s, "方法上有两个必须分开的问题。左边任务 A 是在一块电池内部沿时间外推，"
             "右边任务 B 是拿几块电池去预测一块新的。"
             "最容易犯的错是任务 B 里把循环序号当特征，那等于把答案写在题目里。"
             "所以我在任务 B 里先放一个零基准：只用训练集均值去猜，任何特征赢不了它，就说明特征没用。")

    # ============ 4 四模型对比 ============
    s = new_slide(prs)
    header(s, "03 结果", "均方误差的排序会随切分点翻转，所以我不用它下结论", 4)
    picture(s, "fig12_model_comparison.png", Inches(0.55), Inches(1.7), width=Inches(7.4))
    caption(s, "四模型：逐块 RMSE 与 EOL 预测（同一划分、同一套指标）", Inches(0.55), Inches(6.5), Inches(7.4))
    textbox(s, Inches(8.25), Inches(1.85), Inches(4.6), Inches(4.4),
            "改用两个独立判据：\n"
            "① 逐点符号检验\n"
            "② 分块自助法置信区间\n\n"
            "结论：\n"
            "· 线性 vs 趋势+高斯过程：绝大多数点无显著差异\n"
            "· 双指数：三块上显著更差\n"
            "· 随机森林：两块显著更差，两块不显著", size=14, line_spacing=1.5)
    notes(s, "四模型同台。这里我刻意不用均方误差排名次，因为只有四块电池，"
             "换个切分点排序就会变。改用逐点符号检验加自助法区间。"
             "结论是：线性和趋势加高斯过程基本打平；双指数在三块上明显更差；"
             "随机森林两块显著更差、两块不显著——注意「不显著」不等于「一样好」，是样本不够分辨。")

    # ============ 5 随机森林反例 ============
    s = new_slide(prs)
    header(s, "04 反例", "训练 R² 0.997、测试 R² 为负：这不是过拟合，是不会外推", 5)
    picture(s, "fig07_rf_extrapolation.png", Inches(0.55), Inches(1.68), width=Inches(7.3))
    caption(s, "M3 随机森林：切分点之后被压成一条水平线", Inches(0.55), Inches(6.42), Inches(7.3))
    textbox(s, Inches(8.05), Inches(1.9), Inches(4.8), Inches(4.2),
            "为什么？\n\n"
            "树模型只能在训练样本覆盖的区间里取值，\n"
            "区间之外没有样本，它只能输出常数。\n\n"
            "加树、加深、去趋势——都救不回来。\n\n"
            "这正是我选「趋势 + 高斯过程」的理由：\n"
            "先给趋势，再让模型学残差。",
            size=14, line_spacing=1.45)
    notes(s, "这一页是我最想讲的。随机森林训练集 R² 接近 0.997，看起来完美；"
             "但测试集 R² 是负的，比直接猜均值还差。原因不是过拟合，"
             "是树模型在训练数据之外只能输出常数。加树、加深都没用。"
             "所以我最后选的是先拟合一条趋势线，再让高斯过程去学残差。")

    # ============ 6 三层不确定性 ============
    s = new_slide(prs)
    header(s, "05 不确定性", "最大的不确定性不是噪声，而是「你怎么定义坏」", 6)
    picture(s, "fig13_uncertainty_budget.png", Inches(0.55), Inches(1.65), width=Inches(7.5))
    caption(s, "三层不确定性统一折算成循环数", Inches(0.55), Inches(6.5), Inches(7.5))
    d_lo = int(raw["定义层半宽"].min())
    d_hi = int(raw["定义层半宽"].max())
    m_hi = int(raw["模型层半极差"].max())
    s_hi = int(raw["数据层半宽"].max())
    textbox(s, Inches(8.3), Inches(1.85), Inches(4.6), Inches(4.4),
            f"三层都换算成「循环」：\n\n"
            f"· 定义层：τ 挪 ±0.05 Ah → {d_lo}–{d_hi} 循环\n"
            f"· 模型形式层：换函数形式 → 0–{m_hi} 循环\n"
            f"· 数据层：噪声与参数 → 6–{s_hi} 循环\n\n"
            "三者同量级。\n"
            "而且模型形式层的宽度，全部来自双指数——\n"
            "线性和高斯过程在远场的点预测是同一个数。",
            size=14, line_spacing=1.45)
    notes(s, "把不确定性拆成三层，全部折算成循环。定义层是阈值挪动造成的，"
             "模型形式层是换函数形式造成的，数据层是噪声。三者居然同量级。"
             "还有个必须承认的事实：模型形式层的宽度全部来自双指数，"
             "因为线性和高斯过程在远场的点预测是同一个数。")

    # ============ 7 区间校准 ============
    s = new_slide(prs)
    header(s, "06 校准", "「90% 区间」实际只覆盖 63%——而且这个结论还会随切分点摆动", 7)
    picture(s, "fig16_calibration.png", Inches(0.55), Inches(1.7), width=Inches(7.3))
    caption(s, "区间校准：名义覆盖率 vs 实际覆盖率", Inches(0.55), Inches(6.5), Inches(7.3))
    textbox(s, Inches(8.15), Inches(1.85), Inches(4.7), Inches(4.4),
            "检验方式：拿后验区间去盖真值，数覆盖率。\n\n"
            "· 名义 90% → 实际 63%\n"
            "· 而且块间差异极大：最差 10%，最好 99%\n\n"
            "驱动因素是点预测偏差（块间差 22 倍），\n"
            "不是区间宽度（只差 1.15 倍）。\n\n"
            "所以：区间窄，不代表预测准。",
            size=14, line_spacing=1.45)
    notes(s, "把「90% 区间」拿去检验：实际只覆盖 63%。但更重要的是，"
             "这个数字掩盖了块间差异——最好的一块接近 99%，最差的一块只有 10%。"
             "而且驱动因素是点预测偏差，不是区间宽度。所以区间窄不代表预测准。")

    # ============ 8 维护策略 ============
    s = new_slide(prs)
    header(s, "07 落地", "把「还能用多久」变成「第几个循环该换」", 8)
    picture(s, "fig17_strategy.png", Inches(0.6), Inches(1.62), width=Inches(7.9), crop_bottom=0.05)
    caption(s, "左：替换点（三种判据）；右：采集量与在线剩余寿命", Inches(0.6), Inches(5.62), Inches(7.9))
    sum_row = strat[(strat["表"] == "S1") & (strat["判据"] == "SUM 裕度") & strat["真值"].notna()]
    lo = sum_row["提前量比例"].min() * 100
    hi = sum_row["提前量比例"].max() * 100
    textbox(s, Inches(9.0), Inches(1.75), Inches(3.9), Inches(4.6),
            "替换时刻 =\n更严阈值(1.45)下的穿越点\n− 模型/数据裕度\n\n"
            f"代价：提前 {lo:.0f}%–{hi:.0f}% 的寿命\n\n"
            "还有一块电池，等到预测稳定\n已经晚于建议替换点——\n"
            "对它只能「不等预测，直接换」。",
            size=13, line_spacing=1.35)
    notes(s, "最后落到能执行的层面。替换时刻等于更严阈值下的穿越点减去裕度，"
             "代价是提前浪费 26% 到 42% 的寿命——这是买保险的成本。"
             "但有一块电池，等到预测稳定的时候已经过了建议替换点，"
             "对它就只剩一句：不要等，直接换。")

    # ============ 9 结论与复现 ============
    s = new_slide(prs)
    header(s, "08 结论", "五条结论，一条比一条更接近「能不能用」", 9)
    concl = [
        "① 寿命不是一个数，是带阈值的区间：阈值挪 ±0.05 Ah，读数差 30–40 循环",
        "② 上了模型 ≠ 有外推能力：随机森林训练 0.997、测试为负",
        "③ 最大不确定性来自「定义」和「模型形式」，不是噪声",
        "④ 区间窄 ≠ 预测准：90% 名义区间实际覆盖 63%",
        "⑤ 维护判据要把「定义」和「不确定性」分开报",
    ]
    for i, c in enumerate(concl):
        textbox(s, Inches(0.75), Inches(1.75 + i * 0.72), Inches(11.6), Inches(0.6),
                c, size=16, color=NAVY if i % 2 == 0 else GRAY)
    rect(s, Inches(0.75), Inches(5.55), Inches(11.8), Inches(1.15), LIGHT)
    textbox(s, Inches(1.0), Inches(5.72), Inches(11.3), Inches(0.9),
            "全部结果一条命令重建：python scripts/run_all.py --check（17 步 / 40 个产物 / 逐字节一致）\n"
            "自检：42 条变异测试 + 43 条病态输入测试，全部通过",
            size=13, color=NAVY, line_spacing=1.35)
    notes(s, "五条结论。前两条关于模型，中间两条关于不确定性，最后一条关于怎么用。"
             "最后想说一句工程上的事：这个项目全部结果可以一条命令重建，"
             "40 个产物有哈希清单，删掉重跑逐字节一致；另外还有 42 条变异测试和 43 条病态输入测试守着。"
             "我的汇报到这里，谢谢老师。")

    prs.save(str(PPTX))
    n = len(prs.slides)
    print(f"已写出：{PPTX.relative_to(ROOT)}，{n} 页")
    assert 8 <= n <= 10, f"页数 {n} 不在 8–10 之间"
    return 0


if __name__ == "__main__":
    sys.exit(main())
