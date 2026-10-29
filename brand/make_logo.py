"""Writes the logo files in brand/ (outlined SVG + 2x PNG). Uses the bundled Inter ExtraBold."""
from pathlib import Path

import skia

HERE = Path(__file__).parent
FONT = HERE.parent / "kinescript" / "fonts" / "Inter-ExtraBold.ttf"
INK, PAPER, CUE = 0xff0e0f13, 0xfff3efe6, 0xffff5a36


def draw(c: skia.Canvas, fg: int, word: bool):
    cy, dot_x = 100, 170
    c.drawCircle(dot_x, cy, 30, skia.Paint(AntiAlias=True, Color=CUE))
    sp = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=9,
                    StrokeCap=skia.Paint.kRound_Cap, Color=fg)
    for dy, length in ((-20, 64), (0, 104), (20, 52)):
        c.drawLine(dot_x - 46, cy + dy, dot_x - 46 - length, cy + dy, sp)
    if word:
        f = skia.Font(skia.Typeface.MakeFromFile(str(FONT)), 118)
        c.drawString("kinescript", dot_x + 60, cy + 42, f, skia.Paint(AntiAlias=True, Color=fg))


VARIANTS = {"logo-dark": (PAPER, INK, True, 900), "logo-light": (INK, PAPER, True, 900),
            "logo-transparent": (PAPER, None, True, 900), "mark": (PAPER, None, False, 220),
            "mark-light": (INK, None, False, 220)}

for name, (fg, bg, word, w) in VARIANTS.items():
    stream = skia.FILEWStream(str(HERE / f"{name}.svg"))
    svg = skia.SVGCanvas.Make((w, 200), stream, skia.SVGCanvas.kConvertTextToPaths_Flag)
    if bg is not None:
        svg.drawRect(skia.Rect(0, 0, w, 200), skia.Paint(Color=bg))
    draw(svg, fg, word)
    del svg
    stream.flush()
    del stream
    surf = skia.Surface(w * 2, 400)
    c = surf.getCanvas()
    c.clear(bg if bg is not None else skia.ColorTRANSPARENT)
    c.scale(2, 2)
    draw(c, fg, word)
    surf.makeImageSnapshot().save(str(HERE / f"{name}.png"), skia.kPNG)
    print("wrote", name)
