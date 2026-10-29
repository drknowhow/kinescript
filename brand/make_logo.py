"""Writes the logo files in brand/ (outlined SVG + 2x PNG, app icon, favicon).

The mark is a K built from timing: one vertical bar (a track), one eased interpolation path
for the arms, and a cue dot where the path turns. One line, one curve and one circle, so it
survives 16 px. The wordmark is the bundled Space Grotesk Bold, outlined.
"""
from pathlib import Path

import skia

from kinescript import svgpath

HERE = Path(__file__).parent
FONT = HERE.parent / "kinescript" / "fonts" / "SpaceGrotesk-Bold.ttf"
VOID, SHEET, CUE = 0xff0a0c10, 0xfff2f0e8, 0xffff633f

# Mark geometry in a 200x200 box.
BAR = ((52, 36), (52, 164))
ARMS = "M150 36 C 117 36 117 100 84 100 C 117 100 117 164 150 164"
DOT, DOT_R, STROKE = (84, 100), 17, 18

WORD, WORD_SIZE, TRACK = "kinescript", 128, -3.0


def stroke(fg: int, width: float = STROKE) -> skia.Paint:
    return skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=width,
                      StrokeCap=skia.Paint.kRound_Cap, StrokeJoin=skia.Paint.kRound_Join, Color=fg)


def draw_mark(c: skia.Canvas, fg: int, heavy: bool = False):
    """heavy: thicker strokes and a bigger cue for favicon sizes, where 18 px strokes go faint."""
    sp = stroke(fg, 28 if heavy else STROKE)
    c.drawLine(*BAR[0], *BAR[1], sp)
    c.drawPath(svgpath.parse(ARMS), sp)
    c.drawCircle(*DOT, 24 if heavy else DOT_R, skia.Paint(AntiAlias=True, Color=CUE))


def word_blob() -> tuple[skia.TextBlob, float]:
    font = skia.Font(skia.Typeface.MakeFromFile(str(FONT)), WORD_SIZE)
    glyphs = font.textToGlyphs(WORD)
    widths = font.getWidths(glyphs)
    xs, x = [], 0.0
    for w in widths:
        xs.append(x)
        x += w + TRACK
    blob = skia.TextBlob.MakeFromPosTextH(WORD, xs, 0, font)
    return blob, x - TRACK


def draw_logo(c: skia.Canvas, fg: int):
    draw_mark(c, fg)
    blob, _ = word_blob()
    # Baseline sits on the bottom of the bar so the k's ascender lines up near its top.
    c.drawTextBlob(blob, 196, 160, skia.Paint(AntiAlias=True, Color=fg))


def logo_width() -> int:
    return int(196 + word_blob()[1] + 40)


def draw_icon(c: skia.Canvas, size: int):
    """App icon / favicon: the mark on a Void rounded square."""
    c.drawRRect(skia.RRect.MakeRectXY(skia.Rect(0, 0, size, size), size * 0.22, size * 0.22),
                skia.Paint(AntiAlias=True, Color=VOID))
    c.save()
    k = size / 200 * 0.82
    c.translate(size / 2 - 101 * k, size / 2 - 100 * k)
    c.scale(k, k)
    draw_mark(c, SHEET, heavy=size <= 48)
    c.restore()


def write(name: str, w: int, h: int, paint, bg=None):
    stream = skia.FILEWStream(str(HERE / f"{name}.svg"))
    svg = skia.SVGCanvas.Make((w, h), stream, skia.SVGCanvas.kConvertTextToPaths_Flag)
    if bg is not None:
        svg.drawRect(skia.Rect(0, 0, w, h), skia.Paint(Color=bg))
    paint(svg)
    del svg
    stream.flush()
    del stream
    surf = skia.Surface(w * 2, h * 2)
    c = surf.getCanvas()
    c.clear(bg if bg is not None else skia.ColorTRANSPARENT)
    c.scale(2, 2)
    paint(c)
    surf.makeImageSnapshot().save(str(HERE / f"{name}.png"), skia.kPNG)
    print("wrote", name)


def main():
    lw = logo_width()
    write("logo-dark", lw, 200, lambda c: draw_logo(c, SHEET), VOID)
    write("logo-light", lw, 200, lambda c: draw_logo(c, VOID), SHEET)
    write("logo-transparent", lw, 200, lambda c: draw_logo(c, SHEET))
    write("mark", 200, 200, lambda c: draw_mark(c, SHEET))
    write("mark-light", 200, 200, lambda c: draw_mark(c, VOID))
    write("icon", 512, 512, lambda c: draw_icon(c, 512))
    for px in (16, 32, 180):
        surf = skia.Surface(px, px)
        surf.getCanvas().clear(skia.ColorTRANSPARENT)
        draw_icon(surf.getCanvas(), px)
        surf.makeImageSnapshot().save(str(HERE / f"favicon-{px}.png"), skia.kPNG)
    print("wrote favicons")


if __name__ == "__main__":
    main()
