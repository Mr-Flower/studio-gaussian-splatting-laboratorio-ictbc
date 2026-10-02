"""Disegna l'icona del programma (assets/icona.ico): un arco fatto di macchie sfumate, come gaussiane.

Serve solo a chi vuole rigenerarla:  python scripts/crea_icona.py
"""
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

SIZE = 512
BACKGROUND = (20, 34, 58, 255)
COLORS = [(42, 120, 214), (88, 166, 255), (235, 104, 52), (255, 196, 120), (27, 175, 122)]


def arch_points():
    """Punti lungo due pilastri e un arco a tutto sesto, in coordinate 0..1."""
    points = []
    for i in range(7):
        y = 0.86 - i * 0.07
        points += [(0.24, y), (0.76, y)]
    for i in range(13):
        t = i / 12
        # semicerchio da sinistra a destra, centro (0.5, 0.44), raggio 0.26
        angle = math.pi * (1 - t)
        points.append((0.5 + 0.26 * math.cos(angle), 0.44 - 0.26 * math.sin(angle)))
    return points


def draw() -> Image.Image:
    image = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    ImageDraw.Draw(image).rounded_rectangle((0, 0, SIZE - 1, SIZE - 1), radius=96, fill=BACKGROUND)
    blobs = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    pen = ImageDraw.Draw(blobs)
    for index, (x, y) in enumerate(arch_points()):
        radius = 30 + (index * 7) % 14
        color = COLORS[index % len(COLORS)] + (235,)
        cx, cy = x * SIZE, y * SIZE
        pen.ellipse((cx - radius, cy - radius * 0.8, cx + radius, cy + radius * 0.8), fill=color)
    blobs = blobs.filter(ImageFilter.GaussianBlur(7))
    mask = Image.new("L", (SIZE, SIZE), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, SIZE - 1, SIZE - 1), radius=96, fill=255)
    image.paste(Image.alpha_composite(image, blobs), (0, 0), mask)
    return image


if __name__ == "__main__":
    target = Path(__file__).resolve().parent.parent / "assets" / "icona.ico"
    target.parent.mkdir(exist_ok=True)
    draw().save(target, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print(target)
