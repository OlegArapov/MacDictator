"""Generate MacDictator app icon — red recording dot on dark background."""
from PIL import Image, ImageDraw
import subprocess, os

SIZE = 1024
img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)

cx, cy = SIZE // 2, SIZE // 2

bg_r = SIZE // 2 - 20
draw.ellipse([cx - bg_r, cy - bg_r, cx + bg_r, cy + bg_r],
             fill="#1A1A24", outline="#2E2E3E", width=8)

dot_r = 220
draw.ellipse([cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r], fill="#E53935")

# Всё кладём рядом со скриптом, в assets/ — запускают его обычно из корня проекта.
HERE = os.path.dirname(os.path.abspath(__file__))

png_path = os.path.join(HERE, "icon.png")
img.save(png_path, "PNG")

iconset = os.path.join(HERE, "MacDictator.iconset")
os.makedirs(iconset, exist_ok=True)

for s in [16, 32, 64, 128, 256, 512, 1024]:
    img.resize((s, s), Image.LANCZOS).save(f"{iconset}/icon_{s}x{s}.png")
    if s <= 512:
        img.resize((s * 2, s * 2), Image.LANCZOS).save(f"{iconset}/icon_{s}x{s}@2x.png")

icns = os.path.join(HERE, "MacDictator.icns")
subprocess.run(["iconutil", "-c", "icns", iconset, "-o", icns], check=True)
print(f"Done: {icns}")
