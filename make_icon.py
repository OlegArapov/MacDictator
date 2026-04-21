"""Generate MacDictator app icon — mic in a dark circle."""
from PIL import Image, ImageDraw
import subprocess, os

SIZE = 1024
img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)

# Background circle with gradient-like dark color
cx, cy = SIZE // 2, SIZE // 2
r = SIZE // 2 - 20
draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill="#1A1A24", outline="#2E2E3E", width=8)

# Microphone body (rounded rectangle)
mic_w, mic_h = 160, 280
mic_x = cx - mic_w // 2
mic_y = cy - 200
mic_r = 80  # corner radius
draw.rounded_rectangle(
    [mic_x, mic_y, mic_x + mic_w, mic_y + mic_h],
    radius=mic_r, fill="#3B82F6"
)

# Microphone grille lines
for i in range(4):
    ly = mic_y + 60 + i * 50
    draw.line([(mic_x + 30, ly), (mic_x + mic_w - 30, ly)],
              fill="#1A1A24", width=6)

# Arc (holder) around mic
arc_r = 160
arc_cy = mic_y + mic_h - 40
draw.arc(
    [cx - arc_r, arc_cy - arc_r, cx + arc_r, arc_cy + arc_r],
    start=0, end=180, fill="#EEEEF2", width=16
)

# Stand (vertical line from arc bottom)
stand_top = arc_cy + arc_r
stand_bottom = stand_top + 120
draw.line([(cx, stand_top), (cx, stand_bottom)], fill="#EEEEF2", width=16)

# Base
base_w = 140
draw.rounded_rectangle(
    [cx - base_w // 2, stand_bottom - 8, cx + base_w // 2, stand_bottom + 20],
    radius=10, fill="#EEEEF2"
)

# Sound waves (right side)
for i, offset in enumerate([60, 120, 180]):
    alpha = 255 - i * 60
    color = (59, 130, 246, alpha)  # blue with fade
    wave_img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    wave_draw = ImageDraw.Draw(wave_img)
    wave_draw.arc(
        [cx + 80 + offset - 80, cy - 180 - 80, cx + 80 + offset + 80, cy - 180 + 80],
        start=-40, end=40, fill=color, width=12
    )
    img = Image.alpha_composite(img, wave_img)

# Left side waves
for i, offset in enumerate([60, 120, 180]):
    alpha = 255 - i * 60
    color = (59, 130, 246, alpha)
    wave_img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    wave_draw = ImageDraw.Draw(wave_img)
    wave_draw.arc(
        [cx - 80 - offset - 80, cy - 180 - 80, cx - 80 - offset + 80, cy - 180 + 80],
        start=140, end=220, fill=color, width=12
    )
    img = Image.alpha_composite(img, wave_img)

# Save as PNG
png_path = "icon.png"
img.save(png_path, "PNG")

# Convert to .icns using sips + iconutil
iconset = "MacDictator.iconset"
os.makedirs(iconset, exist_ok=True)

sizes = [16, 32, 64, 128, 256, 512, 1024]
for s in sizes:
    resized = img.resize((s, s), Image.LANCZOS)
    resized.save(f"{iconset}/icon_{s}x{s}.png")
    if s <= 512:
        resized2 = img.resize((s * 2, s * 2), Image.LANCZOS)
        resized2.save(f"{iconset}/icon_{s}x{s}@2x.png")

subprocess.run(["iconutil", "-c", "icns", iconset, "-o", "MacDictator.icns"])
print("Done: MacDictator.icns")
