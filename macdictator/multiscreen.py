"""Чистая геометрия мультимонитора: Cocoa-фреймы экранов → Tk-координаты баблов.

NSScreen.frame — origin в низ-лево главного экрана (Y вверх). Tk — глобальная
система с origin в верх-лево главного экрана (Y вниз). Здесь только конвертация,
без AppKit — поэтому юнит-тестируется.
"""


def bubble_positions(screen_frames, bubble_w, margin_top=6):
    """Для каждого экрана → (x, y) в Tk-координатах для бабла вверху-по-центру."""
    if not screen_frames:
        return []
    primary_h = screen_frames[0][3]
    out = []
    for (x, y, w, h) in screen_frames:
        tk_top = primary_h - (y + h)
        bx = int(x) + (int(w) - bubble_w) // 2
        by = int(tk_top) + margin_top
        out.append((bx, by))
    return out
