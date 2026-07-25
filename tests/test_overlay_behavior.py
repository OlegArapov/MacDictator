"""Issue #4: оверлеи должны показываться и поверх fullscreen-спейсов."""
import app


def test_overlay_behavior_includes_fullscreen_auxiliary():
    b = app.NS_COLLECTION_BEHAVIOR_OVERLAY
    assert b & (1 << 0), "canJoinAllSpaces потерян"
    assert b & (1 << 4), "stationary потерян"
    assert b & (1 << 8), ("нет fullScreenAuxiliary — поверх fullscreen-приложений "
                          "пилюля и баблы не видны")
