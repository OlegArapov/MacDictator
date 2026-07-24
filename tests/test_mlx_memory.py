"""Регрессия issue #2: MLX без лимита копит кэш Metal-буферов бесконечно
(13 ГБ IOAccelerator при весах ~3 ГБ), машина уходит в своп."""
import sys
import types

import numpy as np
import pytest

import app


class _FakeMX(types.ModuleType):
    def __init__(self):
        super().__init__("mlx.core")
        self.cache_limits = []
        self.clear_calls = 0

    def set_cache_limit(self, n):
        self.cache_limits.append(n)

    def clear_cache(self):
        self.clear_calls += 1


@pytest.fixture
def fake_mx(monkeypatch):
    fake = _FakeMX()
    fake_pkg = types.ModuleType("mlx")
    fake_pkg.core = fake
    monkeypatch.setitem(sys.modules, "mlx", fake_pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", fake)
    return fake


class _Dummy:
    """Минимальный self для DictatorApp._whisper_transcribe_array."""
    samplerate = 16000
    cancelled = False
    _progress = None
    _progress_prefix = ""

    def __init__(self):
        self.model_var = types.SimpleNamespace(get=lambda: "large-v3")

    def after(self, ms, fn=None, *args):
        pass

    def _vad_split(self, audio, sr):
        return [audio[i:i + sr] for i in range(0, len(audio), sr)]


def test_limit_memory_sets_bounded_cache_limit(fake_mx):
    app._mlx_limit_memory()
    assert fake_mx.cache_limits, "лимит кэша MLX не выставлен"
    assert 0 < fake_mx.cache_limits[-1] <= 2 * 1024 ** 3


def test_limit_memory_survives_missing_mlx(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlx", None)
    monkeypatch.setitem(sys.modules, "mlx.core", None)
    app._mlx_limit_memory()  # не должно бросать


def test_transcribe_loop_clears_cache_per_chunk(fake_mx, monkeypatch):
    monkeypatch.setattr(app, "_mlx_model_downloaded", lambda repo: True)
    monkeypatch.setattr(app, "mlx_whisper", types.SimpleNamespace(
        transcribe=lambda a, **kw: {"segments": [], "text": ""}))
    audio = np.zeros(3 * 16000, dtype=np.float32)  # 3 чанка по 1 секунде
    app.DictatorApp._whisper_transcribe_array(_Dummy(), audio, "ru")
    assert fake_mx.clear_calls >= 3, (
        f"clear_cache вызван {fake_mx.clear_calls} раз, ожидалось >= 3 "
        "(после каждого чанка)")


def test_transcribe_loop_applies_cache_limit(fake_mx, monkeypatch):
    monkeypatch.setattr(app, "_mlx_model_downloaded", lambda repo: True)
    monkeypatch.setattr(app, "mlx_whisper", types.SimpleNamespace(
        transcribe=lambda a, **kw: {"segments": [], "text": ""}))
    audio = np.zeros(16000, dtype=np.float32)
    app.DictatorApp._whisper_transcribe_array(_Dummy(), audio, "ru")
    assert fake_mx.cache_limits, (
        "лимит кэша должен выставляться и в пути транскрибации, "
        "а не только в preload")
