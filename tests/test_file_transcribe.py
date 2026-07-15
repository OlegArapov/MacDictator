import numpy as np
import pytest
import soundfile as sf

import file_transcribe as ft


def test_resample_noop_when_already_16k():
    x = np.linspace(-1, 1, 16000, dtype=np.float32)
    out = ft.resample_to_16k(x, 16000)
    assert out.dtype == np.float32
    assert len(out) == 16000


def test_resample_downsamples_length():
    # 1 s at 48 kHz → ~16000 samples at 16 kHz
    x = np.sin(np.linspace(0, 2 * np.pi * 440, 48000)).astype(np.float32)
    out = ft.resample_to_16k(x, 48000)
    assert out.dtype == np.float32
    assert len(out) == 16000
    assert not np.isnan(out).any()


def test_resample_empty_input():
    out = ft.resample_to_16k(np.zeros(0, dtype=np.float32), 44100)
    assert len(out) == 0


def test_merge_interleaves_by_start_time():
    left = [{"start": 0.0, "text": "привет"}, {"start": 5.0, "text": "как дела"}]
    right = [{"start": 2.0, "text": "здравствуй"}, {"start": 7.0, "text": "хорошо"}]
    out = ft.merge_channel_segments(left, right, "Я", "Собеседник")
    assert out == (
        "Я: привет\n\n"
        "Собеседник: здравствуй\n\n"
        "Я: как дела\n\n"
        "Собеседник: хорошо"
    )


def test_merge_groups_consecutive_same_speaker():
    left = [{"start": 0.0, "text": "раз"}, {"start": 1.0, "text": "два"}]
    right = [{"start": 5.0, "text": "три"}]
    out = ft.merge_channel_segments(left, right, "Я", "Собеседник")
    assert out == "Я: раз два\n\nСобеседник: три"


def test_merge_skips_empty_segments():
    left = [{"start": 0.0, "text": "  "}, {"start": 1.0, "text": "ок"}]
    out = ft.merge_channel_segments(left, [], "Я", "Собеседник")
    assert out == "Я: ок"


def test_merge_empty_returns_empty_string():
    assert ft.merge_channel_segments([], [], "Я", "Собеседник") == ""


def test_unique_txt_path_basic(tmp_path):
    src = tmp_path / "call.mp3"
    src.write_bytes(b"x")
    out = ft.unique_txt_path(str(src))
    assert out == str(tmp_path / "call.txt")


def test_unique_txt_path_collision(tmp_path):
    src = tmp_path / "call.mp3"
    src.write_bytes(b"x")
    (tmp_path / "call.txt").write_text("taken")
    out = ft.unique_txt_path(str(src))
    assert out == str(tmp_path / "call (2).txt")


def _write_wav(path, data, sr):
    sf.write(str(path), data, sr)


def test_decode_stereo_returns_two_channels(tmp_path):
    sr = 48000
    left = np.sin(np.linspace(0, 2 * np.pi * 300, sr)).astype(np.float32)
    right = np.sin(np.linspace(0, 2 * np.pi * 600, sr)).astype(np.float32)
    stereo = np.stack([left, right], axis=1)  # (n, 2)
    p = tmp_path / "call.wav"
    _write_wav(p, stereo, sr)
    channels, out_sr = ft.decode_audio_file(str(p))
    assert out_sr == ft.TARGET_SR
    assert len(channels) == 2
    assert len(channels[0]) == ft.TARGET_SR  # 1 s resampled to 16 kHz
    assert channels[0].dtype == np.float32


def test_decode_mono_returns_one_channel(tmp_path):
    sr = 16000
    mono = np.sin(np.linspace(0, 2 * np.pi * 300, sr)).astype(np.float32)
    p = tmp_path / "memo.wav"
    _write_wav(p, mono, sr)
    channels, out_sr = ft.decode_audio_file(str(p))
    assert len(channels) == 1
    assert len(channels[0]) == 16000


def test_transcribe_file_separate_builds_dialogue(tmp_path):
    sr = 16000
    left = np.sin(np.linspace(0, 2 * np.pi * 300, sr)).astype(np.float32)
    right = np.sin(np.linspace(0, 2 * np.pi * 600, sr)).astype(np.float32)
    stereo = np.stack([left, right], axis=1)
    p = tmp_path / "call.wav"
    sf.write(str(p), stereo, sr)

    calls = {"n": 0}

    def fake(audio, lang, want_segments):
        calls["n"] += 1
        assert want_segments is True
        # первый вызов — левый канал, второй — правый
        if calls["n"] == 1:
            return [{"start": 0.0, "text": "привет"}]
        return [{"start": 1.0, "text": "здравствуй"}]

    out = ft.transcribe_file(str(p), True, fake)
    assert out == "Я: привет\n\nСобеседник: здравствуй"
    assert calls["n"] == 2


def test_transcribe_file_mono_returns_plain_text(tmp_path):
    sr = 16000
    mono = np.sin(np.linspace(0, 2 * np.pi * 300, sr)).astype(np.float32)
    p = tmp_path / "memo.wav"
    sf.write(str(p), mono, sr)

    captured = {}

    def fake(audio, lang, want_segments):
        captured["audio"] = audio
        assert want_segments is False
        return "просто текст"

    out = ft.transcribe_file(str(p), True, fake)  # separate=True, но моно → fallback
    assert out == "просто текст"
    captured_audio = captured["audio"]
    assert captured_audio.ndim == 1
    assert len(captured_audio) == 16000  # 1 с при 16 кГц
    assert not np.isnan(captured_audio).any()


def test_transcribe_file_separate_off_mixes_to_mono(tmp_path):
    sr = 16000
    stereo = np.stack([np.ones(sr, np.float32), np.ones(sr, np.float32)], axis=1)
    p = tmp_path / "call.wav"
    # PCM_16 (дефолт) квантует 1.0 и портит точное сравнение ниже — пишем как float.
    sf.write(str(p), stereo, sr, subtype="FLOAT")

    seen = {}

    def fake(audio, lang, want_segments):
        seen["want_segments"] = want_segments
        seen["audio"] = audio
        return "моно"

    out = ft.transcribe_file(str(p), False, fake)
    assert out == "моно"
    assert seen["want_segments"] is False
    captured_audio = seen["audio"]
    assert np.allclose(captured_audio, np.ones(sr, dtype=np.float32))
    assert len(captured_audio) == sr


def test_transcribe_file_calls_on_decoded_with_durations(tmp_path):
    # стерео-файл 2с при 16кГц
    n = 32000
    data = np.zeros((n, 2), dtype="float32")
    p = tmp_path / "st.wav"
    sf.write(str(p), data, 16000)

    seen = {}

    def on_decoded(durs):
        seen["durs"] = durs

    def fake_transcribe(audio, lang, want_segments):
        return [] if want_segments else ""

    ft.transcribe_file(str(p), separate=True, transcribe_array=fake_transcribe,
                       on_decoded=on_decoded)
    assert len(seen["durs"]) == 2
    assert seen["durs"][0] == pytest.approx(2.0, abs=0.05)
