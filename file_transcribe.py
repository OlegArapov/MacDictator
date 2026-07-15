"""Транскрипция готовых аудиофайлов с опциональным разделением стерео-каналов.

Модуль без GUI и без mlx: движок Whisper инжектится колбэком `transcribe_array`,
поэтому логика полностью юнит-тестируется. Декод — через soundfile (без ffmpeg).
"""
import os

import numpy as np
import soundfile as sf

TARGET_SR = 16000


def resample_to_16k(audio, src_sr):
    """1-D float32 моно → ресемпл к 16 кГц линейной интерполяцией.

    Компромисс v1: без анти-алиасинг-фильтра. Для голосовых звонков приемлемо.
    """
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if src_sr == TARGET_SR:
        return audio
    if len(audio) == 0:
        return np.zeros(0, dtype=np.float32)
    n_out = int(round(len(audio) * TARGET_SR / src_sr))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def merge_channel_segments(segs_left, segs_right, label_left, label_right):
    """Слить сегменты двух каналов в хронологический диалог с метками."""
    tagged = []
    for s in segs_left:
        t = (s.get("text") or "").strip()
        if t:
            tagged.append((float(s.get("start", 0.0)), label_left, t))
    for s in segs_right:
        t = (s.get("text") or "").strip()
        if t:
            tagged.append((float(s.get("start", 0.0)), label_right, t))
    tagged.sort(key=lambda x: x[0])

    lines = []
    cur_label = None
    cur_parts = []
    for _, label, text in tagged:
        if label != cur_label:
            if cur_parts:
                lines.append(f"{cur_label}: {' '.join(cur_parts)}")
            cur_label = label
            cur_parts = [text]
        else:
            cur_parts.append(text)
    if cur_parts:
        lines.append(f"{cur_label}: {' '.join(cur_parts)}")
    return "\n\n".join(lines)


def unique_txt_path(source_path):
    """`.../name.<ext>` → `.../name.txt`, при коллизии — `name (2).txt` и т.д."""
    root, _ = os.path.splitext(source_path)
    candidate = root + ".txt"
    n = 2
    while os.path.exists(candidate):
        candidate = f"{root} ({n}).txt"
        n += 1
    return candidate


def decode_audio_file(path):
    """Декодировать аудиофайл в список каналов (1-D float32) при 16 кГц.

    Формат читается soundfile (mp3/wav/flac/ogg/aiff — без ffmpeg).
    Бросает исключение, если формат не поддерживается libsndfile.
    """
    data, sr = sf.read(path, always_2d=True, dtype="float32")  # (n, channels)
    channels = [
        resample_to_16k(np.ascontiguousarray(data[:, c]), sr)
        for c in range(data.shape[1])
    ]
    return channels, TARGET_SR


def transcribe_file(path, separate, transcribe_array, lang="ru",
                    label_left="Я", label_right="Собеседник", on_decoded=None):
    """Декодировать файл и вернуть текст транскрипции.

    separate=True и ≥2 каналов → диалог с метками; иначе — моно, сплошной текст.
    on_decoded(list_of_channel_durations_sec) вызывается один раз после декода.
    """
    channels, _ = decode_audio_file(path)
    if on_decoded is not None:
        on_decoded([len(c) / TARGET_SR for c in channels])
    if separate and len(channels) >= 2:
        segs_left = transcribe_array(channels[0], lang, True)
        segs_right = transcribe_array(channels[1], lang, True)
        return merge_channel_segments(segs_left, segs_right, label_left, label_right)
    if len(channels) == 1:
        mono = channels[0]
    else:
        mono = np.mean(np.stack(channels, axis=0), axis=0).astype(np.float32)
    return transcribe_array(mono, lang, False)
