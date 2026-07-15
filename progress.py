"""Чистая логика прогресса транскрибации: процент по длительности аудио и ETA.

Без GUI и без времени внутри — `now_wall` инжектится, поэтому юнит-тестируется.
"""
from dataclasses import dataclass


def format_mmss(seconds):
    """Неотрицательные секунды → "M:SS" (минуты без ведущего нуля)."""
    s = max(0, int(round(seconds)))
    m, sec = divmod(s, 60)
    return f"{m}:{sec:02d}"


@dataclass
class ProgressState:
    total_sec: float
    base_sec: float
    done_sec: float
    start_wall: float
    prefix: str

    def fraction(self):
        if self.total_sec <= 0:
            return 0.0
        f = (self.base_sec + self.done_sec) / self.total_sec
        return max(0.0, min(1.0, f))

    def status_line(self, now_wall, min_done_sec_for_eta=0.0,
                    min_chunks_done=2, chunks_done=0):
        pct = int(self.fraction() * 100)
        processed = self.base_sec + self.done_sec
        elapsed = now_wall - self.start_wall
        show_eta = (
            chunks_done >= min_chunks_done
            and processed >= min_done_sec_for_eta
            and elapsed > 0
        )
        if show_eta:
            speed = processed / elapsed
            if speed > 0:
                remaining = (self.total_sec - processed) / speed
                return f"{self.prefix}{pct}% · ~{format_mmss(remaining)}"
        return f"{self.prefix}{pct}%"
