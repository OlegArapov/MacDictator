import pytest
from progress import format_mmss, ProgressState


def test_format_mmss_basic():
    assert format_mmss(0) == "0:00"
    assert format_mmss(5) == "0:05"
    assert format_mmss(65) == "1:05"
    assert format_mmss(600) == "10:00"


def test_format_mmss_rounds_and_clamps_negative():
    assert format_mmss(-3) == "0:00"
    assert format_mmss(59.6) == "1:00"


def test_fraction_basic_and_clamp():
    p = ProgressState(total_sec=100.0, base_sec=0.0, done_sec=42.0,
                      start_wall=1000.0, prefix="Транскрибация ")
    assert p.fraction() == pytest.approx(0.42)
    p.done_sec = 500.0
    assert p.fraction() == 1.0


def test_fraction_zero_total():
    p = ProgressState(total_sec=0.0, base_sec=0.0, done_sec=0.0,
                      start_wall=1000.0, prefix="X ")
    assert p.fraction() == 0.0


def test_fraction_with_base_offset():
    # второй канал стерео: первый канал (60с) завершён, во втором сделано 30 из 40
    p = ProgressState(total_sec=100.0, base_sec=60.0, done_sec=30.0,
                      start_wall=1000.0, prefix="Транскрибация ")
    assert p.fraction() == pytest.approx(0.90)


def test_status_line_no_eta_before_min_chunks():
    p = ProgressState(total_sec=100.0, base_sec=0.0, done_sec=20.0,
                      start_wall=1000.0, prefix="Транскрибация ")
    # прошло 10с реального времени, но обработан всего 1 чанк
    line = p.status_line(now_wall=1010.0, min_chunks_done=2, chunks_done=1)
    assert line == "Транскрибация 20%"


def test_status_line_with_eta():
    p = ProgressState(total_sec=100.0, base_sec=0.0, done_sec=20.0,
                      start_wall=1000.0, prefix="Транскрибация ")
    # 20с аудио за 10с реального → speed=2 сек/сек → остаток 80/2=40с → 0:40
    line = p.status_line(now_wall=1010.0, min_chunks_done=2, chunks_done=3)
    assert line == "Транскрибация 20% · ~0:40"


def test_status_line_zero_elapsed_no_eta():
    p = ProgressState(total_sec=100.0, base_sec=0.0, done_sec=20.0,
                      start_wall=1000.0, prefix="Транскрибация ")
    line = p.status_line(now_wall=1000.0, min_chunks_done=1, chunks_done=5)
    assert line == "Транскрибация 20%"
