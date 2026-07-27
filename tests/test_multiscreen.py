from macdictator.multiscreen import bubble_positions


def test_single_screen_top_center():
    frames = [(0.0, 0.0, 1512.0, 982.0)]
    pos = bubble_positions(frames, bubble_w=128, margin_top=6)
    assert pos == [((1512 - 128) // 2, 6)]


def test_second_screen_to_the_right_same_height():
    # второй монитор справа, та же высота, origin Cocoa (1512, 0)
    frames = [(0.0, 0.0, 1512.0, 982.0), (1512.0, 0.0, 1920.0, 982.0)]
    pos = bubble_positions(frames, bubble_w=128, margin_top=6)
    assert pos[0] == ((1512 - 128) // 2, 6)
    assert pos[1] == (1512 + (1920 - 128) // 2, 6)


def test_second_screen_taller_above():
    # второй монитор выше главного: Cocoa y>0, tk_top уходит в отрицательные
    frames = [(0.0, 0.0, 1512.0, 982.0), (0.0, 982.0, 1512.0, 1080.0)]
    pos = bubble_positions(frames, bubble_w=128, margin_top=6)
    # tk_top = 982 - (982 + 1080) = -1080
    assert pos[1] == ((1512 - 128) // 2, -1080 + 6)


def test_empty():
    assert bubble_positions([], bubble_w=128) == []
