import app


def test_filters_known_youtube_tails():
    assert app._is_hallucination_phrase("Продолжение следует...")
    assert app._is_hallucination_phrase("Субтитры создавал DimaTorzok")
    assert app._is_hallucination_phrase("Подписывайтесь на канал")
    assert app._is_hallucination_phrase("Thanks for watching")


def test_keeps_real_speech():
    assert not app._is_hallucination_phrase(
        "Было принято решение, что 1С надо менять")
    assert not app._is_hallucination_phrase("то есть как бы минимум кастома")


def test_does_not_filter_long_text_containing_phrase():
    # длинный реальный сегмент со словом-триггером не должен вырезаться целиком
    long_text = "музыка играла весь вечер, и мы обсуждали новую версию биллинга " \
                "и всю бизнес-логику, которую надо выносить наверх из системы"
    assert not app._is_hallucination_phrase(long_text)
