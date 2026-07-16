import app


def test_filters_strong_youtube_tails():
    assert app._is_hallucination_phrase("Продолжение следует...")
    assert app._is_hallucination_phrase("Субтитры создавал DimaTorzok")
    assert app._is_hallucination_phrase("Субтитры делал А. Синецкая")
    assert app._is_hallucination_phrase("Подписывайтесь на канал")
    assert app._is_hallucination_phrase("Спасибо за просмотр!")
    assert app._is_hallucination_phrase("Thanks for watching")


def test_filters_weak_word_alone():
    # Whisper выдаёт одиночное слово на музыке/тишине
    assert app._is_hallucination_phrase("Музыка")
    assert app._is_hallucination_phrase("[музыка]")
    assert app._is_hallucination_phrase("Субтитры.")
    assert app._is_hallucination_phrase("Thank you.")


def test_keeps_real_speech_with_trigger_word():
    # РЕГРЕССИЯ: слово-триггер внутри нормальной фразы НЕ должно вырезаться
    assert not app._is_hallucination_phrase("Мне нравится эта музыка")
    assert not app._is_hallucination_phrase("Надо поправить субтитры в ролике")
    assert not app._is_hallucination_phrase("Thank you, обсудим завтра")
    assert not app._is_hallucination_phrase("Музыкальный проект запускаем в марте")


def test_keeps_real_speech():
    assert not app._is_hallucination_phrase(
        "Было принято решение, что 1С надо менять")
    assert not app._is_hallucination_phrase("то есть как бы минимум кастома")


def test_does_not_filter_long_text_containing_phrase():
    long_text = "музыка играла весь вечер, и мы обсуждали новую версию биллинга " \
                "и всю бизнес-логику, которую надо выносить наверх из системы"
    assert not app._is_hallucination_phrase(long_text)
