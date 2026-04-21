# MacDictator

Приложение для диктовки на macOS. Превращает речь в текст локально на Apple Silicon через MLX Whisper (модель `large-v3`) и вставляет результат в активное окно. Опционально чистит текст и переводит через DeepSeek или OpenAI.

## Что умеет

- Локальная транскрипция на устройстве (MLX Whisper `large-v3`) — интернет не нужен.
- Глобальная горячая клавиша: нажал — диктуешь, нажал ещё раз — текст вставляется туда, где курсор.
- Оверлей-индикатор с VU-метром и статусом.
- Очистка текста от слов-паразитов и правка грамматики (опционально, через LLM).
- Перевод (опционально, через LLM).
- История распознаваний.

## Требования

- macOS на Apple Silicon (M1 и новее) — MLX работает только на Apple GPU.
- Права на доступ к микрофону и Accessibility (для вставки текста через `Cmd+V`).

## Установка — простой способ (.dmg)

[**Скачать последний релиз**](https://github.com/OlegArapov/MacDictator/releases/latest)

1. Скачай `MacDictator-X.Y.Z.dmg`
2. Открой DMG и перетащи `MacDictator.app` в `Applications`
3. При первом запуске macOS скажет "cannot be opened because it is from an unidentified developer". Правый клик по приложению → **Open** → подтверди. Это один раз.
4. Разреши доступ к **Микрофону** и **Accessibility** в *System Settings → Privacy & Security*
5. При первой транскрипции скачается модель Whisper `large-v3` (~3 ГБ) в `~/.cache/huggingface/`. Единоразово.

## Установка — из исходников

Нужен Python 3.10+ (рекомендую 3.12 через Homebrew: `brew install python@3.12`).

```bash
git clone https://github.com/OlegArapov/MacDictator.git
cd MacDictator

python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Установка зависимостей занимает пару минут — `torch`, `mlx`, `numpy` тяжёлые.

## Запуск

**Двойным кликом** — открой `MacDictator.command` в Finder. Скрипт сам проверит venv (пересоздаст, если сломан), доустановит зависимости и запустит приложение.

**Из терминала:**

```bash
./MacDictator.command
# или
venv/bin/python app.py
```

**Первый запуск:**
1. macOS попросит разрешить **Микрофон** и **Accessibility** — подтверди в *System Settings → Privacy & Security*. Без Accessibility не работает авто-вставка `Cmd+V`.
2. Первая транскрипция скачает модель `whisper-large-v3-mlx` (~3 ГБ) в `~/.cache/huggingface/hub/`. Кеш глобальный — если модель уже есть (например, от другого проекта), скачивания не будет.

## Перенос проекта в другую папку

Venv привязан к абсолютному пути — после `mv` папки интерпретатор в `venv/bin/python` будет ссылаться на старое место и venv сломается. Решение:

```bash
rm -rf venv
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Модель Whisper в `~/.cache/huggingface/` не теряется и качать её заново не нужно.

## Горячие клавиши

| Клавиша | Действие |
|---|---|
| **Правый Cmd** | Старт / стоп записи |
| **Ctrl+Shift+Space** | Альтернативный старт / стоп |
| **Esc** | Отменить запись без распознавания |

После остановки записи текст автоматически вставляется (`Cmd+V`) в активное окно.

## Настройки

Открываются из окна приложения. Основные опции:

- **Engine** — `MLX` (локально) или OpenAI API.
- **Cleanup** — `Off` / `Lite` (убрать паразитов) / `Medium` (правка грамматики) / `Max` (переписать как связный текст).
- **Cleanup model / Translate model** — `DeepSeek` или `OpenAI`.
- **Translate** — `Off` / язык перевода.
- **Send** — `Paste` (авто-вставка), `Copy` (только в буфер) или `Off`.
- **Mic** — выбор микрофона.

Настройки хранятся в `settings.json`, история — в `history.json` (оба в `.gitignore`).

## API-ключи

Нужны только если используешь очистку/перевод через LLM. Введи их в окне настроек — сохранятся в `keys.json` рядом с `app.py`. Файл в `.gitignore` и никогда не попадёт в репо.

- OpenAI: https://platform.openai.com/api-keys
- DeepSeek: https://platform.deepseek.com/api_keys

## Сборка .app

```bash
./build.sh
```

Соберёт `dist/MacDictator.app` через `py2app`. Требуется иконка `MacDictator.icns` в корне (сгенерировать из PNG можно через `make_icon.py`).

## Структура

- `app.py` — основное приложение (UI, запись, транскрипция, LLM-обработка)
- `tray.py` — иконка в трее
- `prompts.json` — системные промпты для очистки и перевода
- `make_icon.py` — генератор `.icns` из PNG
- `setup.py`, `build.sh` — сборка `.app`

## Лицензия

Проект для личного использования, лицензия пока не задана.
