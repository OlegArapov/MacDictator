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
- Python 3.10+.
- Права на доступ к микрофону и Accessibility (для вставки текста через `Cmd+V`).

## Установка

```bash
git clone https://github.com/OlegArapov/MacDictator.git
cd MacDictator

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt  # если будет добавлен; пока см. зависимости ниже
```

Если `requirements.txt` ещё нет, установи зависимости вручную:

```bash
pip install customtkinter openai sounddevice soundfile numpy \
            pyperclip pyautogui pynput psutil mlx-whisper mlx
```

## Запуск

```bash
source venv/bin/activate
python app.py
```

При первом запуске macOS попросит разрешить доступ к микрофону и Accessibility — подтверди в **System Settings → Privacy & Security**.

Первый запуск транскрипции скачает модель `whisper-large-v3-mlx` (~3 ГБ) из Hugging Face в кеш `~/.cache/huggingface`.

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
