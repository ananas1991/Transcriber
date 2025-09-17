# Telegram‑бот для транскрибации аудио и видео

Telegram‑бот, который автоматически расшифровывает голосовые сообщения, видео‑кружки и обычные видео в текст на русском языке.

## Попробовать

Откройте демо‑бота и протестируйте прямо сейчас: 

[![Transcriber](https://img.shields.io/badge/Transcriber-blue?logo=telegram)](https://t.me/ultrageniustranscriberbot)


## Ключевые возможности

- 🎵 Расшифровка голосовых сообщений (OGG/Opus)
- 🎥 Поддержка кружков 
- 📹 Расшифровка обычных видео и аудио
- ⚡ Асинхронная обработка и извлечение аудио через `ffmpeg`
- 🧹 Автоматическая очистка временных файлов
- 🐳 Готовый Docker‑образ для развёртывания

## стек

- `Python`, асинхронные хэндлеры
- `python-telegram-bot` — работа с Telegram Bot API
- `openai` — клиент к OpenAI API (модель `whisper-1`)
- `ffmpeg` — извлечение аудио из видео (16 kHz, mono, 64 kbps)
- Docker — продакшн‑контейнеризация

## Как это работает

1. Пользователь отправляет боту голосовое, видео‑кружок или видео.
2. Бот скачивает медиа во временную директорию.
3. Если это видео/кружок — извлекает аудио через `ffmpeg` в `.mp3` (16 kHz, mono).
4. Отправляет аудио в OpenAI Whisper (`whisper-1`) с указанием языка `ru`.
5. Возвращает пользователю читабельный текст и удаляет временные файлы.

## Быстрый старт

### Вариант A — через Docker (рекомендовано)

```bash
# Сборка образа
docker build -t transcriber-bot .

# Запуск с .env файлом
docker run -d --env-file .env --name transcriber transcriber-bot

# Либо напрямую
docker run -d \
  -e TELEGRAM_TOKEN="<your_bot_token>" \
  -e OPENAI_API_KEY="<your_openai_key>" \
  --name transcriber \
  transcriber-bot
```

### Вариант B — локально

```bash
# Зависимости Python
pip install -r requirements.txt

# Установите ffmpeg, если не установлен
# Ubuntu/Debian
sudo apt-get update && sudo apt-get install -y ffmpeg
# macOS (Homebrew)
brew install ffmpeg

cp .env.example .env  # отредактируйте значения

# Запуск бота
python code.py
```

## Переменные окружения

- `TELEGRAM_TOKEN` — токен бота от @BotFather
- `OPENAI_API_KEY` — ключ OpenAI API

Рекомендуется хранить их в `.env` файле:

```env
TELEGRAM_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

## Как получить API токены

- Токен Telegram: напишите @BotFather → `/newbot` → следуйте инструкциям.
- Ключ OpenAI: создайте аккаунт на platform.openai.com → раздел API Keys.

## Использование

1. Запустите бота и отправьте `/start`.
2. Пришлите голосовое, кружок, видео или аудиофайл (до ~20 МБ).
3. Получите текстовую расшифровку в ответ.

## Архитектура 

- Асинхронная обработка: `python-telegram-bot` + `AsyncOpenAI`.
- Извлечение аудио: `ffmpeg` с профилем под Whisper (16 kHz, mono, 64 kbps).
- Логирование: единый формат, логируются ключевые этапы и ошибки.
- Обработка ошибок: сообщения пользователю, очистка временных файлов.
- Ограничения: видео до ~20 МБ (лимит Telegram API).
- Безопасные имена временных файлов (UUID) во избежание коллизий.
- Корректная отправка длинных ответов (разбиение на части < 4096 символов).

## Лицензия

MIT — см. файл [LICENSE](LICENSE).
