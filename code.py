import os
import asyncio
import tempfile
from pathlib import Path
import logging
from typing import Optional, List
import shutil
from uuid import uuid4

try:
    # Optional: load .env in local runs
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    # dotenv is optional; Docker/env vars still work
    pass

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from openai import AsyncOpenAI

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Конфигурация
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not TELEGRAM_TOKEN or not OPENAI_API_KEY:
    logger.error("TELEGRAM_TOKEN и OPENAI_API_KEY должны быть установлены в переменных окружения")
    raise ValueError("Missing required environment variables")

# Проверка наличия ffmpeg
if shutil.which("ffmpeg") is None:
    logger.warning("ffmpeg не найден в PATH — обработка видео может не работать")

# Инициализация OpenAI клиента
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Кэш для временных файлов
temp_dir = tempfile.gettempdir()

def _tmp_name(prefix: str, ext: str) -> Path:
    """Генерация безопасного имени во временной директории"""
    return Path(temp_dir) / f"{prefix}_{uuid4().hex}.{ext}"

def _split_text(text: str, limit: int = 4000) -> List[str]:
    """Разбивает длинный текст на части, безопасные для Telegram (<=4096)."""
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + limit, len(text))
        # стараться резать по пробелам/строкам
        if end < len(text):
            cut = text.rfind("\n", start, end)
            if cut == -1:
                cut = text.rfind(" ", start, end)
            if cut != -1 and cut > start:
                end = cut
        chunks.append(text[start:end])
        start = end
    return chunks or [text]

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /start"""
    await update.message.reply_text(
        "Transcriber - бот разработан @denisvatlin\n\nПерешлите сюда любые кружки/аудио/видео/голосовые и получите извлеченный текст."
    )

async def transcribe_audio(file_path: Path) -> Optional[str]:
    """Транскрибация аудио с помощью Whisper API"""
    try:
        with open(file_path, "rb") as audio_file:
            transcript = await client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="ru"  # Указываем русский язык для оптимизации
            )
        return transcript.text
    except Exception as e:
        logger.error(f"Ошибка транскрипции: {e}")
        return None

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка голосовых сообщений"""
    message = await update.message.reply_text("⏳ Обрабатываю...")
    
    try:
        # Получаем файл
        file = await update.message.voice.get_file()
        
        # Создаем временный файл
        temp_path = _tmp_name("voice", "ogg")
        
        # Скачиваем файл
        await file.download_to_drive(temp_path)
        
        # Транскрибируем
        text = await transcribe_audio(temp_path)
        
        # Удаляем временный файл
        temp_path.unlink(missing_ok=True)
        
        if text:
            chunks = _split_text(text)
            # Первая часть — редактируем сообщение-заглушку
            await message.edit_text(f"📝 Расшифровка:\n\n{chunks[0]}")
            # Остальные части — отдельными сообщениями
            for part in chunks[1:]:
                await update.message.reply_text(part)
        else:
            await message.edit_text("❌ Не удалось расшифровать сообщение.")
            
    except Exception as e:
        logger.error(f"Ошибка обработки голосового сообщения: {e}")
        await message.edit_text("❌ Произошла ошибка при обработке.")

async def extract_audio_from_video(video_path: Path, audio_path: Path) -> bool:
    """Извлечение аудио из видео файла используя ffmpeg"""
    try:
        import subprocess
        
        # Быстрая экстракция аудио без перекодирования где возможно
        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-vn",  # Без видео
            "-acodec", "libmp3lame",  # MP3 кодек
            "-ar", "16000",  # Частота дискретизации 16kHz для Whisper
            "-ac", "1",  # Моно для уменьшения размера
            "-b:a", "64k",  # Битрейт для баланса качества и размера
            "-y",  # Перезаписать если существует
            str(audio_path)
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        result = await process.communicate()
        return process.returncode == 0
        
    except Exception as e:
        logger.error(f"Ошибка извлечения аудио: {e}")
        return False

async def handle_video_note(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка видео-кружков"""
    message = await update.message.reply_text("⏳ Обрабатываю кружок...")
    
    try:
        # Получаем файл
        file = await update.message.video_note.get_file()
        
        # Создаем временный файл для видео
        video_path = _tmp_name("video_note", "mp4")
        audio_path = _tmp_name("audio_note", "mp3")
        
        # Скачиваем видео
        await file.download_to_drive(video_path)
        
        # Извлекаем аудио из видео
        if await extract_audio_from_video(video_path, audio_path):
            # Удаляем видео файл
            video_path.unlink(missing_ok=True)
            
            # Транскрибируем аудио
            text = await transcribe_audio(audio_path)
            
            # Удаляем аудио файл
            audio_path.unlink(missing_ok=True)
            
            if text:
                chunks = _split_text(text)
                await message.edit_text(f"📝 Расшифровка кружка:\n\n{chunks[0]}")
                for part in chunks[1:]:
                    await update.message.reply_text(part)
            else:
                await message.edit_text("❌ Не удалось расшифровать кружок.")
        else:
            # Убираем временные файлы в случае ошибки
            video_path.unlink(missing_ok=True)
            audio_path.unlink(missing_ok=True)
            await message.edit_text(
                "❌ Ошибка обработки видео. Убедитесь, что ffmpeg установлен."
            )
            
    except Exception as e:
        logger.error(f"Ошибка обработки видео-кружка: {e}")
        await message.edit_text("❌ Произошла ошибка при обработке.")

async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка видео файлов"""
    message = await update.message.reply_text("⏳ Обрабатываю видео...")
    
    try:
        # Получаем файл
        file = await update.message.video.get_file()
        
        # Проверяем размер файла (ограничение Telegram API ~ 20MB)
        if file.file_size > 20 * 1024 * 1024:  # 20MB
            await message.edit_text(
                "❌ Видео слишком большое. Максимальный размер: 20MB."
            )
            return
        
        # Создаем временные файлы
        video_path = _tmp_name("video", "mp4")
        audio_path = _tmp_name("audio", "mp3")
        
        # Скачиваем видео
        await file.download_to_drive(video_path)
        
        # Извлекаем аудио из видео
        if await extract_audio_from_video(video_path, audio_path):
            # Удаляем видео файл
            video_path.unlink(missing_ok=True)
            
            # Транскрибируем аудио
            text = await transcribe_audio(audio_path)
            
            # Удаляем аудио файл
            audio_path.unlink(missing_ok=True)
            
            if text:
                chunks = _split_text(text)
                await message.edit_text(f"📝 Расшифровка видео:\n\n{chunks[0]}")
                for part in chunks[1:]:
                    await update.message.reply_text(part)
            else:
                await message.edit_text("❌ Не удалось расшифровать видео.")
        else:
            # Убираем временные файлы в случае ошибки
            video_path.unlink(missing_ok=True)
            audio_path.unlink(missing_ok=True)
            await message.edit_text(
                "❌ Ошибка обработки видео. Убедитесь, что ffmpeg установлен."
            )
            
    except Exception as e:
        logger.error(f"Ошибка обработки видео: {e}")
        await message.edit_text("❌ Произошла ошибка при обработке.")

async def handle_unknown(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка неподдерживаемых типов сообщений"""
    await update.message.reply_text(
        "ℹ️ Пожалуйста, отправьте голосовое сообщение, видео-кружок или видео файл для расшифровки."
    )

def main() -> None:
    """Главная функция"""
    # Создаем приложение
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Добавляем обработчики
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice))
    
    # Поддержка обычных аудиофайлов (mp3/m4a/ogg/wav)
    async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = await update.message.reply_text("⏳ Обрабатываю аудио...")
        try:
            file = await update.message.audio.get_file()
            temp_path = _tmp_name("audio", "bin")
            await file.download_to_drive(temp_path)

            text = await transcribe_audio(temp_path)
            temp_path.unlink(missing_ok=True)

            if text:
                chunks = _split_text(text)
                await message.edit_text(f"📝 Расшифровка аудио:\n\n{chunks[0]}")
                for part in chunks[1:]:
                    await update.message.reply_text(part)
            else:
                await message.edit_text("❌ Не удалось расшифровать аудио.")
        except Exception as e:
            logger.error(f"Ошибка обработки аудио: {e}")
            await message.edit_text("❌ Произошла ошибка при обработке.")

    application.add_handler(MessageHandler(filters.AUDIO, handle_audio))
    application.add_handler(MessageHandler(filters.VIDEO_NOTE, handle_video_note))
    application.add_handler(MessageHandler(filters.VIDEO, handle_video))
    application.add_handler(MessageHandler(
        ~filters.VOICE & ~filters.VIDEO_NOTE & ~filters.VIDEO & ~filters.COMMAND,
        handle_unknown
    ))
    
    # Запускаем бота
    logger.info("Бот запущен...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
