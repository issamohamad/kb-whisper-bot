import os
import sys
import logging
import tempfile
import threading
import torch
from flask import Flask
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pydub import AudioSegment

# Create Flask app for health checks
app = Flask(__name__)

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Startup debug info
print("Starting application...")
print(f"Python version: {sys.version}")
print(f"Environment PORT variable: {os.environ.get('PORT', 'not set')}")

# Load Telegram token from environment; remove the fallback in production if possible.
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "7642881098:AAFGbpNoK8vjo3dnv7UVI4_KvVRGV3zH9jc")
if not TELEGRAM_TOKEN:
    raise ValueError("No Telegram token provided! Please set the TELEGRAM_TOKEN environment variable.")
else:
    # For security, only log a masked version
    print(f"Telegram token loaded: {TELEGRAM_TOKEN[:5]}...{TELEGRAM_TOKEN[-5:]}")

# Global variables for the transcription model
transcription_pipe = None
device = "cpu"
torch_dtype = torch.float32
model_loaded = False

# Flask health check endpoint
@app.route('/')
def hello_world():
    status = "Model loaded and ready" if model_loaded else "Model loading in progress"
    return f'KB-Whisper Bot is running! Status: {status}'

def setup_model():
    """Load the transcription model in a background thread."""
    global transcription_pipe, device, torch_dtype, model_loaded
    if transcription_pipe is not None:
        return
    print("Starting model setup...")
    if torch.cuda.is_available():
        device = "cuda:0"
        torch_dtype = torch.float16
    logger.info(f"Setting up KB-Whisper model on {device}...")
    try:
        model_id = "KBLab/kb-whisper-small"  # Smaller model for faster inference
        print(f"Loading model: {model_id}")
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, use_safetensors=True
        )
        model.to(device)
        processor = AutoProcessor.from_pretrained(model_id)
        # `pipeline` expects an integer device index (0 for first GPU, -1 for
        # CPU). When running on GPU ``device`` is the string ``"cuda:0"`` so we
        # need to convert it to ``0`` to avoid a type error.
        pipeline_device = 0 if device != "cpu" else -1
        transcription_pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=pipeline_device,
        )
        logger.info("KB-Whisper model loaded successfully!")
        print("KB-Whisper model loaded successfully!")
        model_loaded = True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        print(f"Error loading model: {e}")

def start(update: Update, context: CallbackContext) -> None:
    """Handle /start command."""
    user = update.effective_user
    update.message.reply_text(
        f"Hi {user.first_name}! I'm a transcription bot using KB-Whisper.\n"
        "Send me an audio message or file, and I'll transcribe it with timestamps.\n\n"
        "Supported languages: Swedish (default), English, Norwegian, Danish, Finnish\n"
        "Use /language [code] to change language (sv, en, no, da, fi)"
    )
    logger.info(f"User {user.id} ({user.first_name}) started the bot")

def help_command(update: Update, context: CallbackContext) -> None:
    """Handle /help command."""
    update.message.reply_text(
        "Send me an audio message or file to transcribe it.\n\n"
        "Commands:\n"
        "/start - Start the bot\n"
        "/help - Show this help message\n"
        "/language [code] - Change language (sv, en, no, da, fi)\n"
        "/info - Show bot information"
    )

def set_language(update: Update, context: CallbackContext) -> None:
    """Set the transcription language."""
    if not context.args:
        update.message.reply_text("Please specify a language code: sv, en, no, da, fi (e.g., /language en)")
        return
    language = context.args[0].lower()
    valid_languages = {"sv", "en", "no", "da", "fi"}
    if language not in valid_languages:
        update.message.reply_text(f"Invalid language code. Valid codes are: {', '.join(valid_languages)}")
        return
    context.user_data["language"] = language
    language_names = {"sv": "Swedish", "en": "English", "no": "Norwegian", "da": "Danish", "fi": "Finnish"}
    update.message.reply_text(f"Language set to {language_names[language]} ({language}).")
    logger.info(f"User {update.effective_user.id} changed language to {language}")

def info(update: Update, context: CallbackContext) -> None:
    """Display bot information."""
    user_language = context.user_data.get("language", "sv")
    language_names = {"sv": "Swedish", "en": "English", "no": "Norwegian", "da": "Danish", "fi": "Finnish"}
    model_status = "Loaded and ready" if model_loaded else "Still loading"
    update.message.reply_text(
        f"KB-Whisper Transcription Bot\n\n"
        f"Model: KBLab/kb-whisper-small\n"
        f"Device: {device}\n"
        f"Model status: {model_status}\n"
        f"Current language: {language_names[user_language]} ({user_language})\n\n"
        "Created with ❤️ using KB-Whisper from KBLab"
    )

def process_audio(update: Update, context: CallbackContext) -> None:
    """Process incoming audio messages for transcription."""
    global model_loaded
    if not model_loaded or transcription_pipe is None:
        update.message.reply_text("I'm still loading the transcription model. Please try again later.")
        return
    language = context.user_data.get("language", "sv")
    processing_message = update.message.reply_text(f"Processing your audio in {language} language... Please wait.")
    user = update.effective_user
    logger.info(f"User {user.id} sent an audio file")
    try:
        if update.message.voice:
            file = update.message.voice.get_file()
            file_extension = ".ogg"
        elif update.message.audio:
            file = update.message.audio.get_file()
            file_extension = os.path.splitext(update.message.audio.file_name)[1]
        else:
            file = update.message.document.get_file()
            file_extension = os.path.splitext(update.message.document.file_name)[1]
        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
            temp_path = temp_file.name
        file.download(temp_path)
        wav_path = temp_path
        if file_extension.lower() not in ['.wav']:
            try:
                audio = AudioSegment.from_file(temp_path)
                wav_path = temp_path + ".wav"
                audio.export(wav_path, format="wav")
            except Exception as e:
                logger.error(f"Error converting audio: {e}")
                processing_message.edit_text(f"Error processing audio: {str(e)}")
                os.unlink(temp_path)
                return
        try:
            result = transcription_pipe(
                wav_path,
                chunk_length_s=30,
                stride_length_s=3,
                return_timestamps=True,
                generate_kwargs={"task": "transcribe", "language": language, "return_timestamps": True}
            )
            formatted_output = ""
            if "chunks" in result:
                for chunk in result["chunks"]:
                    minutes = int(chunk["timestamp"][0]) // 60
                    seconds = int(chunk["timestamp"][0]) % 60
                    formatted_output += f"[{minutes:02d}:{seconds:02d}] {chunk['text']}\n\n"
            else:
                formatted_output = result["text"]
            if len(formatted_output) <= 4000:
                processing_message.edit_text(formatted_output)
            else:
                processing_message.edit_text("Transcription complete! Sending in multiple messages...")
                for i in range(0, len(formatted_output), 4000):
                    update.message.reply_text(formatted_output[i:i+4000])
            logger.info(f"Transcription sent to user {user.id}")
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            processing_message.edit_text(f"Error during transcription: {str(e)}")
        os.unlink(temp_path)
        if wav_path != temp_path and os.path.exists(wav_path):
            os.unlink(wav_path)
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        processing_message.edit_text(f"Error: {str(e)}")

def error_handler(update, context):
    """Log errors from updates."""
    logger.error(f"Update {update} caused error {context.error}")

def setup_telegram_bot():
    """Initialize and start the Telegram bot in a background thread."""
    try:
        print(f"Setting up Telegram bot with token: {TELEGRAM_TOKEN[:5]}...{TELEGRAM_TOKEN[-5:]}")
        # IMPORTANT: Pass the token using a keyword argument.
        updater = Updater(token=TELEGRAM_TOKEN, use_context=True)
        dispatcher = updater.dispatcher
        dispatcher.add_handler(CommandHandler("start", start))
        dispatcher.add_handler(CommandHandler("help", help_command))
        dispatcher.add_handler(CommandHandler("language", set_language))
        dispatcher.add_handler(CommandHandler("info", info))
        audio_filter = Filters.audio | Filters.voice | Filters.document.audio | Filters.document.category("audio")
        dispatcher.add_handler(MessageHandler(audio_filter, process_audio))
        dispatcher.add_error_handler(error_handler)
        updater.start_polling()
        print("Telegram bot started successfully!")
        # Do not call updater.idle() here to keep the thread non-blocking.
    except Exception as e:
        print(f"Error starting Telegram bot: {e}")
        import traceback
        traceback.print_exc()

def main():
    # Start heavy tasks (Telegram bot, model loading) in background threads.
    threading.Thread(target=setup_telegram_bot, daemon=True).start()
    threading.Thread(target=setup_model, daemon=True).start()
    
    # Start Flask server in main thread; bind to the port provided by $PORT (default 8080).
    port = int(os.environ.get('PORT', 8080))
    print(f"Starting Flask server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)

if __name__ == '__main__':
    main()
