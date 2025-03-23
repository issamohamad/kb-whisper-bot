import os
import logging
import tempfile
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import torch
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pydub import AudioSegment

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Hardcode your token (only for troubleshooting - normally use environment variables)
TELEGRAM_TOKEN = "7642881098:AAFGbpNoK8vjo3dnv7UVI4_KvVRGV3zH9jc"

# Global variables for the transcription model
transcription_pipe = None
device = "cpu"
torch_dtype = torch.float32

# Simple HTTP server to satisfy Cloud Run requirements
def start_http_server():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'KB-Whisper Bot is running!')
    
    httpd = HTTPServer(('', 8080), Handler)
    print("Starting HTTP server on port 8080")
    httpd.serve_forever()

def setup_model():
    """Setup the transcription model at startup."""
    global transcription_pipe, device, torch_dtype
    
    if transcription_pipe is not None:
        return
    
    # Determine the device to use
    if torch.cuda.is_available():
        device = "cuda:0"
        torch_dtype = torch.float16
    
    logger.info(f"Setting up KB-Whisper model on {device}...")
    
    # Load model
    model_id = "KBLab/kb-whisper-small"  # Use small model for faster inference
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, use_safetensors=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    
    transcription_pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device if device != "cpu" else -1,
    )
    
    logger.info("KB-Whisper model loaded successfully!")

def start(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    update.message.reply_text(
        f"Hi {user.first_name}! I'm a transcription bot using KB-Whisper.\n"
        f"Send me an audio message or file, and I'll transcribe it with timestamps.\n\n"
        f"Supported languages: Swedish (default), English, Norwegian, Danish, Finnish\n\n"
        f"Use /language [code] to change language (sv, en, no, da, fi)"
    )
    logger.info(f"User {user.id} ({user.first_name}) started the bot")

def help_command(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /help is issued."""
    update.message.reply_text(
        "Send me an audio message or file to transcribe it.\n\n"
        "Commands:\n"
        "/start - Start the bot\n"
        "/help - Show this help message\n"
        "/language [code] - Change language (sv, en, no, da, fi)\n"
        "/info - Show bot information"
    )

def set_language(update: Update, context: CallbackContext) -> None:
    """Set the language for transcription."""
    if not context.args:
        update.message.reply_text(
            "Please specify a language code: sv, en, no, da, fi\n"
            "Example: /language en"
        )
        return
    
    language = context.args[0].lower()
    valid_languages = {"sv", "en", "no", "da", "fi"}
    
    if language not in valid_languages:
        update.message.reply_text(
            f"Invalid language code. Please use one of: {', '.join(valid_languages)}"
        )
        return
    
    # Store in user_data to remember for this user
    context.user_data["language"] = language
    
    language_names = {
        "sv": "Swedish",
        "en": "English",
        "no": "Norwegian",
        "da": "Danish",
        "fi": "Finnish"
    }
    
    update.message.reply_text(
        f"Language set to {language_names[language]} ({language})."
    )
    logger.info(f"User {update.effective_user.id} changed language to {language}")

def info(update: Update, context: CallbackContext) -> None:
    """Show information about the bot."""
    global device
    
    user_language = context.user_data.get("language", "sv")
    language_names = {
        "sv": "Swedish",
        "en": "English",
        "no": "Norwegian",
        "da": "Danish",
        "fi": "Finnish"
    }
    
    update.message.reply_text(
        f"KB-Whisper Transcription Bot\n\n"
        f"Model: KBLab/kb-whisper-small\n"
        f"Device: {device}\n"
        f"Current language: {language_names[user_language]} ({user_language})\n\n"
        f"Created with ❤️ using KB-Whisper from KBLab"
    )

def process_audio(update: Update, context: CallbackContext) -> None:
    """Process audio messages and files."""
    # Check if model is loaded
    if transcription_pipe is None:
        update.message.reply_text("I'm still loading the transcription model. Please try again in a moment.")
        return
    
    # Get user language preference or default to Swedish
    language = context.user_data.get("language", "sv")
    
    # Tell the user we're processing
    processing_message = update.message.reply_text(
        f"Processing your audio in {language} language...\n"
        f"This may take a while depending on the audio length."
    )
    
    user = update.effective_user
    logger.info(f"User {user.id} ({user.first_name}) sent an audio file")
    
    # Get the audio file
    try:
        if update.message.voice:
            file = update.message.voice.get_file()
            file_extension = ".ogg"  # Voice messages are usually in OGG format
            duration = update.message.voice.duration
            file_info = "voice message"
        elif update.message.audio:
            file = update.message.audio.get_file()
            file_extension = os.path.splitext(update.message.audio.file_name)[1]
            duration = update.message.audio.duration
            file_info = update.message.audio.file_name
        else:
            # For documents (other file types)
            file = update.message.document.get_file()
            file_extension = os.path.splitext(update.message.document.file_name)[1]
            duration = 0  # Unknown duration for documents
            file_info = update.message.document.file_name
        
        logger.info(f"Received file: {file_info}")
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
            temp_path = temp_file.name
        
        # Download the file
        file.download(temp_path)
        
        # Process audio to WAV if needed
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
        
        # Estimate processing time
        processing_time = (duration or os.path.getsize(wav_path) / 100000) * (1 if device == "cuda:0" else 5)
        
        # Update the processing message with estimated time
        processing_message.edit_text(
            f"Processing your audio in {language} language...\n"
            f"Estimated time: {int(processing_time)} seconds"
        )
        
        # Transcribe the audio
        try:
            result = transcription_pipe(
                wav_path,
                chunk_length_s=30,
                stride_length_s=3,
                return_timestamps=True,
                generate_kwargs={
                    "task": "transcribe",
                    "language": language,
                    "return_timestamps": True
                }
            )
            
            # Format the output with timestamps
            formatted_output = ""
            if "chunks" in result:
                for chunk in result["chunks"]:
                    minutes = int(chunk["timestamp"][0]) // 60
                    seconds = int(chunk["timestamp"][0]) % 60
                    timestamp = f"[{minutes:02d}:{seconds:02d}]"
                    formatted_output += f"{timestamp} {chunk['text']}\n\n"
            else:
                formatted_output = result["text"]
            
            # Split the message if it's too long for Telegram
            if len(formatted_output) <= 4000:
                processing_message.edit_text(formatted_output)
            else:
                processing_message.edit_text("Transcription complete! Sending in multiple messages due to length...")
                
                # Split and send in chunks of 4000 characters
                for i in range(0, len(formatted_output), 4000):
                    chunk = formatted_output[i:i+4000]
                    update.message.reply_text(chunk)
            
            logger.info(f"Transcription sent to user {user.id}")
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            processing_message.edit_text(f"Error during transcription: {str(e)}")
        
        # Clean up
        os.unlink(temp_path)
        if wav_path != temp_path and os.path.exists(wav_path):
            os.unlink(wav_path)
            
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        processing_message.edit_text(f"Error: {str(e)}")

def error_handler(update, context):
    """Handle errors."""
    logger.error(f"Update {update} caused error {context.error}")

def main() -> None:
    """Start the bot."""
    # Start HTTP server in a thread to satisfy Cloud Run
    http_thread = threading.Thread(target=start_http_server, daemon=True)
    http_thread.start()
    
    # Create the Updater and pass it the bot's token
    updater = Updater(TELEGRAM_TOKEN)
    
    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher
    
    # Add conversation handlers
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_command))
    dispatcher.add_handler(CommandHandler("language", set_language))
    dispatcher.add_handler(CommandHandler("info", info))
    
    # Add message handlers for audio files
    audio_filter = Filters.audio | Filters.voice | Filters.document.audio | Filters.document.category("audio")
    dispatcher.add_handler(MessageHandler(audio_filter, process_audio))
    
    # Add error handler
    dispatcher.add_error_handler(error_handler)
    
    # Setup the model before starting
    setup_model()
    
    # Start the Bot
    updater.start_polling()
    print("Bot started successfully!")
    
    # Run the bot until you press Ctrl-C
    updater.idle()

if __name__ == '__main__':
    main()
