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

# Hardcoded token (this is explicit for troubleshooting)
TELEGRAM_TOKEN = "7642881098:AAFGbpNoK8vjo3dnv7UVI4_KvVRGV3zH9jc"
print(f"Token being used: {TELEGRAM_TOKEN[:5]}...{TELEGRAM_TOKEN[-5:]}")

# Simple HTTP server to satisfy Cloud Run requirements
def start_http_server():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'KB-Whisper Bot is running!')
    
    port = int(os.environ.get('PORT', '8080'))
    httpd = HTTPServer(('0.0.0.0', port), Handler)
    print(f"Starting HTTP server on port {port}")
    httpd.serve_forever()

def main():
    # First, print confirmation that the script is running
    print("Script is starting...")
    
    # Start HTTP server in a separate thread
    http_thread = threading.Thread(target=start_http_server, daemon=True)
    http_thread.start()
    print("HTTP server thread started")
    
    try:
        # Create the Updater with the token
        print("Creating Telegram updater...")
        updater = Updater(token=TELEGRAM_TOKEN)
        print("Updater created successfully")
        
        # For testing purposes only - just keep the HTTP server running
        print("Bot is now running!")
        while True:
            import time
            time.sleep(60)
            print("Still alive...")
    
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
