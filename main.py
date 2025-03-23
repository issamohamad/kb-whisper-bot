import os
from http.server import HTTPServer, BaseHTTPRequestHandler

# Simple HTTP server 
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(b'KB-Whisper Bot is online!')

def main():
    # Print confirmation that the script is running
    print("Starting minimal HTTP server...")
    
    # Get port from environment variable
    port = int(os.environ.get('PORT', '8080'))
    print(f"Using port: {port}")
    
    # Start HTTP server
    httpd = HTTPServer(('0.0.0.0', port), Handler)
    print(f"Server running at 0.0.0.0:{port}")
    httpd.serve_forever()

if __name__ == '__main__':
    main()
