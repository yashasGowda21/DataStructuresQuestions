import os
import sys
import time
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from itertools import cycle

class ImageServer(BaseHTTPRequestHandler):
    image_cycle = None
    current_image = None
    lock = threading.Lock()

    @classmethod
    def set_image_folder(cls, folder_path):
        """Initialize the image folder and create a cycle of images."""
        abs_folder_path = os.path.abspath(folder_path)
        
        if not os.path.isdir(abs_folder_path):
            print(f"❌ Error: '{abs_folder_path}' is NOT a valid directory.")
            sys.exit(1)
        
        images = [os.path.join(abs_folder_path, f) for f in sorted(os.listdir(abs_folder_path))
                  if f.lower().endswith(('jpg', 'jpeg', 'png'))]

        if not images:
            print(f"❌ Error: No images found in '{abs_folder_path}'.")
            sys.exit(1)
        
        cls.image_cycle = cycle(images)
        cls.current_image = next(cls.image_cycle)
        print(f"✅ Serving first image: {cls.current_image}")

        # Start background thread for cycling images
        threading.Thread(target=cls.update_image_periodically, daemon=True).start()
    
    @classmethod
    def update_image_periodically(cls):
        """Rotate to the next image every 5 seconds in the background."""
        while True:
            time.sleep(5)  # Wait 5 seconds before switching
            with cls.lock:
                cls.current_image = next(cls.image_cycle)
                print(f"🔄 Now serving: {cls.current_image}")

    def do_GET(self):
        """Handle HTTP GET requests for images."""
        try:
            if self.path in ['/image.jpg', '/image.jpeg', '/image.png']:
                with self.lock:
                    image_path = ImageServer.current_image
                
                with open(image_path, 'rb') as file:
                    image_data = file.read()
                
                mime_type, _ = mimetypes.guess_type(image_path)
                self.send_response(200)
                self.send_header('Content-type', mime_type if mime_type else 'application/octet-stream')
                self.end_headers()
                self.wfile.write(image_data)

            elif self.path == '/favicon.ico':
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b'Favicon not found')
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b'Not found')
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b'Error fetching image')
            print(f"⚠️ Error: {e}")

def run(server_class=HTTPServer, handler_class=ImageServer, port=8000):
    """Start the HTTP server."""
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f"🚀 Server started on port {port}")
    httpd.serve_forever()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <image_folder_path>")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    print(f"📂 Using folder path: {folder_path}")

    ImageServer.set_image_folder(folder_path)
    run()
