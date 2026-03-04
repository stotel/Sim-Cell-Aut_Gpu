#!/usr/bin/env python3
import http.server
import socketserver
import os

# Change to the project root directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

PORT = 8000
Handler = http.server.SimpleHTTPRequestHandler

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Server running at http://localhost:{PORT}/web-interface/main.html")
    print(f"Press Ctrl+C to stop the server")
    httpd.serve_forever()
