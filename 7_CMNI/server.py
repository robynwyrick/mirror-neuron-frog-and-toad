#!/usr/bin/env python
import os
import http.server
import socketserver

out_dir = "./"

port = 8888
print(f"\n▶ Serving `{out_dir}/` at http://0.0.0.0:{port}/")
print("  • If you’re SSH’ing from your Mac, on your Mac run:")
print("      ssh -L 8888:localhost:8888 robyn@mountain")
print("    then open http://localhost:8888 in your browser.")
os.chdir(out_dir)
handler = http.server.SimpleHTTPRequestHandler
with socketserver.TCPServer(("", port), handler) as httpd:
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 HTTP server stopped.")

