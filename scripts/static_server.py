#!/usr/bin/env python3
"""Simple HTTP server for hospital HTML apps."""
import http.server, socketserver, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 项目根（scripts.config 包导入）
from scripts.config import PROJECT_ROOT, SERVE_PORT

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else int(SERVE_PORT)
DIR = str(PROJECT_ROOT / "html")

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIR, **kwargs)

with socketserver.TCPServer(("0.0.0.0", PORT), Handler) as httpd:
    print(f"[医院] Hospital Apps at http://localhost:{PORT}")
    httpd.serve_forever()
