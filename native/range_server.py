#!/usr/bin/env python3
"""Localhost Range server. Stands in for an S3 or Telegram-S3 gateway."""

import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = sys.argv[1]
PORT = int(sys.argv[2])


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        name = self.path.split("?", 1)[0].lstrip("/")
        path = os.path.join(ROOT, name)
        if not os.path.isfile(path):
            self.send_error(404)
            return
        with open(path, "rb") as fh:
            data = fh.read()
        start, end = 0, len(data) - 1
        status = 200
        rng = self.headers.get("Range")
        if rng:
            if not rng.startswith("bytes=") or "," in rng:
                self.send_error(416)
                return
            spec = rng[6:]
            left, _, right = spec.partition("-")
            start = int(left) if left else 0
            end = int(right) if right else len(data) - 1
            if start < 0 or end >= len(data) or start > end:
                self.send_error(416)
                return
            status = 206
        body = data[start : end + 1]
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Accept-Ranges", "bytes")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        return


ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
