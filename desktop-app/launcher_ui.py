#!/usr/bin/env python3
"""Lightweight HTTP server that serves the Thalamus UI and proxies API calls."""

import http.server
import json
import os
import webbrowser
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

THALAMUS_PORT = int(os.environ.get("THALAMUS_PORT", "3013"))
UI_PORT = int(os.environ.get("UI_PORT", "3014"))
API = f"http://127.0.0.1:{THALAMUS_PORT}"

HTML_PATH = Path(__file__).parent / "index.html"


def proxy_get(path, timeout=10):
    try:
        r = urlopen(f"{API}{path}", timeout=timeout)
        return r.status, r.read()
    except HTTPError as e:
        # Preserve API error body (e.g. upstream Cursor failure) instead of losing it to str(e).
        return e.code, e.read()
    except Exception as e:
        return 502, json.dumps({"error": str(e)}).encode()


def proxy_post(path, body=b"", timeout=60, extra_headers=None):
    try:
        headers = {"Content-Type": "application/json"}
        if extra_headers:
            headers.update(extra_headers)
        req = Request(f"{API}{path}", data=body, method="POST", headers=headers)
        r = urlopen(req, timeout=timeout)
        return r.status, r.read()
    except HTTPError as e:
        return e.code, e.read()
    except Exception as e:
        return 502, json.dumps({"error": str(e)}).encode()


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, code, body):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body if isinstance(body, bytes) else body.encode())

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PATH.read_bytes())

        elif self.path == "/api/health":
            c, b = proxy_get("/health")
            self._json(c, b)

        elif self.path == "/api/token_status":
            c, b = proxy_get("/token/status")
            self._json(c, b)

        elif self.path == "/api/login":
            c, b = proxy_get("/cursor/login")
            self._json(c, b)
            try:
                d = json.loads(b)
                if d.get("url"):
                    webbrowser.open(d["url"])
            except Exception:
                pass

        elif self.path.startswith("/api/poll"):
            qs = self.path.split("?", 1)[1] if "?" in self.path else ""
            c, b = proxy_get(f"/cursor/poll?{qs}")
            self._json(c, b)

        elif self.path == "/api/models":
            c, b = proxy_get("/v1/models", timeout=15)
            self._json(c, b)

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""

        if self.path == "/api/clear":
            c, b = proxy_post("/token/clear")
            self._json(c, b)

        elif self.path == "/api/messages":
            c, b = proxy_post(
                "/v1/messages", body=body, timeout=60,
                extra_headers={"anthropic-version": "2023-06-01"},
            )
            self._json(c, b)

        else:
            self.send_response(404)
            self.end_headers()


if __name__ == "__main__":
    srv = http.server.HTTPServer(("127.0.0.1", UI_PORT), Handler)
    print(f"UI server on http://127.0.0.1:{UI_PORT}")
    srv.serve_forever()
