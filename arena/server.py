#!/usr/bin/env python3
"""Serve the Fly x Jev arena and proxy decisions to TypeSafe's Jev.

The API key stays server-side. Connections to api.typesafe.ai are kept alive
in a small pool so each decision skips the TLS handshake.

Key lookup: the TYPESAFE_API_KEY environment variable, else a TYPESAFE_API_KEY=
line in the repo-root .env file (one level above this directory, so it is never
inside the served static root).
"""

from __future__ import annotations

import http.client
import json
import mimetypes
import os
import queue
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
ENV_FILE = ROOT.parent / ".env"
JEV_HOST = "api.typesafe.ai"
JEV_PATH = "/v1/systemone"

mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("application/octet-stream", ".f32")
mimetypes.add_type("application/octet-stream", ".u8")


def read_env_file() -> dict[str, str]:
    """Parse simple KEY=value lines from ENV_FILE (comments and blanks ignored)."""
    values: dict[str, str] = {}
    if not ENV_FILE.is_file():
        return values
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip().removeprefix("export ").strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        values[name.strip()] = value.strip().strip("'\"")
    return values


ENV = read_env_file()


def setting(name: str, default: str | None = None) -> str | None:
    """Environment variable first, then the .env file, then the default."""
    return os.environ.get(name) or ENV.get(name) or default


KEY = setting("TYPESAFE_API_KEY")
JEV_MODEL = setting("JEV_MODEL", "jev-latest")
PORT = int(setting("PORT", "8777"))
POOL: queue.LifoQueue[http.client.HTTPSConnection] = queue.LifoQueue()


def jev_request(payload: dict) -> tuple[int, dict]:
    body = json.dumps({"model": JEV_MODEL, **payload})
    headers = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
    for attempt in range(2):
        try:
            conn = POOL.get_nowait()
        except queue.Empty:
            conn = http.client.HTTPSConnection(JEV_HOST, timeout=15)
        try:
            conn.request("POST", JEV_PATH, body=body, headers=headers)
            resp = conn.getresponse()
            data = resp.read()
            POOL.put(conn)
            return resp.status, json.loads(data or b"{}")
        except (http.client.HTTPException, OSError):
            conn.close()
            if attempt == 1:
                raise
    raise RuntimeError("unreachable")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def send(self, status: int, payload: bytes, content_type: str):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def send_json(self, status: int, value):
        self.send(status, json.dumps(value).encode(), "application/json")

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/health":
            return self.send_json(200, {"jev_key": bool(KEY), "model": JEV_MODEL})
        rel = "index.html" if path in ("", "/") else path.lstrip("/")
        target = (ROOT / rel).resolve()
        if ROOT not in target.parents or not target.is_file() or target.suffix == ".py" or target.name.startswith("."):
            return self.send(404, b"not found", "text/plain")
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self.send(200, target.read_bytes(), ctype)

    def do_POST(self):
        if urlparse(self.path).path != "/api/decide":
            return self.send(404, b"not found", "text/plain")
        if not KEY:
            return self.send_json(503, {"error": "No TypeSafe API key. Set TYPESAFE_API_KEY in the environment or in .env at the repo root (see .env.example), then restart the server."})
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self.send_json(400, {"error": "invalid JSON"})
        started = time.perf_counter()
        try:
            status, data = jev_request({"state": payload["state"], "questions": payload["questions"]})
        except Exception as exc:  # network failure: report, let the client keep its last action
            return self.send_json(502, {"error": str(exc)})
        data["latency_ms"] = round((time.perf_counter() - started) * 1000)
        self.send_json(status, data)


if __name__ == "__main__":
    print(f"Fly x Jev arena → http://127.0.0.1:{PORT}  (jev key: {'yes' if KEY else 'MISSING'})")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
