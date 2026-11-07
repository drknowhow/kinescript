"""What kine edit and kine brand wizard share as local web servers: 127.0.0.1 only, a Host check against DNS
rebinding, JSON-only POST bodies of a bounded size, and errors as JSON the page can show."""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler

MAX_BODY = 1 << 20
LOCAL_HOSTS = {"localhost", "127.0.0.1", "[::1]"}


class BadRequest(ValueError):
    pass


def host_name(header: str) -> str:
    if header.startswith("["):
        return header.split("]")[0] + "]"
    return header.rsplit(":", 1)[0] if ":" in header else header


class LocalHandler(BaseHTTPRequestHandler):
    """A handler whose GET and POST run through _guard: foreign hosts get 403, `client_errors` 400, a missing file
    404 and anything else 500, each as {"error": ...}. Subclasses define _get and _post."""
    client_errors: tuple = (BadRequest,)
    max_body = MAX_BODY

    def log_message(self, *a):
        pass

    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json")

    def _gone(self, e: FileNotFoundError) -> str:
        return f"not found: {e.filename or e}"

    def _guard(self, fn):
        if host_name(self.headers.get("Host") or "") not in LOCAL_HOSTS:  # blocks DNS rebinding
            return self._json({"error": "bad host"}, 403)
        try:
            fn()
        except self.client_errors as e:
            self._json({"error": str(e)}, 400)
        except FileNotFoundError as e:
            self._json({"error": self._gone(e)}, 404)
        except Exception as e:  # the file may be mid-edit; the page shows the message
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def do_GET(self):
        self._guard(self._get)

    def do_POST(self):
        self._guard(self._post)

    def _get(self):
        self._json({"error": "not found"}, 404)

    def _post(self):
        self._json({"error": "not found"}, 404)

    def _body(self) -> dict:
        """The POST body as a JSON object, refused when empty, too large, not JSON or not an object."""
        n = int(self.headers.get("Content-Length") or 0)
        if not 0 < n <= self.max_body:
            self.close_connection = True
            raise BadRequest("body is empty or too large")
        data = self.rfile.read(n)  # read before refusing, so the client is not reset mid-send
        if "application/json" not in (self.headers.get("Content-Type") or ""):
            raise BadRequest("send application/json")
        try:
            body = json.loads(data)
        except json.JSONDecodeError:
            raise BadRequest("body is not valid JSON") from None
        if not isinstance(body, dict):
            raise BadRequest("body must be an object")
        return body
