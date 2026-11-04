"""Local caching relay for OSM standard tiles (main decision 2026-10-30, option B).

Only tiles the screens actually request are fetched, then kept in data/processed/tile_cache/.
A response identical to a known placeholder ("Access blocked") is dropped and refetched at most twice
with a pause; after that a plain gray tile is served and the block is logged. No bulk prefetching.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import logging
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CACHE = ROOT / "data/processed/tile_cache"
LOG = ROOT / "artifacts/logs/tile_proxy.log"
UPSTREAM = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
USER_AGENT = "YuseongParkingDemo/0.1 (student competition prototype)"
RETRY_PAUSE_S = (2.0, 5.0)          # at most two refetches
UPSTREAM_SLOTS = threading.BoundedSemaphore(2)
PLACEHOLDERS = {hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE / "reference_tiles").glob("*.png")}
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()
_counts = {"cache": 0, "fetched": 0, "refetch": 0, "blocked_final": 0, "error": 0}


def _gray_tile():
    buffer = io.BytesIO()
    Image.new("RGB", (256, 256), "#e9ecef").save(buffer, "PNG")
    return buffer.getvalue()


GRAY = _gray_tile()


def _lock(key):
    with _locks_guard:
        return _locks.setdefault(key, threading.Lock())


def _fetch(z, x, y):
    request = urllib.request.Request(UPSTREAM.format(z=z, x=x, y=y), headers={"User-Agent": USER_AGENT})
    with UPSTREAM_SLOTS, urllib.request.urlopen(request, timeout=15) as response:
        return response.status, response.read()


def tile(z, x, y):
    """Return (png bytes, source) for one tile; source is cache, fetched or gray."""
    path = CACHE / str(z) / str(x) / f"{y}.png"
    if path.exists():
        _counts["cache"] += 1
        return path.read_bytes(), "cache"
    with _lock(f"{z}/{x}/{y}"):
        if path.exists():
            _counts["cache"] += 1
            return path.read_bytes(), "cache"
        for attempt in range(len(RETRY_PAUSE_S) + 1):
            try:
                status, body = _fetch(z, x, y)
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                _counts["error"] += 1
                logging.warning("error %s/%s/%s attempt=%d %s", z, x, y, attempt, error)
                return GRAY, "gray"
            blocked = hashlib.sha256(body).hexdigest() in PLACEHOLDERS
            if status == 200 and not blocked:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".part")
                tmp.write_bytes(body)
                tmp.replace(path)
                _counts["fetched"] += 1
                logging.info("fetched %s/%s/%s attempt=%d bytes=%d", z, x, y, attempt, len(body))
                return body, "fetched"
            logging.warning("blocked %s/%s/%s attempt=%d status=%d bytes=%d", z, x, y, attempt, status, len(body))
            if attempt < len(RETRY_PAUSE_S):
                _counts["refetch"] += 1
                time.sleep(RETRY_PAUSE_S[attempt])
        _counts["blocked_final"] += 1
        logging.warning("gray %s/%s/%s after %d refetches; counts=%s", z, x, y, len(RETRY_PAUSE_S), _counts)
        return GRAY, "gray"


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 - http.server API
        parts = self.path.split("?")[0].strip("/").split("/")
        try:
            if len(parts) != 4 or parts[0] != "tiles" or not parts[3].endswith(".png"):
                raise ValueError
            z, x, y = int(parts[1]), int(parts[2]), int(parts[3][:-4])
            if not (0 <= z <= 19 and 0 <= x < 2 ** z and 0 <= y < 2 ** z):
                raise ValueError
        except ValueError:
            self.send_error(404)
            return
        body, source = tile(z, x, y)
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store" if source == "gray" else "public, max-age=604800")
        self.send_header("X-Tile-Source", source)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # noqa: A002 - quiet access log; fetch outcomes are logged in tile()
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8503)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=LOG, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info("start port=%d placeholders=%d cache=%s", args.port, len(PLACEHOLDERS), CACHE)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
