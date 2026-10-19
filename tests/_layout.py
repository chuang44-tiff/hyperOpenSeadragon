"""Layout helper shared by the public test subset.

These files are copied byte-for-byte between the private development repo
(``tests/public/``, sources under ``dziGeneration/``) and the public repo
(``tests/``, sources at the repo root). Everything that differs between the two
layouts is resolved here.
"""
import contextlib
import functools
import http.server
import os
import socket
import sys
import threading
import unittest
from pathlib import Path

REQUIRE_ALL = os.environ.get("HYPEROSD_REQUIRE_ALL") == "1"


def _find_root():
    here = Path(__file__).resolve().parent
    for cand in (here, *here.parents):
        if (cand / "zstackHyper.html").is_file():
            return cand
    raise RuntimeError("repo root (a directory holding zstackHyper.html) not found above %s" % here)


ROOT = _find_root()
GEN_DIR = ROOT / "dziGeneration" if (ROOT / "dziGeneration").is_dir() else ROOT
TEMPLATE = GEN_DIR / "templates" / "zstackHyper.html"
HOSTS = [ROOT / "zstackHyper.html", TEMPLATE]
DEMO_DATA = ROOT / "demo_data"

if str(GEN_DIR) not in sys.path:
    sys.path.insert(0, str(GEN_DIR))


def require_deps(reason):
    """Call from setUpClass with the reason a dependency is unusable (or None).

    A missing dependency skips, unless HYPEROSD_REQUIRE_ALL=1, where it fails:
    CI must never go green by silently skipping everything.
    """
    if not reason:
        return
    if REQUIRE_ALL:
        raise AssertionError("HYPEROSD_REQUIRE_ALL=1 but a dependency is unusable: %s" % reason)
    raise unittest.SkipTest(reason)


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


@contextlib.contextmanager
def static_server(directory=None):
    """Serve *directory* (default: the repo root) on a free localhost port, in-process.

    Yields the base URL (no trailing slash). The server thread is joined and the
    socket closed on exit, so ``unittest discover`` leaks nothing.
    """
    handler = functools.partial(_QuietHandler, directory=str(directory or ROOT))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    thread.start()
    try:
        yield "http://127.0.0.1:%d" % server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
