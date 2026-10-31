"""Loopback-only release service shared by updater integration and binary tests."""

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class ReleaseServer:
    """Serve GitHub-shaped release metadata and an archive without GitHub access."""

    def __init__(self, archive, asset_name, build=8):
        """Create a release fixture, bound only to the loopback interface.

        Args: archive is the archive bytes; asset_name is its release filename;
            build is the offered build number.
        Returns: None.
        """
        self.archive = archive
        self.asset_name = asset_name
        self.build = build
        self.failure = None
        self.requests = []
        self.download_started = threading.Event()
        self.allow_download = threading.Event()
        self.allow_download.set()
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            """Translate requests into this fixture's release responses."""

            def do_GET(self):
                """Serve one metadata, redirect or archive request.

                Args: None.
                Returns: None.
                """
                fixture.requests.append((self.path, self.headers.get("Authorization")))
                status, payload = 200, b""
                if self.path == "/publish":
                    fixture.build = 8
                    payload = b"published"
                elif self.path == "/repos/test/espi-live/releases/latest":
                    if fixture.failure == "http":
                        status, payload = 503, b"unavailable"
                    else:
                        payload = json.dumps({"assets": [
                            {"name": "build-info.json", "url": fixture.url + "/manifest"},
                            {"name": fixture.asset_name, "url": fixture.url + "/asset",
                             "size": len(fixture.archive)},
                        ]}).encode()
                elif self.path == "/manifest":
                    payload = json.dumps({
                        "build": fixture.build, "commit": "abcdef1234567",
                        "archives": {fixture.asset_name:
                                     hashlib.sha256(fixture.archive).hexdigest()},
                    }).encode()
                elif self.path == "/asset":
                    self.send_response(302)
                    self.send_header("Location", fixture.url + "/download")
                    self.end_headers()
                    return
                elif self.path == "/download":
                    fixture.download_started.set()
                    if not fixture.allow_download.wait(10):
                        status, payload = 504, b"test download was not released"
                    elif fixture.failure == "corrupt":
                        payload = b"corrupt archive"
                    elif fixture.failure == "interrupted":
                        payload = fixture.archive[:len(fixture.archive) // 2]
                    else:
                        payload = fixture.archive
                else:
                    status, payload = 404, b"not found"
                self.send_response(status)
                length = (len(fixture.archive) if self.path == "/download"
                          and fixture.failure == "interrupted" else len(payload))
                self.send_header("Content-Length", str(length))
                self.end_headers()
                try:
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # A failed test may close its client before teardown.

            def log_message(self, format, *args):
                """Keep HTTP access logs out of unittest output.

                Args: format and args are the HTTP server's log fields.
                Returns: None.
                """

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       kwargs={"poll_interval": 0.01}, daemon=True)

    def __enter__(self):
        """Start serving fixture requests.

        Args: None.
        Returns: This fixture.
        """
        self.thread.start()
        return self

    def __exit__(self, *exc):
        """Release downloads and stop the server and its handlers.

        Args: exc is the context manager's exception information.
        Returns: None.
        """
        self.allow_download.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)
