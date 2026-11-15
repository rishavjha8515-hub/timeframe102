import hashlib
import tempfile
import unittest
from pathlib import Path

from pipeline.fetch_cutouts import CutoutDownloadFailure, _download


class FakeResponse:
    def __init__(self, status, body=b"", *, headers=None, chunks=None):
        self.status_code = status
        self._body = body
        self.headers = headers or {}
        self._chunks = chunks if chunks is not None else [body]
        self.closed = False

    @property
    def text(self):
        return self._body.decode("utf-8", errors="replace")

    def iter_content(self, chunk_size):
        yield from self._chunks

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return next(self.responses)


class VictoriaDownloadTests(unittest.TestCase):
    def test_retries_503_then_gets_directly_without_head(self):
        busy = FakeResponse(503, b"Server busy", headers={"Retry-After": "0"})
        good = FakeResponse(200, b"fits", headers={"Content-Length": "4"})
        session = FakeSession([busy, good])
        waits = []
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "frame.fits"
            result = _download(session, "https://irsa.ipac.caltech.edu/test", destination,
                               max_bytes=100, sleep=waits.append)
            self.assertEqual(destination.read_bytes(), b"fits")
        self.assertEqual(result, (4, hashlib.sha256(b"fits").hexdigest(), 200, 2))
        self.assertEqual(waits, [5.0])
        self.assertTrue(all(call[1]["stream"] is True for call in session.calls))

    def test_oversize_content_length_fails_before_reading_body(self):
        response = FakeResponse(200, b"too large", headers={"Content-Length": "101"})
        session = FakeSession([response])
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "frame.fits"
            with self.assertRaises(CutoutDownloadFailure) as context:
                _download(session, "https://irsa.ipac.caltech.edu/test", destination,
                          max_bytes=100, sleep=lambda _: None)
            self.assertFalse(destination.exists())
        self.assertEqual(context.exception.http_status, 200)
        self.assertIn("exceeds", str(context.exception))

    def test_nonretryable_http_error_preserves_status_and_body(self):
        response = FakeResponse(404, b"missing")
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(CutoutDownloadFailure) as context:
                _download(FakeSession([response]), "https://irsa.ipac.caltech.edu/test",
                          Path(temp) / "frame.fits", sleep=lambda _: None)
        self.assertEqual(context.exception.http_status, 404)
        self.assertEqual(context.exception.response_body, "missing")
        self.assertEqual(context.exception.attempts, 1)


if __name__ == "__main__":
    unittest.main()
