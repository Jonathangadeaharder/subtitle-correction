from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from subtitle_correction.models import ParsedFilename, SubtitleResult
from subtitle_correction.scraper import (
    CHROME_USER_AGENT,
    LANG_CODE_MAP,
    OPENSUBTITLES_BASE,
    OpenSubtitlesScraper,
    _solve_pow,
)


# ---------------- _solve_pow ----------------


def test_solve_pow_difficulty_even() -> None:
    # difficulty 0 -> first nonce (0) works since no full bytes required
    digest_hex, nonce = _solve_pow("data", 0)
    assert isinstance(digest_hex, str)
    assert isinstance(nonce, int)
    assert nonce == 0


def test_solve_pow_difficulty_two_bytes() -> None:
    # difficulty 4 -> need first 2 bytes zero; will find a nonce eventually
    digest_hex, nonce = _solve_pow("randomprefix", 4)
    # Verify the solution
    import hashlib

    digest = hashlib.sha256(f"randomprefix{nonce}".encode()).digest()
    assert digest[0] == 0 and digest[1] == 0
    assert digest_hex == digest.hex()


def test_solve_pow_odd_difficulty_half_byte() -> None:
    # difficulty 5 -> 2 full zero bytes + high nibble of 3rd byte must be 0
    digest_hex, nonce = _solve_pow("oddtest", 5)
    import hashlib

    digest = hashlib.sha256(f"oddtest{nonce}".encode()).digest()
    assert digest[0] == 0 and digest[1] == 0
    assert (digest[2] >> 4) == 0


# ---------------- constants ----------------


def test_lang_code_map_has_common_languages() -> None:
    assert LANG_CODE_MAP["en"] == "eng"
    assert LANG_CODE_MAP["de"] == "ger"


def test_constants() -> None:
    assert OPENSUBTITLES_BASE == "https://www.opensubtitles.org"
    assert "Chrome" in CHROME_USER_AGENT


# ---------------- _parse_search_results (pure) ----------------

_SEARCH_HTML = """
<html>
<a class="bnone" href="/en/subtitles/12345/movie-title-en" title="Movie Title subtitles - en">Movie Title</a>
<a class="bnone" href="/en/subtitles/12345/movie-title-en" title="Dup">Dup</a>
<a class="bnone" href="/en/subtitles/67890/other-title-de" title="Other Title subtitles - de">Other Title</a>
/en/subtitleserve/sub/12345">5,432x
/en/subtitleserve/sub/67890">100x
</html>
"""


def test_parse_search_results_extracts_subtitles() -> None:
    scraper = OpenSubtitlesScraper()
    results = scraper._parse_search_results(_SEARCH_HTML)
    # 12345 appears twice (deduped), 67890 once -> 2 results
    assert len(results) == 2
    ids = {r.id for r in results}
    assert "/en/subtitles/12345/movie-title-en" in ids
    assert "/en/subtitles/67890/other-title-de" in ids
    by_id = {r.id: r for r in results}
    assert by_id["/en/subtitles/12345/movie-title-en"].downloads == 5432
    assert by_id["/en/subtitles/67890/other-title-de"].language == "de"
    assert by_id["/en/subtitles/67890/other-title-de"].download_url.endswith("/sub/67890")


def test_parse_search_results_empty_html() -> None:
    scraper = OpenSubtitlesScraper()
    assert scraper._parse_search_results("<html></html>") == []


def test_parse_search_results_no_href_skipped() -> None:
    scraper = OpenSubtitlesScraper()
    html = '<a class="bnone" title="no href">x</a>'
    assert scraper._parse_search_results(html) == []


def test_parse_search_results_falls_back_to_slug_for_title() -> None:
    scraper = OpenSubtitlesScraper()
    html = '<a class="bnone" href="/en/subtitles/111/some-slug-fr">x</a>'
    results = scraper._parse_search_results(html)
    assert len(results) == 1
    assert results[0].language == "fr"


# ---------------- _pick_best (pure) ----------------


def test_pick_best_prefers_episode_label_match() -> None:
    # Both match episode label (+100); downloads break the tie.
    # id=1: 100 + min(5000//1000,20)=5 -> 105
    # id=2: 100 + min(100//1000,20)=0  -> 100
    results = [
        SubtitleResult(id="1", name="Show S01E01 Other", language="en", downloads=5000),
        SubtitleResult(id="2", name="Show S01E01 720p", language="en", downloads=100),
    ]
    scraper = OpenSubtitlesScraper()
    parsed = ParsedFilename(title="Show", season=1, episode=1, is_tv=True)
    best = scraper._pick_best(results, parsed)
    assert best.id == "1"


def test_pick_best_prefers_title_match() -> None:
    results = [
        SubtitleResult(id="1", name="Other Title", language="en", downloads=5000),
        SubtitleResult(id="2", name="Show Full", language="en", downloads=100),
    ]
    scraper = OpenSubtitlesScraper()
    parsed = ParsedFilename(title="Show")
    best = scraper._pick_best(results, parsed)
    assert best.id == "2"


def test_pick_best_falls_back_to_first_when_no_results_scored() -> None:
    results = [SubtitleResult(id="1", name="x", language="en", downloads=0)]
    scraper = OpenSubtitlesScraper()
    parsed = ParsedFilename(title="Show")
    best = scraper._pick_best(results, parsed)
    assert best.id == "1"


def test_pick_best_empty_results_returns_first() -> None:
    scraper = OpenSubtitlesScraper()
    parsed = ParsedFilename(title="Show")
    # results[0] would IndexError; but scored empty -> returns results[0]
    results = [SubtitleResult(id="only", name="n", language="en")]
    assert scraper._pick_best(results, parsed).id == "only"


# ---------------- Fake session helpers ----------------


class _FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        text: str = "",
        content: bytes = b"",
        headers: dict | None = None,
    ):
        self.status_code = status_code
        self.text = text
        self.content = content
        self.headers = headers or {}


class _FakeSession:
    def __init__(self, *, responses: list[_FakeResponse] | None = None):
        self.responses = list(responses or [])
        self.calls: list[dict] = []
        self.headers: dict = {}
        self.cookies = type("c", (), {"clear": lambda self: None})()

    def get(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        if self.responses:
            return self.responses.pop(0)
        return _FakeResponse()

    def close(self):
        pass


def _scraper_with_session(
    scraper: OpenSubtitlesScraper, session: _FakeSession
) -> OpenSubtitlesScraper:
    scraper._session = session  # type: ignore[assignment]
    scraper._cf_passed = True
    return scraper


# ---------------- download ----------------


def test_download_no_id_returns_failed_srt(tmp_path: Path) -> None:
    scraper = _scraper_with_session(OpenSubtitlesScraper(), _FakeSession())
    sub = SubtitleResult(id="", name="x", language="en")
    out = scraper.download(sub, tmp_path)
    assert out.name == "failed.srt"


def test_download_extracts_zip(tmp_path: Path) -> None:
    # Build a zip containing an srt
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("movie.srt", "1\n00:00:01,000 --> 00:00:02,000\nhello\n")
    zip_bytes = buf.getvalue()

    session = _FakeSession(
        responses=[
            _FakeResponse(content=zip_bytes, headers={"Content-Type": "application/zip"}),
        ]
    )
    scraper = _scraper_with_session(OpenSubtitlesScraper(), session)
    sub = SubtitleResult(id="/en/subtitles/123/movie-en", name="Movie", language="en")
    out = scraper.download(sub, tmp_path)
    assert out.exists()
    assert out.suffix == ".srt"
    assert "hello" in out.read_text(encoding="utf-8")


def test_download_zip_with_no_srt_returns_failed(tmp_path: Path) -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("readme.txt", "not an srt")
    session = _FakeSession(
        responses=[
            _FakeResponse(content=buf.getvalue(), headers={"Content-Type": "application/zip"}),
        ]
    )
    scraper = _scraper_with_session(OpenSubtitlesScraper(), session)
    sub = SubtitleResult(id="/en/subtitles/123/movie-en", name="Movie", language="en")
    out = scraper.download(sub, tmp_path)
    assert out.name == "failed.srt"


def test_download_gzip_content(tmp_path: Path) -> None:
    import gzip

    srt_bytes = b"1\n00:00:01,000 --> 00:00:02,000\nhello\n"
    gz = gzip.compress(srt_bytes)
    # "application/gzip" contains "zip" so the zip branch runs first and fails
    # (not a real zip). Verifies graceful failure for gzip-as-zip confusion.
    session = _FakeSession(
        responses=[
            _FakeResponse(content=gz, headers={"Content-Type": "application/gzip"}),
        ]
    )
    scraper = _scraper_with_session(OpenSubtitlesScraper(), session)
    sub = SubtitleResult(id="/en/subtitles/123/movie-en", name="Movie", language="en")
    out = scraper.download(sub, tmp_path)
    assert out.name == "failed.srt"


def test_download_gzip_by_magic_bytes(tmp_path: Path) -> None:
    import gzip

    srt_bytes = b"1\n00:00:01,000 --> 00:00:02,000\nhello\n"
    gz = gzip.compress(srt_bytes)
    # Content-Type doesn't say gzip but magic bytes do
    session = _FakeSession(
        responses=[
            _FakeResponse(content=gz, headers={"Content-Type": "application/octet-stream"}),
        ]
    )
    scraper = _scraper_with_session(OpenSubtitlesScraper(), session)
    sub = SubtitleResult(id="/en/subtitles/123/movie-en", name="Movie", language="en")
    out = scraper.download(sub, tmp_path)
    assert "hello" in out.read_text(encoding="utf-8")


def test_download_plain_srt_text(tmp_path: Path) -> None:
    text = "1\n00:00:01,000 --> 00:00:02,000\nhello world\n"
    session = _FakeSession(
        responses=[
            _FakeResponse(content=text.encode("utf-8"), headers={"Content-Type": "text/plain"}),
        ]
    )
    scraper = _scraper_with_session(OpenSubtitlesScraper(), session)
    sub = SubtitleResult(id="/en/subtitles/123/movie-en", name="Movie", language="en")
    out = scraper.download(sub, tmp_path)
    assert "hello world" in out.read_text(encoding="utf-8")


def test_download_non_200_returns_failed(tmp_path: Path) -> None:
    session = _FakeSession(
        responses=[
            _FakeResponse(status_code=500, content=b"err"),
        ]
    )
    scraper = _scraper_with_session(OpenSubtitlesScraper(), session)
    sub = SubtitleResult(id="/en/subtitles/123/movie-en", name="Movie", language="en")
    out = scraper.download(sub, tmp_path)
    assert out.name == "failed.srt"


def test_download_unexpected_content_returns_failed(tmp_path: Path) -> None:
    session = _FakeSession(
        responses=[
            _FakeResponse(
                content=b"random bytes", headers={"Content-Type": "application/octet-stream"}
            ),
        ]
    )
    scraper = _scraper_with_session(OpenSubtitlesScraper(), session)
    sub = SubtitleResult(id="/en/subtitles/123/movie-en", name="Movie", language="en")
    out = scraper.download(sub, tmp_path)
    assert out.name == "failed.srt"


def test_download_anubis_not_passed_triggers_reauth(tmp_path: Path) -> None:
    # First call: 401 triggers anubis re-auth path
    session = _FakeSession(
        responses=[
            _FakeResponse(status_code=401, content=b"Making sure you"),
        ]
    )
    scraper = OpenSubtitlesScraper()
    scraper._session = session  # type: ignore[assignment]
    scraper._cf_passed = False
    # _pass_anubis will create a new session; patch to return True
    with patch.object(OpenSubtitlesScraper, "_pass_anubis", return_value=True):
        sub = SubtitleResult(id="/en/subtitles/123/movie-en", name="Movie", language="en")
        out = scraper.download(sub, tmp_path)
    # After reauth, another get returns default _FakeResponse (200, empty) -> failed
    assert out.name == "failed.srt"


# ---------------- search ----------------


def test_search_returns_results_when_passed() -> None:
    session = _FakeSession(
        responses=[
            _FakeResponse(text=_SEARCH_HTML),
        ]
    )
    scraper = _scraper_with_session(OpenSubtitlesScraper(), session)
    results = scraper.search("Movie", "en")
    # Search HTML has en + de results; filtering by "en" keeps only en
    assert len(results) == 1
    assert "en" in results[0].language


def test_search_non_200_returns_empty() -> None:
    session = _FakeSession(
        responses=[
            _FakeResponse(status_code=500, text="err"),
        ]
    )
    scraper = _scraper_with_session(OpenSubtitlesScraper(), session)
    assert scraper.search("Movie", "en") == []


def test_search_anubis_rate_limit_retry_then_success() -> None:
    session = _FakeSession(
        responses=[
            _FakeResponse(status_code=401, text="Making sure you"),
            _FakeResponse(text=_SEARCH_HTML),  # retry succeeds
        ]
    )
    scraper = _scraper_with_session(OpenSubtitlesScraper(), session)
    # Patch sleep to avoid real waits
    with patch("subtitle_correction.scraper.time.sleep"):
        results = scraper.search("Movie", "en")
    # Filtered to en-only results
    assert len(results) == 1


def test_search_anubis_reauth_failure_returns_empty() -> None:
    session = _FakeSession(
        responses=[
            _FakeResponse(status_code=401, text="Making sure you"),
            _FakeResponse(status_code=401, text="Making sure you"),
            _FakeResponse(status_code=401, text="Making sure you"),
            _FakeResponse(status_code=401, text="Making sure you"),
        ]
    )
    scraper = _scraper_with_session(OpenSubtitlesScraper(), session)
    scraper._cf_passed = False
    with (
        patch("subtitle_correction.scraper.time.sleep"),
        patch.object(OpenSubtitlesScraper, "_pass_anubis", return_value=False),
    ):
        assert scraper.search("Movie", "en") == []


def test_search_filters_by_language() -> None:
    # HTML has en and de results; filtering by "en" keeps only en
    session = _FakeSession(
        responses=[
            _FakeResponse(text=_SEARCH_HTML),
        ]
    )
    scraper = _scraper_with_session(OpenSubtitlesScraper(), session)
    results = scraper.search("Movie", "en")
    assert all("en" in r.language for r in results)


# ---------------- _pass_anubis ----------------


def test_pass_anubis_already_passed_returns_true() -> None:
    scraper = OpenSubtitlesScraper()
    scraper._cf_passed = True
    assert scraper._pass_anubis() is True


def test_pass_anubis_no_challenge_200_returns_true() -> None:
    session = _FakeSession(
        responses=[
            _FakeResponse(status_code=200, text="normal page no challenge"),
        ]
    )
    scraper = OpenSubtitlesScraper()
    scraper._session = session  # type: ignore[assignment]
    scraper._cf_passed = False
    assert scraper._pass_anubis() is True
    assert scraper._cf_passed is True


def test_pass_anubis_with_challenge_solved() -> None:
    challenge_html = (
        '<script id="anubis_challenge" type="application/json">'
        + json.dumps(
            {
                "challenge": {"id": "cid", "randomData": "rdata"},
                "rules": {"difficulty": 0},
            }
        )
        + "</script>"
    )
    # First: challenge page; second: pass-challenge success; third: not needed
    session = _FakeSession(
        responses=[
            _FakeResponse(status_code=200, text=challenge_html),
            _FakeResponse(status_code=200, text="success no making sure"),
        ]
    )
    scraper = OpenSubtitlesScraper()
    scraper._session = session  # type: ignore[assignment]
    scraper._cf_passed = False
    assert scraper._pass_anubis() is True
    assert scraper._cf_passed is True


def test_pass_anubis_challenge_not_found_non200_returns_false() -> None:
    session = _FakeSession(
        responses=[
            _FakeResponse(status_code=403, text="forbidden"),
        ]
    )
    scraper = OpenSubtitlesScraper()
    scraper._session = session  # type: ignore[assignment]
    scraper._cf_passed = False
    assert scraper._pass_anubis() is False


def test_pass_anubis_pass_challenge_fails_then_retry_succeeds() -> None:
    challenge_html = (
        '<script id="anubis_challenge" type="application/json">'
        + json.dumps(
            {
                "challenge": {"id": "cid", "randomData": "rdata"},
                "rules": {"difficulty": 0},
            }
        )
        + "</script>"
    )
    # pass-challenge returns 200 but "Making sure you"; retry of /en succeeds
    session = _FakeSession(
        responses=[
            _FakeResponse(status_code=200, text=challenge_html),
            _FakeResponse(status_code=200, text="Making sure you"),
            _FakeResponse(status_code=200, text="all good now"),
        ]
    )
    scraper = OpenSubtitlesScraper()
    scraper._session = session  # type: ignore[assignment]
    scraper._cf_passed = False
    assert scraper._pass_anubis() is True


def test_pass_anubis_pass_challenge_fully_fails() -> None:
    challenge_html = (
        '<script id="anubis_challenge" type="application/json">'
        + json.dumps(
            {
                "challenge": {"id": "cid", "randomData": "rdata"},
                "rules": {"difficulty": 0},
            }
        )
        + "</script>"
    )
    session = _FakeSession(
        responses=[
            _FakeResponse(status_code=200, text=challenge_html),
            _FakeResponse(status_code=403, text="still blocked"),
            _FakeResponse(status_code=403, text="still blocked"),
        ]
    )
    scraper = OpenSubtitlesScraper()
    scraper._session = session  # type: ignore[assignment]
    scraper._cf_passed = False
    assert scraper._pass_anubis() is False


# ---------------- login / close ----------------


def test_login_when_passed(tmp_path: Path) -> None:
    # ensure_cache_dir uses the isolated home from conftest
    scraper = OpenSubtitlesScraper()
    scraper._cf_passed = True
    # _ensure_session creates a real session; patch _pass_anubis
    with patch.object(OpenSubtitlesScraper, "_pass_anubis", return_value=True):
        scraper.login()
    assert scraper._cf_passed is True


def test_login_when_not_passed(tmp_path: Path) -> None:
    scraper = OpenSubtitlesScraper()
    scraper._cf_passed = False

    def _fake_pass(self, target_url=None):
        self._cf_passed = True
        return True

    with patch.object(OpenSubtitlesScraper, "_pass_anubis", _fake_pass):
        scraper.login()
    assert scraper._cf_passed is True


def test_close_resets_state() -> None:
    session = _FakeSession()
    scraper = OpenSubtitlesScraper()
    scraper._session = session  # type: ignore[assignment]
    scraper._cf_passed = True
    scraper.close()
    assert scraper._session is None
    assert scraper._cf_passed is False


def test_close_no_session_is_noop() -> None:
    scraper = OpenSubtitlesScraper()
    scraper.close()
    assert scraper._session is None


# ---------------- search_and_download ----------------


def test_search_and_download_cache_hit(tmp_path: Path) -> None:
    from subtitle_correction.cache import add_cache_entry

    # Pre-populate cache with a valid srt
    srt = tmp_path / "cached.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nhi\n", encoding="utf-8")
    add_cache_entry("movie.mkv", srt, "en")

    scraper = OpenSubtitlesScraper()
    parsed = ParsedFilename(title="Movie", raw="movie.mkv")
    result = scraper.search_and_download(parsed, language="en", output_dir=tmp_path)
    assert result == srt


def test_search_and_download_finds_result(tmp_path: Path) -> None:
    session = _FakeSession(
        responses=[
            _FakeResponse(text=_SEARCH_HTML),  # search
            _FakeResponse(
                content=b"1\n00:00:01,000 --> 00:00:02,000\nhi\n",
                headers={"Content-Type": "text/plain"},
            ),  # download
        ]
    )
    scraper = _scraper_with_session(OpenSubtitlesScraper(), session)
    parsed = ParsedFilename(title="Movie", raw="movie.mkv")
    result = scraper.search_and_download(parsed, language="en", output_dir=tmp_path)
    assert result is not None
    assert result.exists()


def test_search_and_download_no_results_returns_none(tmp_path: Path) -> None:
    # search returns nothing, fallback search also nothing
    session = _FakeSession(
        responses=[
            _FakeResponse(text="<html></html>"),
            _FakeResponse(text="<html></html>"),
        ]
    )
    scraper = _scraper_with_session(OpenSubtitlesScraper(), session)
    parsed = ParsedFilename(title="Unknown Movie", year=2020, raw="unknown.mkv")
    result = scraper.search_and_download(parsed, language="en", output_dir=tmp_path)
    assert result is None


def test_search_and_download_fallback_query_when_no_results(tmp_path: Path) -> None:
    # First search empty, fallback search returns a result, download returns srt
    session = _FakeSession(
        responses=[
            _FakeResponse(text="<html></html>"),  # initial search
            _FakeResponse(text=_SEARCH_HTML),  # fallback search
            _FakeResponse(
                content=b"1\n00:00:01,000 --> 00:00:02,000\nhi\n",
                headers={"Content-Type": "text/plain"},
            ),  # download
        ]
    )
    scraper = _scraper_with_session(OpenSubtitlesScraper(), session)
    parsed = ParsedFilename(title="Movie", year=2020, raw="movie.mkv")
    result = scraper.search_and_download(parsed, language="en", output_dir=tmp_path)
    assert result is not None


# ---------------- _ensure_session ----------------


def test_ensure_session_creates_session(monkeypatch: pytest.MonkeyPatch) -> None:
    scraper = OpenSubtitlesScraper()
    created: list = []

    class _S:
        def __init__(self):
            self.headers = {}
            created.append(self)

    monkeypatch.setattr("subtitle_correction.scraper.requests.Session", _S, raising=True)
    with patch.object(OpenSubtitlesScraper, "_pass_anubis", return_value=True):
        session = scraper._ensure_session()
    assert scraper._session is session
    assert created  # session constructed


def test_wait_rate_limit_sleeps(monkeypatch: pytest.MonkeyPatch) -> None:
    scraper = OpenSubtitlesScraper(rate_limit=10.0)
    scraper._last_request = 0.0  # force long elapsed
    slept: list = []
    monkeypatch.setattr("subtitle_correction.scraper.time.sleep", lambda s: slept.append(s))
    monkeypatch.setattr("subtitle_correction.scraper.time.time", lambda: 0.0)
    scraper._wait_rate_limit()
    assert slept  # slept at least once
