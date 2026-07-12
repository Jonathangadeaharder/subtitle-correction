import hashlib
import html as html_mod
import io
import json
import random
import re
import time
import zipfile
from pathlib import Path

import requests
from rich.console import Console

from .cache import (
    DOWNLOADS_DIR,
    add_cache_entry,
    ensure_cache_dir,
    find_cached_subtitle,
)
from .models import ParsedFilename, SubtitleResult

console = Console()

OPENSUBTITLES_BASE = "https://www.opensubtitles.org"

CHROME_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36"
)

LANG_CODE_MAP = {
    "en": "eng",
    "de": "ger",
    "es": "spa",
    "fr": "fre",
    "it": "ita",
    "pt": "por",
    "nl": "dut",
    "sv": "swe",
    "da": "dan",
    "fi": "fin",
    "nb": "nor",
    "pl": "pol",
    "cs": "cze",
    "sk": "slo",
    "hu": "hun",
    "ro": "rum",
    "bg": "bul",
    "el": "gre",
    "tr": "tur",
    "ru": "rus",
    "ja": "jpn",
    "ko": "kor",
    "zh": "chi",
    "ar": "ara",
    "he": "heb",
    "hi": "hin",
    "th": "tha",
    "vi": "vie",
    "id": "ind",
    "ms": "may",
}


def _solve_pow(random_data: str, difficulty: int) -> tuple[str, int]:
    """Solve Anubis proof-of-work challenge."""
    full_bytes = difficulty // 2
    half_byte = difficulty % 2 != 0

    nonce = 0
    while True:
        digest = hashlib.sha256(f"{random_data}{nonce}".encode()).digest()

        ok = all(digest[s] == 0 for s in range(full_bytes))
        if ok and half_byte and (digest[full_bytes] >> 4) != 0:
            ok = False

        if ok:
            return digest.hex(), nonce

        nonce += 1


class OpenSubtitlesScraper:
    """Pure HTTP scraper for opensubtitles.org with Anubis PoW bypass."""

    def __init__(self, rate_limit: float = 5.0, **_kwargs):
        self.rate_limit = rate_limit
        self._last_request = 0.0
        self._session: requests.Session | None = None
        self._cf_passed = False

    def _ensure_session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({"User-Agent": CHROME_USER_AGENT})
            self._pass_anubis()
        return self._session

    def _wait_rate_limit(self) -> None:
        elapsed = time.time() - self._last_request
        jitter = random.uniform(0, 1.0)
        wait = max(0, self.rate_limit + jitter - elapsed)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.time()

    def _pass_anubis(self, target_url: str | None = None) -> bool:
        if self._cf_passed:
            return True

        session = self._session
        if session is None:
            session = requests.Session()
            session.headers.update({"User-Agent": CHROME_USER_AGENT})
            self._session = session

        session.cookies.clear()
        url = target_url or f"{OPENSUBTITLES_BASE}/en"
        resp = session.get(url, allow_redirects=True)

        if resp.status_code == 200 and "anubis_challenge" not in resp.text:
            self._cf_passed = True
            return True

        challenge_match = re.search(
            r'<script id="anubis_challenge" type="application/json">(.*?)</script>',
            resp.text,
            re.DOTALL,
        )
        if not challenge_match:
            if resp.status_code == 200:
                self._cf_passed = True
                return True
            console.print(f"[red]Anubis challenge not found (status {resp.status_code})[/red]")
            return False

        challenge_data = json.loads(challenge_match.group(1))
        challenge = challenge_data["challenge"]
        rules = challenge_data["rules"]

        random_data = challenge["randomData"]
        difficulty = rules["difficulty"]
        challenge_id = challenge["id"]

        console.print(f"[dim]Solving Anubis PoW (difficulty={difficulty})...[/dim]")
        start = time.time()
        hash_hex, nonce = _solve_pow(random_data, difficulty)
        elapsed = time.time() - start
        console.print(f"[green]PoW solved in {elapsed:.2f}s (nonce={nonce})[/green]")

        pass_url = f"{OPENSUBTITLES_BASE}/.within.website/x/cmd/anubis/api/pass-challenge"
        params = {
            "id": challenge_id,
            "response": hash_hex,
            "nonce": str(nonce),
            "redir": f"{OPENSUBTITLES_BASE}/en",
            "elapsedTime": str(int(elapsed * 1000)),
        }
        resp = session.get(pass_url, params=params, allow_redirects=True)

        if resp.status_code == 200 and "Making sure you" not in resp.text:
            self._cf_passed = True
            console.print("[green]Anubis challenge passed[/green]")
            return True

        resp2 = session.get(f"{OPENSUBTITLES_BASE}/en", allow_redirects=True)
        if resp2.status_code == 200 and "Making sure you" not in resp2.text:
            self._cf_passed = True
            console.print("[green]Anubis challenge passed[/green]")
            return True

        console.print(f"[red]Anubis challenge failed (status {resp.status_code})[/red]")
        return False

    def login(self, timeout: int = 300, **_kwargs) -> None:
        ensure_cache_dir()
        self._ensure_session()
        if self._cf_passed:
            console.print("[green]Anubis challenge already passed[/green]")
        else:
            console.print("[red]Anubis challenge failed[/red]")

    def search(self, query: str, language: str = "en") -> list[SubtitleResult]:
        session = self._ensure_session()
        if not self._cf_passed:
            console.print("[yellow]Anubis not passed, attempting re-auth...[/yellow]")
            if not self._pass_anubis():
                console.print("[red]Cannot search — Anubis auth failed[/red]")
                return []

        self._wait_rate_limit()
        lang_code = LANG_CODE_MAP.get(language, language)
        search_url = (
            f"{OPENSUBTITLES_BASE}/en/search/"
            f"sublanguageid-{lang_code}/"
            f"moviename-{query.replace(' ', '+')}"
        )

        console.print(f"[dim]Searching: {query} (lang={language})[/dim]")
        resp = session.get(search_url, allow_redirects=True)

        if resp.status_code == 401 or "Making sure you" in resp.text:
            for retry_wait in [30, 60, 120]:
                console.print(
                    f"[yellow]Anubis rate-limited, waiting {retry_wait}s (retry)...[/yellow]"
                )
                time.sleep(retry_wait)
                resp = session.get(search_url, allow_redirects=True)
                if resp.status_code == 200 and "Making sure you" not in resp.text:
                    break
            if resp.status_code == 401:
                console.print("[yellow]Still 401, re-authenticating...[/yellow]")
                self._cf_passed = False
                if not self._pass_anubis():
                    return []
                self._wait_rate_limit()
                resp = session.get(search_url, allow_redirects=True)

        if resp.status_code != 200:
            console.print(f"[red]Search failed: HTTP {resp.status_code}[/red]")
            return []

        results = self._parse_search_results(resp.text)
        lang_lower = language.lower()
        filtered = [r for r in results if lang_lower in r.language.lower()]
        if filtered:
            console.print(f"[dim]Filtered {len(results)} -> {len(filtered)} results[/dim]")
            return filtered
        if results:
            console.print(
                f"[yellow]No '{language}' results, returning {len(results)} unfiltered[/yellow]"
            )
        return results

    def _parse_search_results(self, html: str) -> list[SubtitleResult]:
        results: list[SubtitleResult] = []
        seen_ids: set[str] = set()

        bnone_pattern = re.compile(
            r'<a\s+class="bnone"\s+([^>]*)>',
            re.DOTALL,
        )

        for match in bnone_pattern.finditer(html):
            attrs = match.group(1)
            href_match = re.search(r'href="(/en/subtitles/(\d+)/([^"]*))"', attrs)
            if not href_match:
                continue

            sub_path = href_match.group(1)
            sub_id = href_match.group(2)
            slug = href_match.group(3)

            if sub_id in seen_ids:
                continue
            seen_ids.add(sub_id)

            title_match = re.search(r'title="([^"]*)"', attrs)
            title = html_mod.unescape(title_match.group(1)) if title_match else slug
            title = title.replace("subtitles - ", "").strip()

            lang_match = re.search(r"-([a-z]{2,3})$", slug)
            lang = lang_match.group(1) if lang_match else ""

            dl_match = re.search(
                rf'/en/subtitleserve/sub/{sub_id}"[^>]*>\s*(\d[\d,]*)\s*x',
                html,
            )
            downloads = 0
            if dl_match:
                downloads = int(dl_match.group(1).replace(",", ""))

            download_url = f"{OPENSUBTITLES_BASE}/en/download/sub/{sub_id}"

            results.append(
                SubtitleResult(
                    id=sub_path,
                    name=title,
                    language=lang,
                    download_url=download_url,
                    downloads=downloads,
                )
            )

        return results

    def download(self, subtitle: SubtitleResult, output_dir: Path | None = None) -> Path:
        output_dir = output_dir or DOWNLOADS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        session = self._ensure_session()
        if not self._cf_passed:
            console.print("[yellow]Anubis not passed, attempting re-auth...[/yellow]")
            if not self._pass_anubis():
                console.print("[red]Cannot download — Anubis auth failed[/red]")
                return output_dir / "failed.srt"

        self._wait_rate_limit()

        sub_id = ""
        if subtitle.download_url:
            m = re.search(r"/sub/(\d+)", subtitle.download_url)
            if m:
                sub_id = m.group(1)
        if not sub_id and subtitle.id:
            m = re.search(r"/subtitles/(\d+)", subtitle.id)
            if m:
                sub_id = m.group(1)

        if not sub_id:
            console.print("[red]Cannot extract subtitle ID[/red]")
            return output_dir / "failed.srt"

        download_url = f"{OPENSUBTITLES_BASE}/en/download/sub/{sub_id}"
        console.print(f"[cyan]Downloading subtitle {sub_id}...[/cyan]")

        resp = session.get(download_url, allow_redirects=True)

        if resp.status_code == 401 or "Making sure you" in resp.text:
            console.print("[yellow]Anubis challenge triggered, re-solving...[/yellow]")
            self._cf_passed = False
            if not self._pass_anubis():
                return output_dir / "failed.srt"
            self._wait_rate_limit()
            resp = session.get(download_url, allow_redirects=True)

        if resp.status_code != 200:
            console.print(f"[red]Download failed: HTTP {resp.status_code}[/red]")
            return output_dir / "failed.srt"

        content_type = resp.headers.get("Content-Type", "")

        if "zip" in content_type:
            try:
                z = zipfile.ZipFile(io.BytesIO(resp.content))
                srt_files = [f for f in z.namelist() if f.endswith(".srt")]
                if not srt_files:
                    console.print("[red]No SRT file in ZIP[/red]")
                    return output_dir / "failed.srt"

                srt_data = z.read(srt_files[0])
                safe_name = re.sub(r"[^\w.-]", "_", subtitle.name or srt_files[0])[:80]
                if not safe_name.endswith(".srt"):
                    safe_name += ".srt"
                output_path = output_dir / safe_name
                output_path.write_bytes(srt_data)
                console.print(
                    f"[green]Downloaded: {output_path.name} ({len(srt_data)} bytes)[/green]"
                )
                return output_path
            except Exception as e:
                console.print(f"[red]ZIP extraction failed: {e}[/red]")
                return output_dir / "failed.srt"

        if "gzip" in content_type or resp.content[:2] == b"\x1f\x8b":
            import gzip

            try:
                srt_data = gzip.decompress(resp.content)
                safe_name = re.sub(r"[^\w.-]", "_", subtitle.name)[:80]
                if not safe_name.endswith(".srt"):
                    safe_name += ".srt"
                output_path = output_dir / safe_name
                output_path.write_bytes(srt_data)
                console.print(
                    f"[green]Downloaded: {output_path.name} ({len(srt_data)} bytes)[/green]"
                )
                return output_path
            except Exception:
                pass

        text = resp.content.decode("utf-8", errors="replace")
        if text.strip().startswith("1") or " --> " in text:
            safe_name = re.sub(r"[^\w.-]", "_", subtitle.name)[:80]
            if not safe_name.endswith(".srt"):
                safe_name += ".srt"
            output_path = output_dir / safe_name
            output_path.write_text(text)
            console.print(f"[green]Downloaded: {output_path.name} ({len(text)} bytes)[/green]")
            return output_path

        console.print(f"[red]Unexpected content type: {content_type}[/red]")
        return output_dir / "failed.srt"

    def search_and_download(
        self,
        parsed: ParsedFilename,
        language: str = "en",
        output_dir: Path | None = None,
    ) -> Path | None:
        cached = find_cached_subtitle(parsed.raw, language)
        if cached:
            console.print(f"[dim]Cache hit: {cached.name}[/dim]")
            return cached

        query = parsed.search_query
        if parsed.episode_label:
            query = f"{query} {parsed.episode_label}"

        console.print(f"[cyan]Searching OpenSubtitles: {query}[/cyan]")
        results = self.search(query, language)

        if not results:
            console.print(f"[yellow]No results for: {query}[/yellow]")
            fallback_query = parsed.title
            if parsed.year:
                fallback_query += f" {parsed.year}"
            console.print(f"[cyan]Trying fallback: {fallback_query}[/cyan]")
            results = self.search(fallback_query, language)

        if not results:
            console.print(f"[red]No subtitles found for: {parsed.title}[/red]")
            return None

        best = self._pick_best(results, parsed)
        console.print(f"[green]Downloading: {best.name}[/green]")
        srt_path = self.download(best, output_dir)

        if srt_path.exists() and srt_path.name != "failed.srt":
            add_cache_entry(parsed.raw, srt_path, language, best.id)

        return srt_path

    def _pick_best(self, results: list[SubtitleResult], parsed: ParsedFilename) -> SubtitleResult:
        scored = []
        for r in results:
            score = 0
            if parsed.episode_label and parsed.episode_label.lower() in r.name.lower():
                score += 100
            if parsed.title.lower() in r.name.lower():
                score += 50
            score += min(r.downloads // 1000, 20)
            scored.append((score, r))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1] if scored else results[0]

    def close(self) -> None:
        if self._session:
            self._session.close()
            self._session = None
        self._cf_passed = False
