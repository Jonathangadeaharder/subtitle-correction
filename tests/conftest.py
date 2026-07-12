"""Shared pytest fixtures.

Redirects the user home directory (and thus the opensubtitles cache under
``~/.opensubtitles-cache``) into a per-test tmp path so cache tests are
hermetic and never touch the real filesystem.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_home(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(home))
    # Some helpers read Path.home() lazily, but cache.py binds module-level
    # paths at import time. Rebind them to the isolated home so cache tests
    # exercise the real read/write logic without touching the real home.
    from subtitle_correction import cache as cache_mod

    cache_root = home / ".opensubtitles-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cache_mod, "CACHE_DIR", cache_root, raising=True)
    monkeypatch.setattr(cache_mod, "METADATA_FILE", cache_root / "metadata.json", raising=True)
    monkeypatch.setattr(cache_mod, "DOWNLOADS_DIR", cache_root / "downloads", raising=True)
    (cache_root / "downloads").mkdir(parents=True, exist_ok=True)
