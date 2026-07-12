from __future__ import annotations

import sys
from pathlib import Path

import pytest

from subtitle_correction.train import CONFIG_PATH, train_impl


def test_train_impl_missing_config_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    # Point CONFIG_PATH at a non-existent file
    fake_config = Path("/nonexistent/config.yaml")
    monkeypatch.setattr("subtitle_correction.train.CONFIG_PATH", fake_config, raising=True)
    with pytest.raises(SystemExit):
        train_impl()


def test_train_impl_runs_subprocess(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Create a fake config file so the existence check passes
    fake_config = tmp_path / "config.yaml"
    fake_config.write_text("model: test\n", encoding="utf-8")
    monkeypatch.setattr("subtitle_correction.train.CONFIG_PATH", fake_config, raising=True)

    captured: dict = {}

    def _fake_run(cmd, *args, **kwargs):
        captured["cmd"] = cmd

        class _R:
            returncode = 0
        return _R()

    monkeypatch.setattr("subtitle_correction.train.subprocess.run", _fake_run, raising=True)
    monkeypatch.setattr("subtitle_correction.train.sys.executable", "/usr/bin/python3", raising=False)

    with pytest.raises(SystemExit) as exc_info:
        train_impl()
    assert exc_info.value.code == 0
    assert captured["cmd"][1] == "-m"
    assert captured["cmd"][2] == "mlx_lm"
    assert "--train" in captured["cmd"]
    assert str(fake_config) in captured["cmd"]


def test_config_path_is_under_repo() -> None:
    assert CONFIG_PATH.name == "config.yaml"
    assert CONFIG_PATH.parent == Path(__file__).resolve().parents[1]
