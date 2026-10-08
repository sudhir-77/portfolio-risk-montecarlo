"""French file parser, cache, and the fixed backtest calendar."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from src.data import (
    ensure_french_zip,
    file_sha256,
    load_industry_returns,
    parse_french_value_weighted,
    resolve_backtest_bounds,
    validate_industry_panel,
)


TINY = """Preamble

  Average Value Weighted Returns -- Daily
,Aa,Bb,Cc
20200102, 1.00, -99.99, 3.00
20200103, -1.00, 2.00, -999

Average Equal Weighted Returns -- Daily
,Aa,Bb,Cc
20200102, 9.00, 9.00, 9.00
20200103, 9.00, 9.00, 9.00
"""


def _zip_bytes(text: str = TINY) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("49_Industry_Portfolios_Daily.csv", text)
    return buffer.getvalue()


def test_parser_uses_value_weights_and_missing_codes():
    frame = parse_french_value_weighted(TINY)
    assert list(frame["date"].dt.strftime("%Y-%m-%d")) == ["2020-01-02", "2020-01-03"]
    assert frame["row_id"].tolist() == [0, 1]
    row = frame.iloc[0]
    assert row["Aa"] == pytest.approx(0.01)
    assert np.isnan(row["Bb"])
    assert row["Cc"] == pytest.approx(0.03)
    assert frame.iloc[1]["Bb"] == pytest.approx(0.02)
    assert np.isnan(frame.iloc[1]["Cc"])
    with pytest.raises(ValueError):
        parse_french_value_weighted("no section here")


def test_validate_rejects_a_short_or_narrow_panel():
    frame = parse_french_value_weighted(TINY)
    with pytest.raises(ValueError):
        validate_industry_panel(frame, min_rows=10, n_industries=3)
    with pytest.raises(ValueError):
        validate_industry_panel(frame, min_rows=2, n_industries=49)
    assert validate_industry_panel(frame, min_rows=2, n_industries=3) == ["Aa", "Bb", "Cc"]


def test_cache_hit_does_not_call_the_fetcher(tmp_path: Path):
    calls = {"n": 0}

    def fetcher(url: str) -> bytes:
        calls["n"] += 1
        return _zip_bytes()

    first = ensure_french_zip(tmp_path, url="http://example.test/french.zip", fetcher=fetcher)
    second = ensure_french_zip(tmp_path, url="http://example.test/french.zip", fetcher=fetcher)
    assert first == second
    assert calls["n"] == 1
    frame, manifest = load_industry_returns(
        tmp_path,
        url="http://example.test/french.zip",
        fetcher=fetcher,
        min_rows=2,
        n_industries=3,
    )
    assert len(frame) == 2
    assert manifest["sha256"] == file_sha256(first)
    assert calls["n"] == 1
    manifest_path = tmp_path / "manifest.json"
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    saved["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(RuntimeError):
        load_industry_returns(
            tmp_path,
            fetcher=fetcher,
            min_rows=2,
            n_industries=3,
        )


def test_bad_http_status(monkeypatch):
    from src import data as data_module

    class Response:
        status_code = 503
        content = b""

    monkeypatch.setattr(data_module.requests, "get", lambda *args, **kwargs: Response())
    with pytest.raises(RuntimeError):
        data_module._http_get("http://example.test/missing.zip")


def test_backtest_bounds_are_created_once(tmp_path: Path):
    start, end = resolve_backtest_bounds(tmp_path, quick=False)
    assert (start, end) == ("2005-01-01", "2024-12-31")
    path = tmp_path / "splits.json"
    original = path.read_text(encoding="utf-8")
    path.write_text(
        json.dumps({"test_start": "2010-01-01", "test_end": "2011-12-31"}),
        encoding="utf-8",
    )
    assert resolve_backtest_bounds(tmp_path, quick=False) == ("2010-01-01", "2011-12-31")
    assert path.read_text(encoding="utf-8") != original
    assert resolve_backtest_bounds(tmp_path, quick=True) == ("2016-01-01", "2016-06-30")


def test_corrupt_zip_is_replaced(tmp_path: Path):
    bad = tmp_path / "49_Industry_Portfolios_Daily_CSV.zip"
    bad.write_bytes(b"not a zip")
    dest = ensure_french_zip(tmp_path, fetcher=lambda url: _zip_bytes())
    assert zipfile.is_zipfile(dest)
