"""Download, parse, and check the Kenneth French 49-industry daily file."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.config import (
    FRENCH_URL,
    HTTP_TIMEOUT_SECONDS,
    MIN_ROWS,
    N_INDUSTRIES,
    QUICK_TEST_END,
    QUICK_TEST_START,
    TEST_END,
    TEST_START,
    USER_AGENT,
    ZIP_NAME,
)

ID_COLUMNS = ("row_id", "date")


def industry_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in ID_COLUMNS]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _http_get(url: str) -> bytes:
    response = requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=HTTP_TIMEOUT_SECONDS,
    )
    if response.status_code != 200:
        raise RuntimeError(f"download failed: HTTP {response.status_code} for {url}")
    if not response.content:
        raise RuntimeError(f"download failed: empty body for {url}")
    return response.content


def _zip_has_csv(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as archive:
            return any(name.lower().endswith(".csv") for name in archive.namelist())
    except (zipfile.BadZipFile, OSError):
        return False


def ensure_french_zip(data_dir: Path, url: str = FRENCH_URL, fetcher=None) -> Path:
    """Return the cached zip, downloading it if the cache is missing or corrupt."""
    data_dir.mkdir(parents=True, exist_ok=True)
    dest = data_dir / ZIP_NAME
    if dest.exists() and not _zip_has_csv(dest):
        dest.unlink()
    if not dest.exists():
        payload = fetcher(url) if fetcher is not None else _http_get(url)
        dest.write_bytes(payload)
        if not _zip_has_csv(dest):
            dest.unlink(missing_ok=True)
            raise RuntimeError("downloaded file is not a zip that contains a CSV")
    return dest


def parse_french_value_weighted(text: str) -> pd.DataFrame:
    """Parse the value-weighted daily section. Returns are decimal, not percent.

    The file also contains an equal-weighted section. That section is ignored.
    French marks missing returns as -99.99 or -999 in percent units.
    """
    lines = text.splitlines()
    start = None
    for index, line in enumerate(lines):
        if "average value weighted returns" in line.lower():
            start = index
            break
    if start is None:
        raise ValueError("value-weighted returns section not found")

    header = None
    header_index = None
    for index in range(start + 1, min(start + 15, len(lines))):
        if lines[index].strip():
            header = lines[index]
            header_index = index
            break
    if header is None or header_index is None:
        raise ValueError("industry header not found")

    names = [part.strip() for part in header.split(",")[1:] if part.strip()]
    if not names:
        raise ValueError("industry header is empty")

    dates: list[str] = []
    rows: list[list[float]] = []
    for line in lines[header_index + 1 :]:
        stripped = line.strip()
        if not stripped:
            break
        lowered = stripped.lower()
        if lowered.startswith("average") or lowered.startswith("copyright"):
            break
        parts = line.split(",")
        date_text = parts[0].strip()
        if len(date_text) != 8 or not date_text.isdigit():
            break
        if len(parts) < len(names) + 1:
            continue
        values: list[float] = []
        for part in parts[1 : len(names) + 1]:
            raw = float(part)
            if raw <= -99.0:
                values.append(np.nan)
            else:
                values.append(raw / 100.0)
        dates.append(date_text)
        rows.append(values)

    if not rows:
        raise ValueError("no daily rows parsed")

    frame = pd.DataFrame(rows, columns=names, dtype=np.float64)
    frame.insert(0, "date", pd.to_datetime(dates, format="%Y%m%d"))
    frame.insert(0, "row_id", np.arange(len(frame), dtype=np.int64))
    return frame


def validate_industry_panel(
    frame: pd.DataFrame,
    min_rows: int = MIN_ROWS,
    n_industries: int = N_INDUSTRIES,
) -> list[str]:
    columns = industry_columns(frame)
    if len(columns) != n_industries:
        raise ValueError(f"expected {n_industries} industries, found {len(columns)}")
    if len(frame) < min_rows:
        raise ValueError(f"expected at least {min_rows} rows, found {len(frame)}")
    if frame["date"].duplicated().any():
        raise ValueError("duplicate dates in the industry file")
    if not frame["date"].is_monotonic_increasing:
        raise ValueError("dates are not sorted")
    return columns


def _read_zip_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        name = next(item for item in archive.namelist() if item.lower().endswith(".csv"))
        raw = archive.read(name)
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def load_industry_returns(
    data_dir: Path,
    url: str = FRENCH_URL,
    fetcher=None,
    min_rows: int = MIN_ROWS,
    n_industries: int = N_INDUSTRIES,
) -> tuple[pd.DataFrame, dict]:
    """Load the cached French panel. Writes data/manifest.json with the checksum."""
    zip_path = ensure_french_zip(data_dir, url=url, fetcher=fetcher)
    digest = file_sha256(zip_path)
    manifest_path = data_dir / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        previous_hash = previous.get("sha256")
        if previous_hash and previous_hash != digest:
            raise RuntimeError(
                "Cached French zip does not match data/manifest.json. "
                "Delete the data folder to download it again."
            )
    frame = parse_french_value_weighted(_read_zip_text(zip_path))
    columns = validate_industry_panel(frame, min_rows=min_rows, n_industries=n_industries)
    manifest = {
        "filename": ZIP_NAME,
        "sha256": digest,
        "n_rows": int(len(frame)),
        "n_industries": len(columns),
        "date_min": frame["date"].iloc[0].strftime("%Y-%m-%d"),
        "date_max": frame["date"].iloc[-1].strftime("%Y-%m-%d"),
        "source_url": url,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return frame, manifest


def resolve_backtest_bounds(data_dir: Path, quick: bool) -> tuple[str, str]:
    """Return (test_start, test_end). The full-run file is written once."""
    if quick:
        return QUICK_TEST_START, QUICK_TEST_END
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "splits.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        return str(saved["test_start"]), str(saved["test_end"])
    payload = {
        "test_start": TEST_START,
        "test_end": TEST_END,
        "validation": None,
        "note": (
            "Fixed calendar split. Days before test_start are history. "
            "No validation slice is used to choose windows or thresholds."
        ),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return TEST_START, TEST_END
