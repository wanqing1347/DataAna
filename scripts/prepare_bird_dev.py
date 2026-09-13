#!/usr/bin/env python3
"""Download and verify the official BIRD-SQL Dev package used by resume evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

OFFICIAL_DEV_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"
EXPECTED_FIRST_120_DIFFICULTY = {
    "simple": 73,
    "moderate": 39,
    "challenging": 8,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare official BIRD-SQL Dev data")
    parser.add_argument(
        "--target",
        default="data/bird",
        help="directory that will contain the downloaded/extracted BIRD Dev package",
    )
    parser.add_argument(
        "--url",
        default=OFFICIAL_DEV_URL,
        help="dataset archive URL; defaults to the official BIRD Dev download",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="download dev.zip again even when it already exists",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="do not download/extract; only verify an existing extracted package",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".part")
    print(f"[bird-data] downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as response, temp.open("wb") as output:
        total = int(response.headers.get("Content-Length") or 0)
        copied = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            copied += len(chunk)
            if total:
                print(
                    f"\r[bird-data] {copied / 1024 / 1024:.1f}/"
                    f"{total / 1024 / 1024:.1f} MiB",
                    end="",
                    flush=True,
                )
        if total:
            print()
    temp.replace(destination)


def extract_zip(archive: Path, destination: Path) -> None:
    print(f"[bird-data] extracting {archive} -> {destination}")
    with zipfile.ZipFile(archive) as package:
        package.extractall(destination)


def find_dev_root(target: Path) -> Path:
    candidates: list[Path] = []
    if (target / "dev.json").is_file():
        candidates.append(target)
    for path in target.rglob("dev.json"):
        parent = path.parent
        if (parent / "dev_databases").exists() or (parent / "dev_databases.zip").exists():
            candidates.append(parent)
    unique = sorted(set(path.resolve() for path in candidates), key=lambda path: len(path.parts))
    if not unique:
        raise RuntimeError(
            f"cannot find a BIRD Dev root under {target}; expected dev.json and dev_databases"
        )
    return unique[0]


def ensure_databases(dev_root: Path) -> Path:
    db_root = dev_root / "dev_databases"
    nested_archive = dev_root / "dev_databases.zip"
    if nested_archive.is_file() and not db_root.is_dir():
        extract_zip(nested_archive, dev_root)
    if (db_root / "dev_databases").is_dir():
        db_root = db_root / "dev_databases"
    if not db_root.is_dir():
        raise RuntimeError(f"dev_databases not found under {dev_root}")
    return db_root


def load_dev(dev_json: Path) -> list[dict[str, Any]]:
    data = json.loads(dev_json.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise RuntimeError(f"{dev_json} is not a JSON array")
    required = {"question_id", "db_id", "question", "evidence", "SQL", "difficulty"}
    for index, row in enumerate(data):
        if not isinstance(row, dict) or not required.issubset(row):
            raise RuntimeError(f"invalid BIRD row at index {index}")
    return data


def verify(dev_root: Path, archive: Path | None, source_url: str) -> dict[str, Any]:
    dev_json = dev_root / "dev.json"
    if not dev_json.is_file():
        raise RuntimeError(f"dev.json not found: {dev_json}")
    db_root = ensure_databases(dev_root)
    rows = load_dev(dev_json)
    first_120 = rows[:120]
    difficulty = dict(Counter(str(row["difficulty"]) for row in first_120))

    missing_dbs = []
    for db_id in sorted({str(row["db_id"]) for row in first_120}):
        db_path = db_root / db_id / f"{db_id}.sqlite"
        if not db_path.is_file():
            missing_dbs.append(str(db_path))
    if missing_dbs:
        preview = "\n".join(f" - {path}" for path in missing_dbs[:10])
        raise RuntimeError(f"missing SQLite databases for first 120 questions:\n{preview}")

    fingerprint_ok = difficulty == EXPECTED_FIRST_120_DIFFICULTY
    manifest = {
        "source_url": source_url,
        "dev_root": str(dev_root.resolve()),
        "dev_json_sha256": sha256_file(dev_json),
        "archive_sha256": sha256_file(archive) if archive and archive.is_file() else None,
        "question_count": len(rows),
        "first_120_difficulty": difficulty,
        "expected_first_120_difficulty": EXPECTED_FIRST_120_DIFFICULTY,
        "resume_subset_fingerprint_ok": fingerprint_ok,
        "first_120_database_count": len({str(row["db_id"]) for row in first_120}),
    }
    if not fingerprint_ok:
        raise RuntimeError(
            "first-120 difficulty fingerprint does not match the historical resume run: "
            f"actual={difficulty}, expected={EXPECTED_FIRST_120_DIFFICULTY}"
        )
    return manifest


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    target = Path(args.target)
    if not target.is_absolute():
        target = repo_root / target
    target = target.resolve()
    archive = target / "dev.zip"

    if not args.verify_only:
        if args.force_download or not archive.is_file():
            download(args.url, archive)
        else:
            print(f"[bird-data] reusing {archive}")
        try:
            dev_root = find_dev_root(target)
        except RuntimeError:
            extract_zip(archive, target)
            dev_root = find_dev_root(target)
    else:
        dev_root = find_dev_root(target)

    manifest = verify(dev_root, archive if archive.is_file() else None, args.url)
    manifest_path = target / "manifest.resume-120.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"[bird-data] dev-root={manifest['dev_root']}")
    print(f"[bird-data] questions={manifest['question_count']}")
    print(f"[bird-data] first-120 difficulty={manifest['first_120_difficulty']}")
    print(f"[bird-data] dev.json sha256={manifest['dev_json_sha256']}")
    print(f"[bird-data] manifest={manifest_path}")
    print("[bird-data] RESUME_120_DATA_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
