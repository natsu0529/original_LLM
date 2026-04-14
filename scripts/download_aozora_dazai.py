#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin
from urllib.request import Request, urlopen


BASE_URL = "https://www.aozora.gr.jp/"
CATALOG_ZIP_URL = urljoin(BASE_URL, "index_pages/list_person_all_utf8.zip")
AUTHOR_ID = "000035"
AUTHOR_NAME = "太宰 治"
DEFAULT_TIMEOUT = 30.0
DEFAULT_SLEEP_SECONDS = 0.2
USER_AGENT = "original_LLM/0.1 (Aozora Bunko downloader)"


@dataclass
class Work:
    author_id: str
    author_name: str
    work_id: str
    title: str
    orthography: str
    status: str
    card_url: str


@dataclass
class DownloadRecord:
    work_id: str
    title: str
    orthography: str
    card_url: str
    download_url: str | None
    archive_path: str | None
    text_path: str | None
    source_filename: str | None
    source_encoding: str | None
    status: str
    error: str | None = None


class DownloadTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_download_table = False
        self.in_row = False
        self.in_cell = False
        self.current_cell_parts: list[str] = []
        self.current_cell_links: list[str] = []
        self.current_row: list[dict[str, str | None]] = []
        self.rows: list[list[dict[str, str | None]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = dict(attrs)
        if tag == "table" and attr_map.get("class") == "download":
            self.in_download_table = True
            return

        if not self.in_download_table:
            return

        if tag == "tr":
            self.in_row = True
            self.current_row = []
            return

        if self.in_row and tag in {"td", "th"}:
            self.in_cell = True
            self.current_cell_parts = []
            self.current_cell_links = []
            return

        if self.in_cell and tag == "a":
            href = attr_map.get("href")
            if href:
                self.current_cell_links.append(href)

    def handle_data(self, data: str) -> None:
        if self.in_cell:
            self.current_cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if not self.in_download_table:
            return

        if self.in_row and tag in {"td", "th"}:
            text = normalize_space("".join(self.current_cell_parts))
            href = self.current_cell_links[0] if self.current_cell_links else None
            self.current_row.append({"text": text, "href": href})
            self.in_cell = False
            self.current_cell_parts = []
            self.current_cell_links = []
            return

        if self.in_row and tag == "tr":
            if self.current_row:
                self.rows.append(self.current_row)
            self.in_row = False
            self.current_row = []
            return

        if tag == "table":
            self.in_download_table = False


def normalize_space(value: str) -> str:
    return " ".join(value.replace("\u3000", " ").split())


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_output_dir = repo_root / "data" / "raw" / "aozora" / "dazai"

    parser = argparse.ArgumentParser(
        description="Download all public Aozora Bunko works for Osamu Dazai."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Directory where archives, extracted text, and manifest are written.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N works from the public Dazai catalog.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=DEFAULT_SLEEP_SECONDS,
        help="Sleep between HTTP requests to avoid hammering Aozora Bunko.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload files even if the archive and extracted text already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List target works without downloading any files.",
    )
    return parser.parse_args()


def fetch_bytes(url: str, timeout: float) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def load_catalog(timeout: float) -> list[Work]:
    catalog_bytes = fetch_bytes(CATALOG_ZIP_URL, timeout=timeout)
    with zipfile.ZipFile(io.BytesIO(catalog_bytes)) as catalog_zip:
        csv_name = next(name for name in catalog_zip.namelist() if name.lower().endswith(".csv"))
        with catalog_zip.open(csv_name) as raw_file:
            text_file = io.TextIOWrapper(raw_file, encoding="utf-8-sig", newline="")
            reader = csv.DictReader(text_file)
            works: list[Work] = []
            for row in reader:
                author_id = row["人物ID"].strip()
                status = row["状態"].strip()
                if author_id != AUTHOR_ID or status != "公開":
                    continue

                work_id = str(int(row["作品ID"]))
                works.append(
                    Work(
                        author_id=author_id,
                        author_name=row["著者名"].strip(),
                        work_id=work_id,
                        title=row["作品名"].strip(),
                        orthography=row["仮名遣い種別"].strip(),
                        status=status,
                        card_url=urljoin(BASE_URL, f"cards/{AUTHOR_ID}/card{work_id}.html"),
                    )
                )
    return works


def choose_download_link(card_html: str) -> tuple[str | None, str | None]:
    parser = DownloadTableParser()
    parser.feed(card_html)

    candidates: list[tuple[str, str]] = []
    preferred: list[tuple[str, str]] = []

    for row in parser.rows:
        if len(row) < 3:
            continue
        file_type = row[0]["text"] or ""
        href = row[2]["href"]
        if not href:
            continue
        if "テキストファイル" not in file_type:
            continue
        full_url = urljoin(BASE_URL, f"cards/{AUTHOR_ID}/{href.lstrip('./')}")
        candidates.append((file_type, full_url))
        if "ルビなし" in file_type:
            preferred.append((file_type, full_url))

    if preferred:
        return preferred[0]
    if candidates:
        return candidates[0]
    return None, None


def decode_text(data: bytes) -> tuple[str, str]:
    encodings = ("cp932", "shift_jis", "utf-8", "euc_jp")
    for encoding in encodings:
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return data.decode("cp932", errors="replace"), "cp932-replace"


def extract_first_text_file(archive_bytes: bytes) -> tuple[str, str, str]:
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        names = archive.namelist()
        text_name = next(
            (name for name in names if name.lower().endswith(".txt")),
            names[0],
        )
        with archive.open(text_name) as raw_file:
            text_bytes = raw_file.read()
    text, encoding = decode_text(text_bytes)
    return text_name, text, encoding


def write_manifest(path: Path, records: Iterable[DownloadRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as manifest_file:
        for record in records:
            manifest_file.write(json.dumps(asdict(record), ensure_ascii=False))
            manifest_file.write("\n")


def download_work(
    work: Work,
    output_dir: Path,
    timeout: float,
    force: bool,
) -> DownloadRecord:
    archives_dir = output_dir / "zips"
    texts_dir = output_dir / "txt"
    archives_dir.mkdir(parents=True, exist_ok=True)
    texts_dir.mkdir(parents=True, exist_ok=True)

    archive_path = archives_dir / f"work_{work.work_id}.zip"
    text_path = texts_dir / f"work_{work.work_id}.txt"

    if not force and archive_path.exists() and text_path.exists():
        return DownloadRecord(
            work_id=work.work_id,
            title=work.title,
            orthography=work.orthography,
            card_url=work.card_url,
            download_url=None,
            archive_path=str(archive_path),
            text_path=str(text_path),
            source_filename=None,
            source_encoding=None,
            status="skipped_existing",
        )

    card_html = fetch_bytes(work.card_url, timeout=timeout).decode("utf-8", errors="replace")
    file_type, download_url = choose_download_link(card_html)
    if not download_url:
        return DownloadRecord(
            work_id=work.work_id,
            title=work.title,
            orthography=work.orthography,
            card_url=work.card_url,
            download_url=None,
            archive_path=None,
            text_path=None,
            source_filename=None,
            source_encoding=None,
            status="missing_text_download",
            error="No downloadable text archive was found on the card page.",
        )

    archive_bytes = fetch_bytes(download_url, timeout=timeout)
    archive_path.write_bytes(archive_bytes)

    source_filename, decoded_text, source_encoding = extract_first_text_file(archive_bytes)
    text_path.write_text(decoded_text, encoding="utf-8")

    return DownloadRecord(
        work_id=work.work_id,
        title=work.title,
        orthography=work.orthography,
        card_url=work.card_url,
        download_url=download_url,
        archive_path=str(archive_path),
        text_path=str(text_path),
        source_filename=source_filename,
        source_encoding=source_encoding if file_type else source_encoding,
        status="downloaded",
    )


def main() -> int:
    args = parse_args()
    works = load_catalog(timeout=args.timeout)
    if args.limit is not None:
        works = works[: args.limit]

    print(f"Found {len(works)} public works for {AUTHOR_NAME}.", file=sys.stderr)

    if args.dry_run:
        for work in works:
            print(f"{work.work_id}\t{work.orthography}\t{work.title}")
        return 0

    manifest_path = args.output_dir / "manifest.jsonl"
    records: list[DownloadRecord] = []

    for index, work in enumerate(works, start=1):
        print(
            f"[{index}/{len(works)}] {work.work_id} {work.title}",
            file=sys.stderr,
        )
        try:
            record = download_work(
                work=work,
                output_dir=args.output_dir,
                timeout=args.timeout,
                force=args.force,
            )
        except Exception as exc:  # noqa: BLE001
            record = DownloadRecord(
                work_id=work.work_id,
                title=work.title,
                orthography=work.orthography,
                card_url=work.card_url,
                download_url=None,
                archive_path=None,
                text_path=None,
                source_filename=None,
                source_encoding=None,
                status="error",
                error=str(exc),
            )

        records.append(record)
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    write_manifest(manifest_path, records)

    downloaded = sum(record.status == "downloaded" for record in records)
    skipped = sum(record.status == "skipped_existing" for record in records)
    failed = len(records) - downloaded - skipped

    print(
        f"Done. downloaded={downloaded} skipped={skipped} failed={failed} manifest={manifest_path}",
        file=sys.stderr,
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
