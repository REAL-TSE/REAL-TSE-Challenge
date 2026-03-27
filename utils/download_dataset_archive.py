import argparse
import re
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

import requests


CHUNK_SIZE = 1024 * 1024


def parse_google_drive_file_id(raw_value: str) -> str:
    raw_value = raw_value.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{10,}", raw_value):
        return raw_value

    parsed = urlparse(raw_value)
    if "drive.google.com" not in parsed.netloc:
        raise ValueError(f"Not a Google Drive url or file id: {raw_value}")

    match = re.search(r"/file/d/([A-Za-z0-9_-]+)", parsed.path)
    if match:
        return match.group(1)

    query = parse_qs(parsed.query)
    for key in ("id",):
        values = query.get(key)
        if values:
            return values[0]

    raise ValueError(f"Unable to parse Google Drive file id from: {raw_value}")


def extract_confirm_token(response: requests.Response) -> Optional[str]:
    for cookie_name, cookie_value in response.cookies.items():
        if cookie_name.startswith("download_warning"):
            return cookie_value

    patterns = (
        r'confirm=([0-9A-Za-z_]+)',
        r'name="confirm"\s+value="([0-9A-Za-z_]+)"',
    )
    for pattern in patterns:
        match = re.search(pattern, response.text)
        if match:
            return match.group(1)
    return None


def response_is_download(response: requests.Response) -> bool:
    content_disposition = response.headers.get("content-disposition", "")
    content_type = response.headers.get("content-type", "")
    if "attachment" in content_disposition.lower():
        return True
    return not content_type.startswith("text/html")


def save_response(response: requests.Response, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    total_bytes = int(response.headers.get("content-length", "0") or "0")
    downloaded = 0

    with tmp_path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if not chunk:
                continue
            handle.write(chunk)
            downloaded += len(chunk)

    tmp_path.replace(output_path)
    if total_bytes:
        print(f"Downloaded {downloaded} / {total_bytes} bytes to {output_path}")
    else:
        print(f"Downloaded {downloaded} bytes to {output_path}")


def download_from_google_drive(file_id: str, output_path: Path) -> None:
    session = requests.Session()
    base_url = "https://drive.google.com/uc"
    params = {"export": "download", "id": file_id}

    response = session.get(base_url, params=params, stream=True, allow_redirects=True, timeout=60)
    if response_is_download(response):
        save_response(response, output_path)
        return

    token = extract_confirm_token(response)
    if token is None:
        alt_url = "https://drive.usercontent.google.com/download"
        response = session.get(
            alt_url,
            params={"id": file_id, "export": "download", "confirm": "t"},
            stream=True,
            allow_redirects=True,
            timeout=60,
        )
        if response_is_download(response):
            save_response(response, output_path)
            return
        raise SystemExit(
            "Google Drive download did not return a file. "
            "Please check sharing permissions for this Google Drive file."
        )

    response = session.get(
        base_url,
        params={"export": "download", "id": file_id, "confirm": token},
        stream=True,
        allow_redirects=True,
        timeout=60,
    )
    if not response_is_download(response):
        raise SystemExit(
            "Google Drive confirmation flow did not return a file. "
            "Please check sharing permissions for this Google Drive file."
        )
    save_response(response, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a dataset archive from Google Drive."
    )
    parser.add_argument("--output", type=Path, required=True, help="Local file path for the archive.")
    parser.add_argument(
        "--gdrive-file-id",
        type=str,
        required=True,
        help="Google Drive file id or a Google Drive sharing URL.",
    )
    args = parser.parse_args()

    file_id = parse_google_drive_file_id(args.gdrive_file_id)
    download_from_google_drive(file_id, args.output.resolve())


if __name__ == "__main__":
    main()
