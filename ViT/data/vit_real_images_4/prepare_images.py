#!/usr/bin/env python3
"""Verify bundled photos; optionally fetch missing bytes from pinned sources.

Uses only Python's standard library. Never replaces an existing image.
This script does not load a model, resize photographs, or install packages.
"""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def checked_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Image path escapes dataset directory")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Do not use the network")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "image_manifest.json").read_text(encoding="utf-8"))
    failures = []
    for item in manifest["images"]:
        try:
            path = checked_path(root, item["path"])
            if path.exists():
                data = path.read_bytes()
            else:
                if args.verify_only:
                    raise FileNotFoundError(f"Missing image: {item['path']}")
                parsed = urlparse(item["source_url"])
                if parsed.scheme != "https" or parsed.hostname != "gitlab.com":
                    raise ValueError("Expected a pinned HTTPS GitLab source")
                request = Request(item["source_url"], headers={"User-Agent": "ViT-image-preparation/1.0"})
                with urlopen(request, timeout=45) as response:
                    data = response.read(10_000_001)
                if len(data) > 10_000_000:
                    raise ValueError("Unexpectedly large response")
                if hashlib.sha256(data).hexdigest() != item["sha256"]:
                    raise ValueError("Downloaded file does not match the official SHA-256")
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as destination:
                    destination.write(data)
            if hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError("SHA-256 mismatch; existing file was not replaced")
            if len(data) != item["size_bytes"]:
                raise ValueError("File size mismatch")
            print(f"OK {item['id']}: {path}")
        except (OSError, URLError, ValueError) as exc:
            failures.append({"image": item["id"], "error": str(exc)})
            print(f"ERROR {item['id']}: {exc}")
    if failures:
        raise SystemExit(1)
    print("Verified 4 real photographs. Ground-truth labels are not provided.")


if __name__ == "__main__":
    main()
