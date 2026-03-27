from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import zstandard as zstd
from tqdm.auto import tqdm


PUSHT_URL = (
    "https://huggingface.co/datasets/quentinll/lewm-pusht/resolve/main/"
    "pusht_expert_train.h5.zst"
)
PUSHT_ARCHIVE = "pusht_expert_train.h5.zst"
PUSHT_OUTPUT = "pusht_expert_train.h5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download LeWM datasets into a repo-local stable-worldmodel cache."
    )
    parser.add_argument(
        "--dataset",
        default="pusht",
        choices=["pusht"],
        help="Dataset to download. Only PushT is defined in proj.md.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path(__file__).resolve().parents[1] / "data" / "stablewm",
        type=Path,
        help="Directory where the extracted .h5 file will be written.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload and overwrite existing files.",
    )
    return parser.parse_args()


def download(url: str, destination: Path, force: bool) -> None:
    if destination.exists() and not force:
        print(f"Archive already exists: {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(url) as response:
        total = int(response.headers.get("Content-Length", "0"))
        with destination.open("wb") as f, tqdm(
            total=total if total > 0 else None,
            unit="B",
            unit_scale=True,
            desc=f"download {destination.name}",
        ) as progress:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                progress.update(len(chunk))


def decompress_zst(archive_path: Path, output_path: Path, force: bool) -> None:
    if output_path.exists() and not force:
        print(f"Dataset already exists: {output_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dctx = zstd.ZstdDecompressor()
    with archive_path.open("rb") as src, output_path.open("wb") as dst:
        with dctx.stream_reader(src) as reader, tqdm(
            unit="B",
            unit_scale=True,
            desc=f"extract {output_path.name}",
        ) as progress:
            while True:
                chunk = reader.read(1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
                progress.update(len(chunk))


def main() -> int:
    args = parse_args()
    if args.dataset != "pusht":
        print(f"Unsupported dataset: {args.dataset}", file=sys.stderr)
        return 1

    output_dir = args.output_dir.resolve()
    archive_path = output_dir / PUSHT_ARCHIVE
    dataset_path = output_dir / PUSHT_OUTPUT

    download(PUSHT_URL, archive_path, force=args.force)
    decompress_zst(archive_path, dataset_path, force=args.force)

    print(f"Dataset ready at {dataset_path}")
    print("Training can now read it with:")
    print("uv run python train.py data=pusht")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
