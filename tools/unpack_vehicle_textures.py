"""Extract a BeamNG vehicle ZIP and make PNG inspection copies of its textures.

Example:
    python tools/unpack_vehicle_textures.py path\\to\\my_vehicle.zip

The default output directory is ``<zip name>_texture_probe`` beside the ZIP.
It contains ``extracted`` (the complete unpacked archive) and ``png`` (PNG
copies of every recognised texture, retaining the archive's folder layout).
"""

from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path, PurePosixPath

from PIL import Image, UnidentifiedImageError

TEXTURE_SUFFIXES = frozenset({".dds", ".png", ".tga", ".jpg", ".jpeg", ".bmp"})


def archive_relative_path(member_name: str) -> Path:
    """Return a safe local path for a ZIP member or reject a traversal path."""
    member = PurePosixPath(member_name.replace("\\", "/"))
    if member.is_absolute() or ".." in member.parts or any(":" in part for part in member.parts):
        raise ValueError(f"unsafe archive member path: {member_name!r}")
    return Path(*member.parts)


def texture_png_path(relative_path: Path) -> Path:
    """Keep the archive layout while replacing a texture's extension with PNG."""
    return relative_path.with_suffix(".png")


def write_texture_png(data: bytes, destination: Path) -> None:
    """Decode one image payload and write an RGBA PNG preview."""
    with Image.open(io.BytesIO(data)) as image:
        image.convert("RGBA").save(destination, format="PNG")


def unpack_and_convert(archive_path: Path, output_directory: Path) -> tuple[int, list[str]]:
    """Extract an archive and return the converted-texture count and failures."""
    extracted_root = output_directory / "extracted"
    png_root = output_directory / "png"
    converted = 0
    failures: list[str] = []

    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue

            try:
                relative_path = archive_relative_path(info.filename)
            except ValueError as error:
                failures.append(str(error))
                continue

            data = archive.read(info)
            extracted_path = extracted_root / relative_path
            extracted_path.parent.mkdir(parents=True, exist_ok=True)
            extracted_path.write_bytes(data)

            if relative_path.suffix.lower() not in TEXTURE_SUFFIXES:
                continue

            png_path = png_root / texture_png_path(relative_path)
            png_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                write_texture_png(data, png_path)
            except (OSError, UnidentifiedImageError) as error:
                failures.append(f"{info.filename}: {error}")
            else:
                converted += 1

    return converted, failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="BeamNG vehicle ZIP file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="new destination directory (default: <zip name>_texture_probe)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    archive_path = args.archive.expanduser().resolve()
    if not archive_path.is_file():
        print(f"Archive not found: {archive_path}", file=sys.stderr)
        return 2

    output_directory = (
        args.output.expanduser().resolve()
        if args.output is not None
        else archive_path.with_suffix("").with_name(f"{archive_path.stem}_texture_probe")
    )
    if output_directory.exists():
        print(
            f"Output directory already exists: {output_directory}\n"
            "Choose a new --output directory so no previous probe is overwritten.",
            file=sys.stderr,
        )
        return 2

    try:
        converted, failures = unpack_and_convert(archive_path, output_directory)
    except (OSError, zipfile.BadZipFile) as error:
        print(f"Could not read {archive_path.name}: {error}", file=sys.stderr)
        return 1

    print(f"Extracted archive to: {output_directory / 'extracted'}")
    print(f"Wrote {converted} PNG texture preview(s) to: {output_directory / 'png'}")
    if failures:
        print("\nSkipped files:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
