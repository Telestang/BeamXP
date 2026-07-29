from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

from beamxp.core.files import vehicle_prefix


def strip_json_comments(text: str) -> str:
    out: list[str] = []
    in_string = False
    escape = False
    line_comment = False
    block_comment = False
    idx = 0
    while idx < len(text):
        ch = text[idx]
        nxt = text[idx + 1] if idx + 1 < len(text) else ""
        if line_comment:
            if ch in "\r\n":
                line_comment = False
                out.append(ch)
            idx += 1
            continue
        if block_comment:
            if ch == "*" and nxt == "/":
                block_comment = False
                idx += 2
            else:
                out.append("\n" if ch in "\r\n" else " ")
                idx += 1
            continue
        if in_string:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            idx += 1
            continue
        if ch == "/" and nxt == "/":
            line_comment = True
            idx += 2
            continue
        if ch == "/" and nxt == "*":
            block_comment = True
            idx += 2
            continue
        out.append(ch)
        if ch == '"':
            in_string = True
        idx += 1
    return "".join(out)


def json_line_needs_comma(current: str, next_line: str) -> bool:
    current = current.strip()
    next_line = next_line.strip()
    if not current or not next_line:
        return False
    if current.endswith(",") or current.endswith("{") or current.endswith("["):
        return False
    if next_line[0] in "]}":
        return False
    if not re.search(r'(?:"|-?\d+(?:\.\d*)?|\.\d+|true|false|null|\]|\})$', current):
        return False
    return bool(re.match(r'(?:"|\{|\[|-?\d+(?:\.\d*)?|\.\d+|true|false|null)', next_line))


def add_missing_json_commas(text: str) -> str:
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    for index, line in enumerate(lines):
        next_line = ""
        for candidate in lines[index + 1 :]:
            if candidate.strip():
                next_line = candidate
                break
        if next_line and json_line_needs_comma(line, next_line):
            line_ending = ""
            content = line
            if content.endswith("\r\n"):
                content, line_ending = content[:-2], "\r\n"
            elif content.endswith("\n"):
                content, line_ending = content[:-1], "\n"
            line = content + "," + line_ending
        out.append(line)
    return "".join(out)


def parse_beamng_json(text: str, *, label: str) -> dict[str, object]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as first_error:
        cleaned = strip_json_comments(text.lstrip("\ufeff"))
        cleaned = re.sub(r",\s*,+", ",", cleaned)
        cleaned = add_missing_json_commas(cleaned)
        cleaned = re.sub(r",(\s*[\]}])", r"\1", cleaned)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as second_error:
            try:
                parsed, end = json.JSONDecoder().raw_decode(cleaned)
                remainder = cleaned[end:].strip()
                if remainder.strip("}").strip():
                    raise second_error
            except json.JSONDecodeError:
                raise RuntimeError(
                    f"Could not parse BeamNG config {label}: {second_error}. "
                    f"Initial strict JSON error was: {first_error}"
                ) from second_error
    if not isinstance(parsed, dict):
        raise RuntimeError(f"BeamNG config {label} did not parse to an object")
    return parsed


def load_pc(source_zip: Path, pc_path: str) -> dict[str, object]:
    with zipfile.ZipFile(source_zip) as zf:
        return parse_beamng_json(
            zf.read(pc_path).decode("utf-8", errors="replace"),
            label=pc_path,
        )


def load_info(source_zip: Path, info_path: str) -> dict[str, object]:
    with zipfile.ZipFile(source_zip) as zf:
        return parse_beamng_json(
            zf.read(info_path).decode("utf-8", errors="replace"),
            label=info_path,
        )


def info_path_for_config(source_zip: Path, vehicle_id: str, config_name: str) -> str | None:
    candidates = [
        f"{vehicle_prefix(vehicle_id)}/info_{config_name}.json",
        f"{vehicle_prefix(vehicle_id)}/{config_name}.json",
    ]
    with zipfile.ZipFile(source_zip) as zf:
        names = {name.replace("\\", "/") for name in zf.namelist()}
    return next((candidate for candidate in candidates if candidate in names), None)


def display_name_for(source_zip: Path, info_path: str | None, config_name: str) -> str:
    if info_path is None:
        return config_name
    try:
        info = load_info(source_zip, info_path)
    except Exception:
        return config_name
    for key in ("Configuration", "Name", "name", "configuration"):
        value = info.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return config_name


def zip_json_by_name(source_zip: Path, wanted_name: str) -> dict[str, object]:
    wanted = wanted_name.replace("\\", "/").lower()
    with zipfile.ZipFile(source_zip) as zf:
        actual = next(
            (
                name
                for name in zf.namelist()
                if name.replace("\\", "/").lower() == wanted
            ),
            None,
        )
        if not actual:
            return {}
        try:
            parsed = parse_beamng_json(
                zf.read(actual).decode("utf-8", errors="replace"),
                label=actual,
            )
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
