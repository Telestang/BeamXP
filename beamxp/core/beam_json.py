from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

from beamxp.core import sjson
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
    if current.endswith((",", "{", "[")):
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


def strip_commas_before_colons(text: str) -> str:
    """Drop commas that sit between a key and its colon.

    BeamNG's own parser tolerates typos such as ``"lightbar_sign",:""`` (stock
    etk800 ``844_police_A.pc`` as of 0.39), so a config the game loads must not
    fail here. Commas inside string literals are left alone.
    """
    out: list[str] = []
    in_string = False
    escape = False
    length = len(text)
    for idx, ch in enumerate(text):
        if in_string:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == ",":
            probe = idx + 1
            while probe < length and text[probe] in " \t\r\n":
                probe += 1
            if probe < length and text[probe] == ":":
                continue
        out.append(ch)
        if ch == '"':
            in_string = True
    return "".join(out)


def parse_beamng_json(text: str, *, label: str) -> dict[str, object]:
    """Read a BeamNG config/jbeam document the way the game reads it.

    Delegates to the SJSON reader ported from the engine's own ``ljson.lua``
    (see :mod:`beamxp.core.sjson`), which accepts everything BeamNG accepts --
    most importantly, commas as pure whitespace. The repair-into-strict-JSON
    passes below it are kept only for callers that still import them; nothing in
    the read path needs them now.
    """
    try:
        parsed = sjson.decode(text)
    except sjson.SJSONError as error:
        raise RuntimeError(f"Could not parse BeamNG config {label}: {error}") from error
    if not isinstance(parsed, dict):
        raise TypeError(f"BeamNG config {label} did not parse to an object")
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


def humanize_config_key(value: str) -> str:
    tokens = [token for token in re.split(r"[_\s-]+", value.strip()) if token]
    out: list[str] = []
    for token in tokens:
        if token.isupper() or re.search(r"\d", token):
            out.append(token)
        elif len(token) == 1:
            out.append(token.upper())
        else:
            out.append(token[:1].upper() + token[1:].lower())
    return " ".join(out)


LOCALE_DIR = "locales/translations"
FALLBACK_LANGUAGE = "en-US"

_translation_cache: dict[str, dict[str, str]] = {}


def _translations_from_dir(directory: Path) -> list[dict[str, str]]:
    if not directory.is_dir():
        return []
    files: list[dict[str, str]] = []
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict):
            files.append({k: v for k, v in data.items() if isinstance(v, str)})
    return files


def _translations_from_zip(source_zip: Path) -> list[dict[str, str]]:
    prefix = f"{LOCALE_DIR}/{FALLBACK_LANGUAGE}/"
    files: list[dict[str, str]] = []
    try:
        with zipfile.ZipFile(source_zip) as zf:
            names = sorted(
                name
                for name in zf.namelist()
                if name.replace("\\", "/").lower().startswith(prefix.lower())
                and name.lower().endswith(".json")
            )
            for name in names:
                try:
                    data = json.loads(zf.read(name).decode("utf-8-sig"))
                except (KeyError, ValueError):
                    continue
                if isinstance(data, dict):
                    files.append({k: v for k, v in data.items() if isinstance(v, str)})
    except (OSError, zipfile.BadZipFile):
        return []
    return files


def beamng_translations(source_zip: Path | None = None) -> dict[str, str]:
    """The game's en-US string table, as ``core_locales.translate`` sees it.

    ``lua/ge/extensions/core/locales.lua`` loads every JSON under
    ``/locales/translations/<language>/`` into a flat key -> string list and
    returns the first file that holds a key, falling back to en-US when the
    active language has no entry. The game's language is not knowable from
    here, so this reads the fallback the game itself would land on.

    A mod vehicle ships its own keys in the same layout inside its zip, mounted
    over the install's, so the zip is searched first.
    """
    cache_key = str(source_zip or "")
    cached = _translation_cache.get(cache_key)
    if cached is not None:
        return cached

    from beamxp.core.files import beamng_game_common_zips

    files: list[dict[str, str]] = []
    if source_zip is not None:
        files.extend(_translations_from_zip(source_zip))
    for common_zip in beamng_game_common_zips():
        # <game>/content/vehicles/common.zip -> <game>/locales/translations/en-US
        install_root = common_zip.parent.parent.parent
        files.extend(
            _translations_from_dir(install_root / LOCALE_DIR / FALLBACK_LANGUAGE)
        )

    merged: dict[str, str] = {}
    for data in reversed(files):
        merged.update(data)
    _translation_cache[cache_key] = merged
    return merged


def localized_string(value: str, source_zip: Path | None = None) -> str | None:
    """What ``_tr`` returns for a locale key, or None when nothing holds it."""
    text = value.strip()
    if not text:
        return None
    translated = beamng_translations(source_zip).get(text)
    return translated.strip() if translated and translated.strip() else None


def display_name_from_localization_key(
    value: str,
    source_zip: Path | None = None,
) -> str | None:
    """The string a ``vehiclesData.*`` key resolves to in the selector.

    The game runs every ``Configuration``/``Name`` through ``_tr``
    (``core/vehicles.lua``), so the key is the authored value and the locale
    table holds the label the player reads -- "Ardente S 410Q Frisson (M)" for
    ``vehiclesData.vivace.ardente_S_410Q_M.Configuration``. Only when the table
    cannot be reached at all does this fall back to prettifying the key's own
    config segment, which drops any word that is not in the config's filename.
    """
    text = value.strip()
    if not text.startswith("vehiclesData."):
        return None
    translated = localized_string(text, source_zip)
    if translated is not None:
        return translated
    parts = [part for part in text.split(".") if part]
    if len(parts) < 3:
        return None
    candidate = parts[-2] if parts[-1] in {"Configuration", "Name", "Description"} else parts[-1]
    if candidate in {"Configuration", "Name", "Description"}:
        return None
    display = humanize_config_key(candidate)
    return display or None


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
            localized_fallback = display_name_from_localization_key(value, source_zip)
            return localized_fallback or value.strip()
    return humanize_config_key(config_name) or config_name


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
