"""Writing a file path down without writing the user's name down with it.

A saved project, a plate design and the record the build ships inside every
mod it produces all name files: fonts, artwork, source zips, texture reports.
Spelt in full, every one of those begins with the user's home directory, so a
project that was otherwise perfectly shareable could not be shared.

Two policies, because the two kinds of file want different things.

``portable_path`` is for a file that gets reopened -- a plate design's font.
A path inside a BeamXP directory becomes a token, which is both name-free and
correct on somebody else's machine, because the directory is the app's own.
Anything else is left exactly as it was: there is no portable spelling for
artwork sitting in someone's Downloads folder, and rewriting it would break
the editor that reopens it.

``shareable_path`` is for a record nobody resolves again -- the copy of the
conversion inside a built mod. There the same tokens apply, plus a project
root, and any path that matches none of them is cut back to its file name,
which is the part a reader of a diagnostic actually wants.

Kept in ``core`` rather than beside either caller because both the conversion
config and the plate library need it, and the config already imports the
plate generator -- the other direction would close a cycle.
"""

from __future__ import annotations

import os
from pathlib import Path


def _user_data_dir() -> Path:
    """Where BeamXP keeps its data, resolved now rather than at import.

    The same reading ``plate_generator`` takes, and for the same reason: the
    environment override is what the headless harnesses and the tests set, and
    they set it after this module is imported. Binding the roots at import
    time would tokenise against the real install while writing into a temp
    one. ``default_user_data_dir`` is imported here rather than at the top
    because ``core.constants`` imports the plate generator, and the generator
    imports this module.
    """
    from beamxp.plates.generator import default_user_data_dir

    override = os.environ.get("BEAMXP_DATA_DIR") or os.environ.get("BEAMHDC_DATA_DIR")
    return Path(override) if override else default_user_data_dir()


def portable_roots() -> tuple[tuple[str, Path], ...]:
    """Directories BeamXP owns, and the token each is written as.

    Longest first: the fonts directory is inside the data directory, so a
    plain ordering would match the wider root and lose the more specific name.
    """
    data = _user_data_dir()
    return (
        ("{fonts}", data / "fonts"),
        ("{plates}", data / "plates"),
        ("{data}", data),
    )


def tokenise(text: str, roots: tuple[tuple[str, Path], ...] | None = None) -> str | None:
    """``text`` rewritten against the first root that contains it, or None."""
    try:
        resolved = Path(text)
    except (TypeError, ValueError):
        return None
    for token, root in (portable_roots() if roots is None else roots):
        try:
            relative = resolved.relative_to(root)
        except ValueError:
            continue
        return f"{token}/{relative.as_posix()}" if relative.parts else token
    return None


def portable_path(value: object) -> object:
    """An absolute path under a BeamXP directory, rewritten as a token."""
    text = str(value or "")
    if not text:
        return value
    return tokenise(text) or value


def resolved_path(value: object) -> object:
    """The inverse: a token expanded back to this machine's directory."""
    text = str(value or "")
    for token, root in portable_roots():
        if text.startswith(token + "/"):
            return str(root / text[len(token) + 1:])
        if text == token:
            return str(root)
    return value


def looks_like_absolute_path(text: str) -> bool:
    """A drive-letter path or a UNC share, and nothing else.

    Deliberately narrow: this decides whether a string in a shipped record
    gets cut back to a file name, and a colour, a pattern or a mesh name must
    never qualify. A leading-slash path is *not* matched, because in this
    codebase that spelling belongs to the game rather than the disk --
    "/vehicles/common/" is a jbeam directory the loader searches, and cutting
    it to "common" would gut the very diagnostics this is protecting.
    """
    if len(text) >= 3 and text[0].isalpha() and text[1] == ":" and text[2] in r"\/":
        return True
    return text.startswith(("\\\\", "//"))


def shareable_path(value: object, project_dir: Path | None = None) -> object:
    """A path with nothing machine-specific left in it, for a shipped record."""
    text = str(value or "")
    if not text or not looks_like_absolute_path(text):
        return value
    roots = portable_roots()
    if project_dir is not None:
        roots = (("{project}", Path(project_dir)), *roots)
    tokenised = tokenise(text, roots)
    if tokenised is not None:
        return tokenised
    # Split on both separators by hand rather than through PurePath: it is
    # PureWindowsPath on Windows and PurePosixPath elsewhere, and only one of
    # those understands a backslash, so a report written on Windows and read
    # anywhere else would come out unsplit.
    return text.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1] or value


def map_paths(node: object, rewrite) -> object:
    """Walk a nested structure, applying ``rewrite`` to every string in it.

    Applied to every string rather than to a list of known fields on purpose:
    a plate design grows fields -- a band image, a full-width band image, a
    front and a rear background -- and one added later would otherwise keep
    writing an absolute path with nobody noticing. Both rewrites leave a
    string that is not a path they recognise exactly as they found it.
    """
    if isinstance(node, dict):
        return {key: map_paths(value, rewrite) for key, value in node.items()}
    if isinstance(node, list):
        return [map_paths(value, rewrite) for value in node]
    if isinstance(node, str):
        return rewrite(node)
    return node
