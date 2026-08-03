"""SJSON reader, ported from BeamNG's own reference parser.

BeamNG does not parse jbeam/.pc as JSON. ``lua/common/json.lua`` binds to
LuaJIT's decoder for **SJSON** (Autodesk's Simplified JSON,
https://github.com/Autodesk/sjson) and ``lua/common/jbeam/io.lua`` feeds jbeam
straight into it with no preprocessing. ``lua/common/ljson.lua`` is BeamNG's
readable Lua implementation of the same grammar; this module is a port of it.

The rule that matters most, from ljson's ``skipWhiteSpace``::

    while (c ~= nil and c <= 32) or c == 44 do -- matches space tab newline or comma

Commas are **whitespace**, everywhere. They are not "tolerated" or "optional
between elements" -- they carry no meaning at all, so ``{"a":1 "b":2}``,
``{"a":1,,,,"b":2}``, ``{"a":1,}`` and ``{"a",:1}`` are one and the same
document. Every stock-content quirk that used to need its own repair pass here
(missing commas, doubled commas, trailing commas, ``"part":, {``, ``"key",:""``)
falls out of that single rule.

The rest of the grammar:

* keys may be quoted or bare ``[A-Za-z0-9_]+``
* the key/value separator may be ``:`` or ``=``
* the root braces are optional -- a bare key/value sequence is an object
* ``//`` line and ``/* */`` block comments
* numbers may carry a leading ``+``, leading zeros (``00``), and may be
  ``Infinity``, ``-Infinity`` or MSVC's ``1#INF00``
* the escape table is small: ``\\t \\n \\f \\r \\b \\" \\\\``, plus ``\\9`` for
  tab and ``\\0`` for carriage return. An unrecognised escape keeps its
  backslash literally
* ``null`` is Lua ``nil``, so the key is simply absent from the object

Two deliberate deviations from ljson, both documented at their implementation:
``\\uXXXX`` is decoded (we read back our own ``json.dumps`` output, which
escapes non-ASCII), and nested ``/*`` is not treated as an error.
"""

from __future__ import annotations

# Escapes exactly as ljson's `escapes` table keys them. Note the oddities: a
# backslash before a literal newline yields a newline, \9 is a tab and \0 is a
# carriage return. There is no \u here -- see _read_string.
_ESCAPES = {
    "t": "\t",
    "n": "\n",
    "f": "\f",
    "r": "\r",
    "b": "\b",
    '"': '"',
    "\\": "\\",
    "\n": "\n",
    "9": "\t",
    "0": "\r",
}

_BARE_KEY_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
)
_DIGITS = frozenset("0123456789")
_HEX = frozenset("0123456789abcdefABCDEF")


class SJSONError(ValueError):
    """Raised when input cannot be read even under SJSON's rules."""


class _Reader:
    __slots__ = ("s", "n")

    def __init__(self, text: str) -> None:
        self.s = text
        self.n = len(text)

    def fail(self, message: str, index: int) -> None:
        index = max(0, min(index, self.n))
        line = self.s.count("\n", 0, index) + 1
        line_start = self.s.rfind("\n", 0, index) + 1
        column = index - line_start + 1
        raise SJSONError(f"{message}: line {line} column {column} (char {index})")

    def skip_space(self, i: int) -> int:
        """Whitespace, comments and commas -- ljson treats all three alike."""
        s = self.s
        n = self.n
        while i < n:
            ch = s[i]
            if ch <= " " or ch == ",":
                i += 1
            elif ch == "/":
                i = self._skip_comment(i)
            else:
                break
        return i

    def _skip_comment(self, i: int) -> int:
        s = self.s
        n = self.n
        nxt = s[i + 1] if i + 1 < n else ""
        if nxt == "/":
            end = s.find("\n", i + 2)
            return n if end < 0 else end + 1
        if nxt == "*":
            # ljson raises on a '/*' nested inside a block comment; accepting it
            # only ever reads more content, never less, so it stays lenient.
            end = s.find("*/", i + 2)
            return n if end < 0 else end + 2
        self.fail("Invalid comment", i)
        return i

    def read_string(self, i: int) -> tuple[str, int]:
        s = self.s
        n = self.n
        start = i
        i += 1
        # Fast path: no escapes, which is almost every string in practice.
        end = s.find('"', i)
        if end < 0:
            self.fail("String not having an end-quote", start)
        if s.find("\\", i, end) < 0:
            return s[i:end], end + 1

        out: list[str] = []
        while i < n:
            ch = s[i]
            if ch == '"':
                return "".join(out), i + 1
            if ch != "\\":
                out.append(ch)
                i += 1
                continue
            nxt = s[i + 1] if i + 1 < n else ""
            # Deviation from ljson: it has no \u, so BeamNG reads "é"
            # literally. We decode it, because this same reader loads the
            # json.dumps output we write ourselves (ensure_ascii escapes
            # non-ASCII), and decoding is a superset of what the engine accepts.
            if nxt == "u" and len(s) >= i + 6 and all(c in _HEX for c in s[i + 2 : i + 6]):
                out.append(chr(int(s[i + 2 : i + 6], 16)))
                i += 6
                continue
            mapped = _ESCAPES.get(nxt)
            if mapped is None:
                # ljson keeps the backslash and carries on from the next char.
                out.append("\\")
                i += 1
            else:
                out.append(mapped)
                i += 2
        self.fail("String not having an end-quote", start)
        return "", i

    def read_number(self, i: int) -> tuple[float | int, int]:
        s = self.s
        n = self.n
        start = i
        negative = False
        if i < n and s[i] in "+-":
            negative = s[i] == "-"
            i += 1
        if s.startswith("Infinity", i):
            return (float("-inf") if negative else float("inf")), i + 8
        # MSVC's infinity spelling, which ljson's readNumber accepts explicitly.
        if s.startswith("1#INF00", i):
            return (float("-inf") if negative else float("inf")), i + 7
        digits_start = i
        while i < n and s[i] in _DIGITS:  # leading zeros are fine: 00 -> 0
            i += 1
        is_float = False
        if i < n and s[i] == ".":
            is_float = True
            i += 1
            while i < n and s[i] in _DIGITS:
                i += 1
        if i < n and s[i] in "eE":
            is_float = True
            i += 1
            if i < n and s[i] in "+-":
                i += 1
            while i < n and s[i] in _DIGITS:
                i += 1
        if i == digits_start:
            self.fail(f"Invalid number {s[start : start + 12]!r}", start)
        text = s[start:i]
        try:
            return (float(text) if is_float else int(text)), i
        except ValueError:
            self.fail(f"Invalid number {text!r}", start)
            return 0, i

    def read_key(self, i: int) -> tuple[str, int]:
        s = self.s
        n = self.n
        if i < n and s[i] == '"':
            key, i = self.read_string(i)
        else:
            start = i
            while i < n and s[i] in _BARE_KEY_CHARS:
                i += 1
            if i == start:
                self.fail("Expected dictionary key", i)
            key = s[start:i]
        i = self.skip_space(i)
        if i >= n or s[i] not in ":=":
            self.fail("Expected dictionary separator ':' or '='", i)
        return key, i + 1

    def read_value(self, i: int) -> tuple[object, int]:
        s = self.s
        if i >= self.n:
            self.fail("Unexpected end of input", i)
        ch = s[i]
        if ch == "{":
            return self.read_object(i)
        if ch == "[":
            return self.read_array(i)
        if ch == '"':
            return self.read_string(i)
        if ch == "t" and s.startswith("true", i):
            return True, i + 4
        if ch == "f" and s.startswith("false", i):
            return False, i + 5
        if ch == "n" and s.startswith("null", i):
            return None, i + 4
        if ch == "I" and s.startswith("Infinity", i):
            return float("inf"), i + 8
        if ch in _DIGITS or ch in "+-":
            return self.read_number(i)
        self.fail(f"Invalid input {ch!r}", i)
        return None, i

    def read_object(self, i: int) -> tuple[dict[str, object], int]:
        out: dict[str, object] = {}
        start = i
        i = self.skip_space(i + 1)
        while i < self.n and self.s[i] != "}":
            key, i = self.read_key(i)
            i = self.skip_space(i)
            value, i = self.read_value(i)
            # null is nil in Lua, which means the key is absent rather than
            # present-and-null. Callers doing pc.get("parts", {}) depend on it.
            if value is None:
                out.pop(key, None)
            else:
                out[key] = value
            i = self.skip_space(i)
        if i >= self.n:
            self.fail("Unclosed object", start)
        return out, i + 1

    def read_array(self, i: int) -> tuple[list[object], int]:
        out: list[object] = []
        start = i
        i = self.skip_space(i + 1)
        while i < self.n and self.s[i] != "]":
            value, i = self.read_value(i)
            # Kept, unlike in objects: jbeam rows are positional, so dropping a
            # null would silently shift every later column in the row.
            out.append(value)
            i = self.skip_space(i)
        if i >= self.n:
            self.fail("Unclosed array", start)
        return out, i + 1


def decode(text: str) -> object:
    """Decode SJSON text to Python values, as BeamNG's parser would read it."""
    reader = _Reader(text.lstrip("﻿"))
    i = reader.skip_space(0)
    if i >= reader.n:
        return {}
    if reader.s[i] in "{[":
        value, i = reader.read_value(i)
        i = reader.skip_space(i)
        if i < reader.n:
            # Trailing content after the root value: ljson stops here, but a
            # second root object means the file is not what the caller expects.
            reader.fail("Unexpected trailing content", i)
        return value
    # The root braces are optional: a bare key/value sequence is an object.
    out: dict[str, object] = {}
    while i < reader.n:
        key, i = reader.read_key(i)
        i = reader.skip_space(i)
        value, i = reader.read_value(i)
        if value is None:
            out.pop(key, None)
        else:
            out[key] = value
        i = reader.skip_space(i)
    return out


__all__ = ["SJSONError", "decode"]
