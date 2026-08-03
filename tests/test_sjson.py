"""Grammar tests for the SJSON reader ported from BeamNG's ljson.lua.

Each case is a rule the engine's parser actually implements, not a guess about
what stock content happens to contain. Where a case came from real content the
vehicle is named, because those are the ones that cost a "Load vehicle failed".
"""

from __future__ import annotations

import json
import unittest

from beamxp.core import sjson
from beamxp.core.beam_json import parse_beamng_json


class CommasAreWhitespaceTests(unittest.TestCase):
    # ljson skipWhiteSpace: `while (c ~= nil and c <= 32) or c == 44`.
    # Commas carry no meaning, which is the whole reason stock content parses.
    EXPECTED = {"a": 1, "b": 2}

    def test_comma_between_pairs(self) -> None:
        self.assertEqual(sjson.decode('{"a":1,"b":2}'), self.EXPECTED)

    def test_no_comma_at_all(self) -> None:
        self.assertEqual(sjson.decode('{"a":1 "b":2}'), self.EXPECTED)

    def test_repeated_commas(self) -> None:
        self.assertEqual(sjson.decode('{"a":1,,,,"b":2}'), self.EXPECTED)

    def test_comma_between_key_and_colon(self) -> None:
        # Stock etk800 844_police_A.pc (0.39): "lightbar_sign",:""
        self.assertEqual(sjson.decode('{"a",:1 "b":2}'), self.EXPECTED)

    def test_comma_between_colon_and_value(self) -> None:
        # Stock bluebuck: "bluebuck_bumper_F":, {...}
        self.assertEqual(sjson.decode('{"a":,1 "b":2}'), self.EXPECTED)

    def test_leading_and_trailing_commas(self) -> None:
        self.assertEqual(sjson.decode('{,"a":1,"b":2,}'), self.EXPECTED)

    def test_commas_are_optional_in_arrays(self) -> None:
        self.assertEqual(sjson.decode('{"a":[1 2,3,,4,]}'), {"a": [1, 2, 3, 4]})

    def test_missing_comma_mid_line_between_objects(self) -> None:
        # Stock log_trailer: {"nodeMove":{...}"variables":{...}}
        self.assertEqual(
            sjson.decode('{"nodeMove":{"x":0.0}"variables":{"$b":-0.4}}'),
            {"nodeMove": {"x": 0.0}, "variables": {"$b": -0.4}},
        )


class SyntaxTests(unittest.TestCase):
    def test_unquoted_keys(self) -> None:
        self.assertEqual(sjson.decode("{slotType:\"acme\" count_2:3}"), {"slotType": "acme", "count_2": 3})

    def test_equals_separator(self) -> None:
        self.assertEqual(sjson.decode('{a = 1, b = "x"}'), {"a": 1, "b": "x"})

    def test_root_braces_are_optional(self) -> None:
        self.assertEqual(sjson.decode('"a":1 "b":2'), {"a": 1, "b": 2})

    def test_line_and_block_comments(self) -> None:
        text = '{ // one\n "a":1 /* two */, "b":2 // three\n }'
        self.assertEqual(sjson.decode(text), {"a": 1, "b": 2})

    def test_comment_between_key_and_value(self) -> None:
        self.assertEqual(sjson.decode('{"a": /* c */ 1}'), {"a": 1})

    def test_bom_is_ignored(self) -> None:
        self.assertEqual(sjson.decode('\ufeff{"a":1}'), {"a": 1})


class NumberTests(unittest.TestCase):
    def test_leading_zeros(self) -> None:
        # Stock autobello autobuggy_lightbar_LED: "$rotY":00
        self.assertEqual(sjson.decode('{"a":00, "b":007}'), {"a": 0, "b": 7})

    def test_leading_plus(self) -> None:
        self.assertEqual(sjson.decode('{"a":+5, "b":+1.5}'), {"a": 5, "b": 1.5})

    def test_infinity_spellings(self) -> None:
        parsed = sjson.decode('{"a":Infinity, "b":-Infinity, "c":1#INF00}')
        self.assertEqual(parsed["a"], float("inf"))
        self.assertEqual(parsed["b"], float("-inf"))
        self.assertEqual(parsed["c"], float("inf"))

    def test_ints_stay_ints_and_floats_stay_floats(self) -> None:
        parsed = sjson.decode('{"i":3, "f":3.0, "e":1e3, "neg":-0.25}')
        self.assertIsInstance(parsed["i"], int)
        self.assertIsInstance(parsed["f"], float)
        self.assertEqual(parsed["e"], 1000.0)
        self.assertEqual(parsed["neg"], -0.25)


class StringTests(unittest.TestCase):
    def test_known_escapes(self) -> None:
        self.assertEqual(sjson.decode(r'{"a":"x\ty\nz\"q\\w"}'), {"a": 'x\ty\nz"q\\w'})

    def test_unknown_escape_keeps_its_backslash(self) -> None:
        # ljson has no \p, and keeps the backslash rather than erroring.
        self.assertEqual(sjson.decode(r'{"a":"c:\path"}'), {"a": r"c:\path"})

    def test_unicode_escape_is_decoded(self) -> None:
        # Deliberate superset of ljson: we read back our own json.dumps output,
        # which escapes non-ASCII. The engine would show these literally.
        self.assertEqual(sjson.decode(r'{"a":"\u00e9"}'), {"a": "é"})

    def test_commas_and_colons_inside_strings_survive(self) -> None:
        self.assertEqual(sjson.decode('{"a":"x,: y"}'), {"a": "x,: y"})


class NullTests(unittest.TestCase):
    def test_null_removes_the_key(self) -> None:
        # null is nil in Lua, so the key is absent rather than present-and-None.
        # material stages rely on this: `"emissiveMap" in stage` must be False.
        parsed = sjson.decode('{"a":1, "emissiveMap":null}')
        self.assertEqual(parsed, {"a": 1})
        self.assertNotIn("emissiveMap", parsed)

    def test_null_is_kept_in_arrays(self) -> None:
        # jbeam rows are positional; dropping would shift every later column.
        self.assertEqual(sjson.decode('{"row":[1,null,3]}'), {"row": [1, None, 3]})


class ErrorTests(unittest.TestCase):
    def test_unclosed_object_reports_a_location(self) -> None:
        with self.assertRaises(sjson.SJSONError) as caught:
            sjson.decode('{"a":1')
        self.assertIn("line 1", str(caught.exception))

    def test_missing_separator_is_an_error(self) -> None:
        with self.assertRaises(sjson.SJSONError):
            sjson.decode('{"a" 1}')

    def test_parse_beamng_json_wraps_errors_with_the_label(self) -> None:
        with self.assertRaises(RuntimeError) as caught:
            parse_beamng_json('{"a":', label="acme.pc")
        self.assertIn("acme.pc", str(caught.exception))


class InteropTests(unittest.TestCase):
    def test_strict_json_still_reads_identically(self) -> None:
        payload = {
            "parts": {"a": "", "b": "x"},
            "vars": {"$n": -1.5},
            "rows": [["id", 1, 2.5, True]],
        }
        text = json.dumps(payload, indent=2)
        self.assertEqual(parse_beamng_json(text, label="strict.pc"), payload)

    def test_round_trips_our_own_non_ascii_output(self) -> None:
        payload = {"Configuration": "Ölfilter Spécial — 日本"}
        text = json.dumps(payload, indent=2)  # ensure_ascii escapes to \uXXXX
        self.assertEqual(parse_beamng_json(text, label="info.json"), payload)


if __name__ == "__main__":
    unittest.main()
