"""Host checks of the catalogue: schema, coverage, sources, FEMM results, lookup and cheat sheet.

  python3 -m unittest discover -s femm_brain_daro_v2/tests

These read files only. The Python tests themselves run in FEMM through verify/run_femm.py tests.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import lookup  # noqa: E402

RESULTS = ROOT / "verify" / "results"
# pyFEMM's general functions (not tied to one kind of problem): each needs an entry.
GENERAL = {"AWG", "IEC", "callfemm", "callfemm_noeval", "closefemm", "hideconsole", "hidepointprops",
           "main_maximize", "main_minimize", "main_resize", "main_restore", "messagebox", "newdocument",
           "opendocument", "openfemm", "showconsole", "showpointprops", "smartmesh"}
# pyFEMM's own helpers for building command text, not FEMM commands.
INTERNAL = {"doargs", "num", "numc", "quote", "quotec"}
OTHER_PHYSICS = ("ei_", "eo_", "hi_", "ho_", "ci_", "co_")


def pyfemm_functions() -> set[str]:
    """Every public function of pyFEMM's femm.py, from the inventory run (verify/inventory.py)."""
    lines = (RESULTS / "inventory.txt").read_text().splitlines()
    return {line.split()[1] for line in lines if line.startswith("FN ")}


def manual_commands() -> set[str]:
    """Commands the manual documents in its common, magnetics pre- and post-processor sections."""
    index = json.loads((RESULTS / "manual_index.json").read_text())
    return {e["name"] for e in index["entries"] if e["section"].startswith(("3.5", "3.6", "3.7"))}


def manual_mentions() -> set[str]:
    """pyFEMM functions the manual writes as a call somewhere in its text, such as
    newdocument(doctype) in the general commands."""
    return set(json.loads((RESULTS / "manual_index.json").read_text())["mentioned"])


class CatalogueTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = lookup.load()
        cls.entries = {e["name"]: e for e in cls.data["entries"]}
        cls.pyfemm = pyfemm_functions()
        cls.manual = manual_commands()

    def test_schema(self):
        self.assertEqual(lookup.validate(self.data), [])

    def test_every_pyfemm_function_is_accounted_for(self):
        """A pyFEMM function is magnetics, another kind of problem, general or internal; a new
        one in femm.py fails here until it is placed."""
        unplaced = {n for n in self.pyfemm if not n.startswith(("mi_", "mo_") + OTHER_PHYSICS)}
        self.assertEqual(sorted(unplaced - GENERAL - INTERNAL), [])
        self.assertEqual(sorted((GENERAL | INTERNAL) - self.pyfemm), [])

    def test_every_magnetics_and_general_function_has_an_entry(self):
        wanted = {n for n in self.pyfemm if n.startswith(("mi_", "mo_"))} | GENERAL
        self.assertEqual(sorted(wanted - self.entries.keys()), [])

    def test_every_documented_command_has_an_entry(self):
        self.assertEqual(sorted(self.manual - self.entries.keys()), [])

    def test_no_entry_for_a_function_that_does_not_exist(self):
        """An entry names a pyFEMM function, except a console entry: a command the manual
        documents for FEMM's built-in console that pyFEMM has no function for."""
        for name, entry in self.entries.items():
            with self.subTest(name=name):
                if entry["source"] == "console":
                    self.assertIn(name, self.manual)
                    self.assertNotIn(name, self.pyfemm)
                else:
                    self.assertIn(name, self.pyfemm)

    def test_source_says_where_a_command_is_documented(self):
        """A documented command is "manual"; one the manual never names is "binding"; for one it
        names only in passing, either can be right, and the entry decides."""
        mentions = manual_mentions()
        for name, entry in self.entries.items():
            if entry["source"] == "console":
                continue
            with self.subTest(name=name):
                if name in self.manual:
                    self.assertEqual(entry["source"], "manual")
                elif name not in mentions:
                    self.assertEqual(entry["source"], "binding")

    def test_every_test_passed_in_femm(self):
        for item in lookup.items(self.data):
            with self.subTest(name=item["name"]):
                verified = item.get("verified")
                self.assertIsNotNone(verified, "never run: use verify/run_femm.py tests")
                self.assertEqual(verified["status"], "pass" if item["test"] else "not run",
                                 verified.get("observed", ""))

    def test_lookup(self):
        self.assertEqual(lookup.find(self.data, "MI_DRAWARC")["name"], "mi_drawarc")
        self.assertEqual(lookup.find(self.data, "torque_sign")["name"], "fact:torque_sign")
        top = [item["name"] for _, item in lookup.search(self.data, ["arc"])[:3]]
        self.assertIn("mi_drawarc", top)
        self.assertIn("maxseg", lookup.describe(self.entries["mi_drawarc"]))

    def test_cheatsheet_is_current(self):
        self.assertEqual((ROOT / "CHEATSHEET.md").read_text(), lookup.cheatsheet(self.data),
                         "regenerate it: python3 -I lookup.py cheatsheet")


if __name__ == "__main__":
    unittest.main()
