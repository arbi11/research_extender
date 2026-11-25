# Adding or changing an entry

Each entry in `catalogue.json` describes one pyFEMM command as this engine really runs it, and
carries a Python test that FEMM ran. Facts (`fact:...`) describe the engine and pyFEMM as a whole.

## How Python reaches FEMM

Scripts are ordinary Python 3 (in the image, `/opt/venv/bin/python3`, version 3.12, without
numpy) using femm-qt's pyFEMM, `femm.py`:

```python
import sys
sys.path.insert(0, "/opt/femm/share/femm/pyfemm")
from femm import *
openfemm(1)          # 1 = headless
...
closefemm()
```

pyFEMM turns each call into a line of text, such as `mi_addnode(1.5,2)`, and sends it over TCP
to FEMM, whose built-in Python runs it. Chapter 3 of the manual documents those built-in
commands; `femm.py` decides how arguments and results cross over.

## One entry

```json
{
  "name": "mi_addnode",
  "topic": "draw",
  "source": "manual",
  "call": "mi_addnode(x, y)",
  "does": "Adds a node (a point) at (x, y).",
  "args": ["x, y: position, in the length unit set by mi_probdef"],
  "returns": null,
  "notes": ["Segments and arcs only connect existing nodes; mi_drawline adds its own. [manual]"],
  "test": "fixture_pre()\nmi_addnode(40, 40)\nx, y = mi_selectnode(40, 40)\ncheck(x == 40 and y == 40, show(x, y))\nreturn 'node selected back at ' + show(x, y)"
}
```

- **name**: the pyFEMM function name; for a `console` entry, the manual's command name.
- **topic**: one of `general document draw select edit assign materials boundaries circuits
  points problem mesh solve file view results contour gap mesh-data`.
- **source**: `manual` (documented in the FEMM 5 manual), `binding` (only pyFEMM's `femm.py`
  defines it), or `console` (the manual documents it for FEMM's built-in console, but pyFEMM
  has no function for it; the entry says how to reach it).
- **call**: the signature as pyFEMM takes it. Optional arguments go in square brackets, for
  example `mi_analyze([flag])`. Most wrappers forward whatever they are given, so FEMM decides
  what it accepts. A `console` entry's call is the `callfemm(...)` or `callfemm_noeval(...)`
  line that reaches the command, for example `callfemm('clearconsole()')`.
- **does**: one sentence in your own words.
- **args**: short strings of the form `"name: meaning, unit, allowed values"`.
- **returns**: what Python gets back: one value, a list for several (`[x1, x2, y1, y2]`), or
  `null` when the wrapper returns nothing. Say so when the wrapper drops what FEMM sent.
- **notes**: pitfalls and facts an agent writing a model needs. End each note with where it
  comes from: `[manual]`, `[binding]` (`femm.py`) or `[tested]` (the test shows it). Say so
  plainly when the manual, pyFEMM and the engine disagree.
- **test**: the body of a Python function (see below), or `null` plus `"untested_because": "..."`.
- **verified**: written by `verify/run_femm.py tests`; do not edit it by hand.

A fact has `name` (`fact:...`), `says` and `test`.

## Rules

- **Paraphrase.** The manual is copyrighted: `does`, `args`, `returns` and `notes` are your own
  words and shorter than the manual. Only `call` reproduces a signature. pyFEMM (`femm.py`,
  MIT licence) can be quoted briefly.
- **The engine is the truth.** When the manual, pyFEMM and a test disagree, the entry follows
  the test and a note records the disagreement.
- **Be brief.** An agent reads these while writing a model.

## Tests

A test body is the body of a Python function. It passes when it raises nothing and returns a
value, usually a short string saying what was seen. From `verify/py/harness.py`:

- `check(cond, message)` fails the test unless `cond` is true;
- `near(a, b, rel=1e-6)` compares within a relative tolerance;
- `show(*values)` joins values into one string;
- `fails(f, *args)` calls `f(*args)` and returns the text of the error it raises, or `None`.

`math`, `os`, `re`, the `femm` module and every pyFEMM command are in scope. Write no
multi-line string literals: the harness indents the body, which would change them; build text
with `"\n".join([...])`. Save files only under `/tmp`.

What pyFEMM does to values (see the facts for the tested details):

- FEMM's errors raise `Exception("FEMM returns:\n<message>")`.
- Several results come back as a list, one as a bare value, none as `[]`. Many wrappers return
  nothing at all, even when FEMM sends a reply.
- Every `i` and `I` in a returned value is turned into `1j` before Python reads it, so text
  results containing those letters come back garbled.
- Arguments are sent with `str()`: strings go inside double quotes with no escaping, and
  `None` reaches FEMM as `None`. A required number then fails; an optional or property value
  counts as left out.

Fixtures (`verify/py/fixtures.py`). Lengths are in mm, the problem is planar, 10 mm deep:

- `fixture_pre()`: a fresh unsolved model with
  - an outer circle r = 50 with boundary `A0` (A = 0)
  - a wire circle r = 1 at the origin: material `Copper`, circuit `Iw` = 100 A series, group 1
  - circles r = 10 and r = 11 with the air-gap boundary `AGap`, and a `<No Mesh>` label at (0, 10.5)
  - labels for inner air at (5, 0) and outer air at (30, 0)
  - an `Iron` square with corners (23,-2) and (27,2), label (25,0), group 2
- `fixture_post()`: the same model without the iron square, solved, with its solution loaded.
- `blank(depth=10)`: an empty mm document with the material `Air` and the boundary `A0`.
- `solve(name)`: saves to /tmp/<name>.fem, solves and loads the solution.
- Helpers: `circle(r, maxseg)`, `label(x, y, material, circuit, turns, group, meshsize)` and
  `arcprop(x, y, maxseg, boundary)`.

Start every test from a fixture or `blank()`. Tests run in the harness's headless FEMM. Only a
command that needs the window (a picture, such as `mi_savebitmap`) may start the GUI build, as a
second `femm.FemmSession(port=N, headless=False)` that the test closes in a `finally`. After each
test the harness closes the documents and, if the test closed FEMM or FEMM died, starts it again. Check a real effect when it is
cheap: a returned value, `mi_selectnode` or `mi_selectlabel` (they return the coordinates they
selected), or a solve followed by an area (`mo_blockintegral(5)`, in m^2) or a field value.

## Running the tests

Each run uses the pinned engine image with no network, one CPU at low priority and 768 MB of
memory. A batch takes a few seconds.

```bash
# try a draft: a JSON list of entries in any file, without touching the catalogue
python3 -I femm_brain_daro_v2/verify/run_femm.py entries draft.json [--only NAME ...]
# run the catalogue's tests and write each outcome into its "verified" field
python3 -I femm_brain_daro_v2/verify/run_femm.py tests [--only NAME ...]
# then regenerate the cheat sheet and run the host checks
python3 -I femm_brain_daro_v2/lookup.py cheatsheet
python3 -m unittest discover -s femm_brain_daro_v2/tests
```

## Sources

| What | Where |
|---|---|
| FEMM 5 manual | `/opt/femm/share/femm/manual.pdf` in the image; its text is cached as `~/.cache/femm-brain/artifact_texts/e1273290….txt`. Chapter 3 documents the built-in Python commands: §3.5 common commands, §3.6 magnetics pre-processor, §3.7 post-processor |
| pyFEMM | `/opt/femm/share/femm/pyfemm/femm.py` in the image, and its guide `pyfemm.md` beside it |
| pyFEMM function list | `verify/results/inventory.txt`, from `verify/inventory.py` |
| Manual index | `verify/results/manual_index.json`, from `verify/manual_index.py` |

The checksums of every source are in `catalogue.json` under `about.sources`.
