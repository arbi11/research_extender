# femm_brain_daro_v2: block 1, a checked FEMM catalogue for Python

A from-scratch second brain for replicating electric machines in FEMM, built one block at a
time. Each block is a file or a small tool, and stays only if it helps on `daro_bench`.

**Block 1** is a lookup catalogue of FEMM 5's magnetics commands as ordinary Python reaches them
through femm-qt's pyFEMM (`from femm import *`), written for an agent scripting a machine's
geometry and results. It has an entry for each of the 192 magnetics and general
commands (every `mi_` and `mo_` function pyFEMM defines, its general functions, and the 5 console
commands the manual documents but pyFEMM lacks), and 18 facts about the engine and pyFEMM as a
whole: how a script starts, errors, how values cross between Python and FEMM, units, signs, the
air-gap band, DXF import, material names. The engine, not the manual, decides what an entry says.
Every entry carries a Python test that ran in FEMM (all 210 tests pass), and each note says where it
comes from: `[manual]`, `[binding]` (pyFEMM's `femm.py`) or `[tested]`.

**Why Python.** The FEMM 5 manual documents Python and keeps Lua only for old scripts, and pyFEMM
uses the same command names as FEMM 4.2's pyFEMM. The first edition of this block used Lua 4
(`femm-lua`). What it learned about FEMM itself carried over and was checked again in Python;
the Lua-only traps are gone.

## Use it

1. Start a script like this (in the FEMM image, run it with `/opt/venv/bin/python3`):

   ```python
   import sys
   sys.path.insert(0, "/opt/femm/share/femm/pyfemm")
   from femm import *
   openfemm(1)    # headless
   ...
   closefemm()
   ```

2. Read [CHEATSHEET.md](CHEATSHEET.md) before writing a model: the facts, where FEMM and pyFEMM
   differ from the manual and FEMM 4.2, the failures that raise no error, and every command's
   signature by topic.
3. Look things up while writing:

```bash
python3 -I femm_brain_daro_v2/lookup.py get mi_drawarc mo_getgapb   # full entries
python3 -I femm_brain_daro_v2/lookup.py get mi_drawarc --test       # plus the Python that checked it
python3 -I femm_brain_daro_v2/lookup.py search torque               # best matches
python3 -I femm_brain_daro_v2/lookup.py facts                       # the facts
python3 -I femm_brain_daro_v2/lookup.py topic draw                  # the commands in one topic
```

## What the tests found

**pyFEMM, the layer between Python and FEMM** (new with Python):

- FEMM's errors raise `Exception("FEMM returns: ...")`, wrong argument counts included. pyFEMM's
  own helpers with fixed parameters (`mi_drawline`, `mo_getb`, ...) raise `TypeError` instead.
- `None` reaches FEMM as the word None: a coordinate fails, but an optional or property value
  counts as left out.
- Strings are not escaped: a double quote inside one breaks the call, and a newline splits it
  into two commands.
- Every `i` and `I` in a reply becomes `1j` before Python reads it, so text results come back
  garbled (`"Air"` comes back as `'A1jr'`).
- Many wrappers return `None` even when FEMM replies; a single result is a bare value.
- The `mo_getall...` commands return everything in one call (the Lua binding overflowed above
  about 230 values).
- `mo_getb` and the other two-coordinate `mo_get...` shortcuts exist only in pyFEMM, not in
  FEMM's built-in Python.
- Commands pyFEMM has no function for (`clearconsole`, `mo_setfocus`, `open`, `print`, `quit`)
  are sent with `callfemm('...')`.
- One script per FEMM: a second `openfemm` while connected raises. Run another FEMM with
  `FemmSession(port=N)`.
- Relative paths resolve in the folder the script was in when pyFEMM connected; a later
  `os.chdir` does not move FEMM.

**FEMM itself, carried over from the Lua edition and checked again in Python:** the current
direction, results in SI units, the air-gap band (it needs a `<No Mesh>` label, and Air in the
band halves Bt), prescribed A, the torque sign, saving before solving, DXF import, femm-qt's
material names and arc segment sizes.

**New in this edition:**

- Every move, copy, rotate, scale or mirror resets the point property and group of every node,
  even with nothing selected. Set them after the last edit.
- Only the single-object selects (`mi_selectnode`, `mi_selectsegment`, `mi_selectarcsegment`,
  `mi_selectlabel`) toggle. Group, rectangle and circle selects only add. In the post-processor,
  block and group selects both toggle.
- `mo_gapintegral` type 5 kills FEMM (and type 6 too at AC); pyFEMM then raises "Connection to
  FEMM lost".
- Headless, `mo_close` frees nothing (about 6 MB per solution), and `mo_reload` leaves a freed
  copy behind that makes a later `mo_setfocus` kill FEMM. Use `mi_loadsolution`, and give each
  solution its own file name.
- The view commands do nothing headless, and `mi_savebitmap` writes a picture only from the GUI
  build.

**Gone with Lua:** `nil` silently cutting a call short, the reply-size limit, and Lua 4's own
language traps.

## Files

| File | What |
|---|---|
| `catalogue.json` | the catalogue: `about` (engine, sources and their SHA-256), `facts`, `entries` |
| `CHEATSHEET.md` | generated from the catalogue by `lookup.py cheatsheet` |
| `lookup.py` | get, search, topic, facts, cheatsheet, check |
| `AUTHORING.md` | how to add or change an entry and rerun its test |
| `verify/` | the FEMM runner, the Python harness and fixtures, the pyFEMM inventory, the manual index, and the batches of the last full run |
| `tests/` | host checks: schema, coverage of pyFEMM and the manual, FEMM results, lookup, cheat sheet |

## How it was checked

- **Engine**: FEMM 5 (femm-qt) in the image `research-extender-femm:dev-hierarchy-resumed-20261025`
  (`sha256:85f07dce…`), driven by the image's Python 3.12 through pyFEMM, headless, with no
  network, one CPU and 768 MB.
- **Coverage**: `verify/inventory.py` lists every function pyFEMM defines, and
  `verify/manual_index.py` lists every command the manual documents. The host tests require an
  entry for each, and check each entry's `source` against both.
- **Tests**: `verify/run_femm.py tests` runs every test in FEMM and writes the outcome (`pass`,
  `fail` or `not run`), what it observed, the engine and the date into the entry's `verified`
  field. Physics entries check against hand calculations: Ampere's law round a wire, the field
  5 mm from it, its resistance, flux linkage and AC voltage, the area of a meshed arc, the gap
  field of a uniform 1 T. The harness restarts FEMM after a test that closes it or crashes it,
  and starts a fresh FEMM every 25 tests, because headless FEMM keeps every solution in memory.

```bash
python3 -I femm_brain_daro_v2/verify/run_femm.py tests     # about 3 minutes, in FEMM
python3 -I femm_brain_daro_v2/lookup.py check
python3 -m unittest discover -s femm_brain_daro_v2/tests
```
