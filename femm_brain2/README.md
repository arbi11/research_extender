# FEMM API lookup catalogue

This first milestone contains ten reviewed FEMM 5.0 API entries for the Lua 4
binding. It is a flat lookup catalogue for an agent writing FEMM scripts.

SQLite is the working lookup database. Commands return JSON. The versioned
`data/commands.json` file is the reviewed import batch used to reproduce this
milestone; updates go through validation and a rebuild, rather than independent
edits to the database and import batch. Raw documentation stays in a local cache.

## Try it

Run these from `/home/akhan/code/research_extender`:

```bash
python3 -m femm_brain2 fetch-sources
python3 -m femm_brain2 validate
python3 -m femm_brain2 build
python3 -m femm_brain2 search "draw arc"
python3 -m femm_brain2 get mi_addarc
python3 -m femm_brain2 search "magnetic problem frequency" --physics magnetics
python3 -m femm_brain2 stats
```

The default database is `~/.local/share/femm-brain2/catalogue.sqlite3` (or the
equivalent under `XDG_DATA_HOME`). Every command accepts `--db /path/to/file`.
Source downloads use `~/.cache/femm-brain2/sources/`, respecting `XDG_CACHE_HOME`.
`fetch-sources`, `validate` and `build` accept `--cache`.

`get` and `search` support `--version 5.0 --binding lua4`. Search also accepts
`--physics` and `--limit`. Exact lookup returns the complete contract. Search
returns concise matches with signatures, review status and source references.
`export` writes the complete catalogue as JSON to stdout. `stats --all-pending`
includes the full list of detected manual commands awaiting review.

Lookup is read-only and works offline after the database is built. A missing
database produces an error with the build command. An inventory name without a
reviewed entry returns `not_reviewed`, rather than an invented API contract.

## What an entry contains

- Actual command name, FEMM version, binding, physics and a flat category.
- Purpose, normalized signature, ordered arguments, units, optional arguments,
  documented defaults and allowed values where supported by the source.
- Return values in order, preconditions and effects on the current model.
- Source URLs, document pages/sections, source hashes and definition hashes.
- Review status, unresolved questions and explicitly labelled example fragments.
- An independent execution status, currently `not_tested` for every entry.

The commands are `newdocument`, `mi_probdef`, `mi_addnode`, `mi_addsegment`,
`mi_addarc`, `mi_addblocklabel`, `mi_setblockprop`, `mi_analyze`,
`mi_loadsolution` and `mo_getpointvalues`.

This is **10 reviewed entries**, not complete FEMM coverage. The importer detects
558 distinct manual definitions; the remaining 548 are an inventory for later
work. This count describes this pinned manual's detected definitions, not a
claim that every alias, convenience wrapper or undocumented function is covered.
Repeated definitions retain their additional page locations in the inventory.

## Sources and who authored the entries

The official source set is pinned in [sources.json](data/sources.json):

1. [FEMM 5.0 User's Manual](https://femm.info/doku/lib/exe/fetch.php?media=femm-qt%3Adocs%3Amanual.pdf),
   dated October 1, 2026, SHA-256 beginning `e1273290fd7856a0`.
2. The FEMM project's Lua launcher guide and `femm.lua` binding, pinned to
   upstream revision `31ee4bf6a8cae1a1860c108e7934ab3403bfd6e3`.

The coding agent read the manual definitions and binding source and authored
the ten semantic records in [commands.json](data/commands.json). PDF text
extraction and command-name inventory detection are mechanical. The Python
validator checks record consistency and source evidence; it does not independently
prove that every agent-written interpretation is correct. Relevant manual pages
were also rendered and reviewed, including the return table that spans pages
108-109.

Full source text is cached locally rather than committed. Definition fingerprints
bind the reviewed records to the cited source passages without copying those
passages into the repository. Fresh downloads are accepted only if their hashes
match the pinned snapshots. Build validates the sources before atomically
replacing an existing catalogue belonging to this project.

Known limitations stay visible in lookups:

- `mi_addarc.maxseg`: the manual wording is ambiguous about subdivision semantics.
  Its units remain unspecified and its semantic review is `needs_clarification`.
  The Lua wrapper forwards the argument unchanged; an implementation review or
  isolated geometry check is still required to settle the interpretation.
- `mi_probdef`: the manual signature has a missing closing parenthesis. The
  normalized signature repairs that typography and separately marks optional
  arguments. Unstated optional defaults remain null.
- Dimensional units absent from the point-value return table remain null.
- Example snippets are agent-authored fragments with listed preconditions, not
  claimed runnable standalone programs or copied official examples.
- Documentation review does not establish availability or behaviour in every
  installed solver build. No FEMM executions are claimed in this milestone.

## Code map

| File | Responsibility |
|---|---|
| `source.py` | Fetch pinned sources, verify their hashes, and locate manual and Lua definitions |
| `data/commands.json` | The ten agent-authored, source-reviewed import records |
| `schema.sql` | SQLite catalogue and FTS5 search tables |
| `catalogue.py` | Validate/import records and perform read-only searches/lookups |
| `__main__.py` | JSON command-line interface |

There are no imports from the original `femm_brain` package. There is no component
library, hierarchy, paper reconstruction or background service in this milestone.

## Check the implementation

Python 3.12+, SQLite with FTS5, and Poppler's `pdftotext` for source validation
are sufficient; no third-party Python packages are required.

```bash
python3 -m unittest discover -s femm_brain2/tests -v
python3 -m femm_brain2 validate
```

The offline software tests use explicitly synthetic documentation to exercise
search, import rejection, provenance checks and database handling. The separate
`validate` command checks the real pinned sources. Neither is a FEMM physics test.
