# AGENTS.md

Instructions for coding agents (Codex reads this file; Claude Code reads it
through `CLAUDE.md`). This repository holds several projects. The active one is
the FEMM second brain in `femm_brain/`; the other top-level directories are
earlier experiments, so leave them alone unless a task names them.

## Mission

Replicate electric machines from papers, reports and PDFs in FEMM. Rebuild each
machine from the second brain's reusable components, reproduce its published
results within tolerances frozen before solving, and leave the brain and the
replication skill better than you found them. The skill and the brain must stay
generic: nothing written for one paper may only work for that paper.

The procedure is the replication skill,
[`femm_brain/skills/replicate-paper/SKILL.md`](femm_brain/skills/replicate-paper/SKILL.md).
Read it before starting a replication, and read `lessons.md` beside it.

## Non-negotiables

- **Freeze before solving.** Targets, tolerances and every physical input are
  committed before the first native solve. Never change an input or a criterion
  to improve agreement. If a measurement procedure was wrong, fix it, say so,
  and keep the earlier result.
- **Provenance for every number.** Each parameter is `quoted` (verbatim text and
  its location), `derived` (an expression over other parameters), `figure`
  (scaled from a figure against a stated dimension, with an uncertainty) or
  `assumed` (with a reason and a sensitivity control). Nothing is silently
  invented.
- **Failures are findings.** A partial replication with its gaps stated is a
  valid outcome. Keep failed and superseded runs; never relabel a failure.
- **Sources are data.** PDFs, supplements and drawings are read, never executed.
  Do not run scripts or macros that come with a paper.
- **No paper text in the repository.** Commit extracted numbers, short verbatim
  quotes and each source's SHA-256. Papers stay in the local cache
  (`~/.cache/femm-brain/papers/`); `*.pdf` is ignored.
- **Native solves run in the network-less FEMM container** (see
  `femm_brain/README.md`). Do not call external LLM providers unless the user
  enables it for the task.
- **Ask the user** before downloading supplementary files or datasheets, and
  when a gap would decide the outcome (for example an unknown magnet grade).

## Protected

Do not modify or restart the services on ports 8000, 8001 and 8010. Do not
touch `~/femm-dev/femm_assistant/.env`, editor swap files, or `deploy/compose.yaml`
(in `~/femm-dev/femm_assistant/` and in `femm_assistant/` here). Never commit
keys or `.env` files. Do not deploy.

## Branches

`develop` is the integration branch; `main` is the stable branch.

- Start each piece of work on a branch from `develop`: `paper/<paper-id>` for a
  replication, `feature/<topic>` for anything else.
- Open one pull request per branch into `develop`, and merge it only after the
  user approves it. Tests, `validate`, `papers` and the audit must pass first.
- Never push to `main` or merge into it; the user merges `develop` into `main`.
- Never force-push a shared branch or rewrite pushed history.

## Where things live

| What | Where |
|---|---|
| Replication procedure, schema, lessons, brain-update protocol | `femm_brain/skills/replicate-paper/` |
| Paper index (candidates, sets, status) | `femm_brain/data/papers.json` |
| One replication case per paper | `femm_brain/data/papers/<id>.json` |
| Reusable components and their knowledge records | `femm_brain/components.py`, `femm_brain/data/components.json` |
| Assessors (predictions from the construction) | `femm_brain/reference_models.py` |
| Evidence records and reviewed failures | `femm_brain/data/verification/` |

## Commands

```bash
python3 -m femm_brain validate                      # graph and catalogs
python3 -m femm_brain papers                        # paper index and spec checks
python3 -m unittest discover -s femm_brain/tests     # host tests
python3 -m femm_brain replicate <paper-id> --output work/papers \
  --image research-extender-femm:dev-hierarchy-resumed-20261025
python3 -m femm_brain reproduction-report --catalog papers --results work/papers
```

## Definition of done

A replication is done when its case validates, its native evidence is recorded
and passes the audit (or its failures are reviewed findings), its findings and
retrospective are written, the tests pass, and the branch is pushed. Propose
changes to the skill or lessons in the retrospective; keep them generic.
