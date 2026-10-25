# Updating the second brain after a replication

The second brain grows only through tested data and generic components. After
recording a replication (stage 8 of `SKILL.md`), make these updates in the same
branch, then run `python3 -m femm_brain validate` and the host tests.

1. **Paper index** (`femm_brain/data/papers.json`): set the paper's `status`
   and `case` path. Add new papers here before starting them, with what a
   replication is planned from:
   - `id`, `set` and `status`;
   - `family`;
   - a `title` (or a `topic` until the exact title is checked);
   - `citation`, `url` and `access`;
   - `machine`, `teaches`, `known_gaps` and `targets`.

   Add the screening fields once the full text has been read: `geometry`,
   `target_sources`, and `software` when the paper names its tool.
   `python3 -m femm_brain papers` checks them.
2. **Case** (`femm_brain/data/papers/<id>.json`): the frozen inputs and targets
   stay as frozen; append `findings`, `geometry_checks`, `coverage` and the
   `retrospective`. If an input had to change after a solve because it was
   wrong, keep the earlier value in the finding that explains the change.
3. **Evidence** (`femm_brain/data/verification/paper_replications.json`):
   regenerate with `reproduction-report --catalog papers`. A failed acceptance
   target is accepted only through `reviewed_paper_failures.json`, bound to the
   case definition, the frozen target and the observed value.
4. **Components**: a new capability is a generic component (code, knowledge
   record, contract and geometry tests, and a synthetic design that checks it).
   Components become numerically checked in the graph only through passing
   assemblies, paper replications included.
5. **Lessons** (`lessons.md`): add what generalizes, citing the case. Remove or
   correct a lesson a case contradicts.
6. **Skill** (`SKILL.md`): change the procedure only when a replication showed a
   gap in it. Bump the version line and describe the change in the case's
   retrospective. Never change the skill during a held-out run.

Never commit papers or long excerpts, never edit another paper's frozen targets,
and never relabel a failure.
