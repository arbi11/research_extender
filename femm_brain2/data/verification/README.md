# Native geometry evidence, November 25, 2026

These are real saved FEMM models, Lua fixtures, logs and hash manifests. They
check the geometry API in one pinned container with the exact reviewed Lua
binding. They do not validate a magnetic field solution or replicate a paper.

| Run | Frozen commit | Outcome |
| --- | --- | --- |
| `geometry-20261125-run1` | `881950d` | 14/16 scenarios passed; original failures retained |
| `geometry-20261125-run2` | `ad3efa1` | 16/16 scenarios passed, covering 33 geometry commands |

The first run exposed two errors in the agent-authored checker expectations:

- Property assignment checked segment row 0, assuming it was the lower edge of
  a rectangle. FEMM saved that edge as row 3. The corrected checker identifies
  the edge by its endpoints and verifies the same frozen mesh size, hide flag
  and group values.
- The mirror expectation assumed that the original line would be removed.
  The pinned C++ `DocumentBase::MirrorSelected` explicitly implements mirroring
  as a copy. The corrected contract and expectation retain both lines.

Both corrections were committed before the second native run. No coordinates,
physical properties or tolerances were changed. The first failed run, its
original protocol and all its model files are preserved here.

To rebuild the lookup with verified evidence:

```bash
python3 -m femm_brain2 build --evidence femm_brain2/data/verification/geometry-20261125-run2/report.json
```

To repeat the native checks, select an empty output folder:

```bash
python3 -m femm_brain2 check-geometry --output /tmp/femm-brain2-new-run
```

The manifest records the actual original Docker invocation and output path.
The files copied here are byte-identical artifacts. Full downloaded binding
source stays in the local cache and is not included in this evidence directory.
