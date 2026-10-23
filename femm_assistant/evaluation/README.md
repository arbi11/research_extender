# Live semantic and CAD evaluation

These commands run in an isolated development FEMM container. They do not connect
to the deployed assistant or its jobs volume. Keep the existing `.env`, Compose
settings, installed FEMM and deployed image tag. Use a separate development tag:

```bash
cd femm_assistant
docker build -t femm-assistant:dev .
mkdir -p evaluation-results
docker run --rm --user "$(id -u):$(id -g)" \
  femm-assistant:dev python -m evaluation.semantic --validate
```

The 28 public development cases cover complete and incomplete requests,
unsupported physics, action mismatches, prompt injection and faithful/unfaithful
edits. Each case has explicit labels and a rationale. All candidate plans also
pass the deterministic geometry schema. Labels are initial developer judgments;
independent review and an unseen holdout are still needed before choosing a
production threshold from these measurements.

## Measure real Jev decisions

Live evaluation uses the existing `OPENROUTER_API_KEY`, `DECISION_MODEL`,
`DECISION_BASE_URL` and `DECISION_THRESHOLD`. Blank model/threshold settings keep
the application defaults. The default cap is six sequential paid calls; use
`--max-cases 28` for the full seed set, or repeat `--case ID` to select cases.
Every case makes one production Decisions request, with no drafting or solve.

```bash
docker run --rm --user "$(id -u):$(id -g)" --memory=1g --cpus=1 \
  --env-file .env -v "$PWD/evaluation-results:/reports" \
  femm-assistant:dev python -m evaluation.semantic \
  --live --max-cases 28 --output /reports/semantic.json
```

Output permissions are checked before provider calls. Each completed case is
saved immediately. Choose a fresh output filename for each run; existing reports
are never overwritten. Reports contain case IDs, requested model, criteria and
dataset hashes, timestamps, probabilities, gate outcomes, errors, Brier scores,
reliability bins and threshold sweeps. Provider errors and unmeasured cases count
separately. Headers, API keys and environment contents are never recorded.
The requested model ID is recorded; a resolved provider revision is not known.

Recompute metrics from saved evidence without any provider request:

```bash
python -m evaluation.semantic --replay evaluation-results/semantic.json \
  --output evaluation-results/semantic-replayed.json
```

Replay requires the original dataset hash. Binary Brier scores use P(yes); the
route score is the sum of squared errors across its three classes, without
normalization. Route reliability uses the winning probability and whether its
label is correct; the separate provider confidence still participates in gating.
Threshold sweeps reuse these observations and never modify runtime settings.
Live/replay exits nonzero for gate-label mismatches or provider errors; a failed
calibration case remains useful evidence and is not silently reclassified.

## Test language through actual native execution

```bash
docker run --rm --user "$(id -u):$(id -g)" --memory=2g --cpus=2 \
  --env-file .env -e HOME=/tmp -v "$PWD/evaluation-results:/reports" \
  femm-assistant:dev python -m evaluation.cad \
  --live --output /reports/cad.json
```

This uses the production HTTP API and real OpenRouter responses. It creates a
round conductor, checks its physical values independently of Jev, meshes, solves,
checks field direction/magnitude and validates the exported ZIP. It then doubles
the current, repeats execution and checks quadratic energy scaling. Incomplete
depth and unsupported AC requests must stop at preflight without a draft.
`--create-only` runs just the creation diagnostic. At most six Decisions calls and
four draft calls occur in a full run, subject to the production deadlines and
6000-token draft limit. Public-prompt drafts are recorded for diagnosing fidelity
failures. No paid test runs automatically in CI.

## Independent numerical references

```bash
docker run --rm --memory=2g --cpus=2 \
  -e FEMM_LIVE_TEST=1 -e FEMM_MESH_EXPORT=1 femm-assistant:dev \
  python -m unittest discover -s tests -p 'test_*live.py' -v
```

The additional reference suite uses the thick-wire solution from
[OpenStax, Ampère's law, Example 12.7](https://openstax.org/books/university-physics-volume-2/pages/12-5-amperes-law):
inside a uniform wire, `B = mu0 I r / (2 pi a²)`; outside,
`B = mu0 I / (2 pi r)`. Probes test both vector components, several radii and
quadrants. Separate runs vary mesh size and outer-domain radius. A coaxial case
adds a nested air cavity and equal opposite return current: in the return shell,
`I_enclosed = I [1 - (r²-b²)/(c²-b²)]`; outside it the ideal field is zero.

These are planar infinite-length, linear-material symmetry references. A centered
circular outer boundary preserves that symmetry; agreement across its radii is
not evidence of general open-boundary convergence. Assertions use explicit
tolerances and print actual measurements in CI. They do not establish accuracy
for arbitrary geometry, nonlinear materials or finite-length conductors.
