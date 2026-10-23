# Language-to-CAD, visual and numerical evaluation

Run live evaluations in a separate development FEMM container. They use real
OpenRouter responses and native FEMM where specified, without connecting to the
deployed assistant or its jobs volume. Preserve the existing `.env`, local
Compose settings and image tag; keep the old laptop unchanged until production
deployment. Use a separate development tag:

```bash
cd femm_assistant
docker build -t femm-assistant:dev .
mkdir -p evaluation-results
```

Keep the API key in the existing untracked `.env`; do not put it in CLI arguments,
reports or Git. Paid evaluations require an explicit `--live` flag. Use a fresh
output filename for every run. These commands describe tests to run; they do not
claim a passing result for the current commit.

## Repeated language-to-CAD execution

```bash
docker run --rm --user "$(id -u):$(id -g)" --memory=2g --cpus=2 \
  --env-file .env -e HOME=/tmp -e FEMM_MESH_EXPORT=1 \
  -v "$PWD/evaluation-results:/reports" \
  femm-assistant:dev python -m evaluation.batch \
  --live --rounds 2 --max-cases 28 --workers 2 --output /reports/batch.json
```

With all 28 public cases selected, two rounds plan 56 case attempts. Every case
uses an isolated application/job directory with the normal production function
route, deterministic validation and bounded repair. The evaluator compares typed
physical assignments independently, renders previews, and runs native
mesh/solve/export for executable cases. Clarification cases must return questions
without a model. The worker count bounds concurrent case attempts; it does not
change the application's single-worker queue design.

Reports checkpoint each attempt, including failures and provider errors, and
retain evidence under the adjacent `batch-artifacts` directory. They include
stage outcomes, model IDs, dataset hash, latency and round summaries. Failed
attempts are never replaced by whole-case retries. The production limit of two
drafts, one logical route and one logical actual-error diagnostic per planning
request still applies. Each Jev logical call can make at most two transport
attempts after connection or timeout failures within the same overall deadline;
this is not a whole-case retry or a score gate. Passing execution is not proof of numerical convergence or engineering
accuracy. The public seed set is not an independently labeled holdout.

The script defaults to two rounds, twelve cases and two workers. Use the explicit
command above for 28 cases. `--rounds`, `--max-cases` and `--workers` are bounded;
`python -m evaluation.batch --help` lists the current options.

## Optional visual-model comparison

```bash
docker run --rm --user "$(id -u):$(id -g)" --memory=2g --cpus=2 \
  --env-file .env -e HOME=/tmp -e FEMM_MESH_EXPORT=1 \
  -v "$PWD/evaluation-results:/reports" \
  femm-assistant:dev python -m evaluation.vision_benchmark \
  --live --output /reports/vision-benchmark.json
```

The default run selects six visual fixtures and two image-input models:
`qwen/qwen3-vl-30b-a3b-instruct` and `google/gemini-3.1-flash-lite`, planning 12
provider calls. Repeat `--model MODEL_ID` up to twice to select models, and use
`--max-cases` to change the bounded fixture count. Fixture setup failures and
provider errors remain in the report; planned calls are not passing results.

All six benchmark fixtures are geometry-only: the reviewer receives the source
request, typed plan and rendered geometry PNG. No native mesh is supplied by this
benchmark. Its cases cover a correct wire, wrong wire radius, a correct asymmetric
core, a missing return winding, reversed return turns and a fully mirrored model.
Because the typed plan accompanies the image, this does not measure image-only
reasoning; reversed turns may be indistinguishable in the image alone. Reports
retain exact fixtures and rendered PNGs, advisory findings, latency and actual
provider-reported usage. Missing usage remains unknown. A match does not certify
geometry, mesh quality, physical fidelity or numerical accuracy. Native-mesh
image review is tested separately through the manual application action, using
an actual completed, passed mesh for the exact current specification.

This comparison does not change the application's configured `VISION_MODEL`.
The manual application action and new-installation drafting default to
`google/gemini-3.1-flash-lite`; existing explicit `AI_MODEL` and `VISION_MODEL`
values remain overrides. Setting `VISION_MODEL=` disables that optional action.
Visual findings and provider failures never disable mesh or solve.

## Focused native CAD acceptance

```bash
docker run --rm --user "$(id -u):$(id -g)" --memory=2g --cpus=2 \
  --env-file .env -e HOME=/tmp -e FEMM_MESH_EXPORT=1 \
  -v "$PWD/evaluation-results:/reports" \
  femm-assistant:dev python -m evaluation.cad \
  --live --output /reports/cad.json
```

This smaller production-API test creates a round conductor, checks its physical
values, meshes, solves, compares the field and validates the exported ZIP. It
then doubles current, repeats execution and checks energy scaling. Incomplete
depth and unsupported AC requests must ask for clarification without producing a
model; the draft model can supply those questions. `--create-only` limits this
to the initial creation case. Jev only selects a function or diagnoses an actual
failed call; it does not approve any of these results.

## Historical semantic reports: offline only

`evaluation.semantic` preserves the retired preflight/fidelity dataset and old
reports for historical replay. It has no provider client, live collection or
`--live` option. The original 28-case Jev evidence and later asymmetric-core
cases describe the earlier approval policy, not current behavior.

Validate the historical labels and typed candidate plans without any API call:

```bash
python -m evaluation.semantic --validate
```

Replay a saved historical report with its original dataset:

```bash
python -m evaluation.semantic --replay evaluation-results/semantic.json \
  --cases path/to/the-original-semantic_cases.json \
  --output evaluation-results/semantic-historical-replay.json
```

The original dataset hash must match. Omit `--cases` only if the saved report used
the current historical dataset. Replay retains legacy Brier scores, reliability
bins, false accepts/rejections and threshold sweeps; output is explicitly marked
historical. Missing/error cases remain unmeasured. The legacy trace schema is
separate from production function-route traces. No replay threshold controls
current geometry preview, meshing or solving. Existing `DECISION_THRESHOLD`
settings are accepted and ignored by production routing.

## Independent numerical references

```bash
docker run --rm --memory=2g --cpus=2 \
  -e FEMM_LIVE_TEST=1 -e FEMM_MESH_EXPORT=1 femm-assistant:dev \
  python -m unittest discover -s tests -p 'test_*live.py' -v
```

The reference suite uses the thick-wire solution from
[OpenStax, Ampère's law, Example 12.7](https://openstax.org/books/university-physics-volume-2/pages/12-5-amperes-law):
inside a uniform wire, `B = mu0 I r / (2 pi a²)`; outside,
`B = mu0 I / (2 pi r)`. Probes test both vector components, several radii and
quadrants. Separate runs vary mesh size and outer-domain radius. A coaxial case
adds a nested air cavity and equal opposite return current: in the return shell,
`I_enclosed = I [1 - (r²-b²)/(c²-b²)]`; outside it the ideal field is zero.

These are planar infinite-length, linear-material symmetry references. A centered
circular outer boundary preserves that symmetry; agreement across its radii is
not evidence of general open-boundary convergence. Assertions use explicit
tolerances and record measurements. They do not establish accuracy for arbitrary
geometry, nonlinear materials or finite-length conductors. Native tests require
the live flag above; a passing unit/API run with skipped live tests does not
verify FEMM execution.

For application and browser verification, use the commands in the main
[README](../README.md#verification). Browser fixtures check UI behavior and stage
prerequisites; actual provider and numerical results come from separate reports.
