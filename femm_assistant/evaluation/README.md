# Language-to-CAD, visual and numerical evaluation

## Audited hierarchical model matrix

Run in an isolated development FEMM Docker image with the existing environment
file, a fresh report directory and `FEMM_MESH_EXPORT=1`:

```bash
python -m evaluation.hierarchy_live --live \
  --cases evaluation/hierarchy_cases.json --output /reports/hierarchy-matrix
```

The defaults are `~z-ai/glm-latest`, `anthropic/claude-haiku-5.5` and
`deepseek/deepseek-v4-pro-0813`. Override them with `--models MODEL_A MODEL_B
MODEL_C`. Every case/model pair has an independent application, job database and
native worker; the deployed services and their settings are not used. This runner
adds no output-token cap, reasoning override, model read deadline, external retry,
physical correction or target fitting. The configured native solver timeout
remains an execution limit. `--live` is mandatory before environment access.

Protocol v3 records task cancellation explicitly as scientifically unassessed,
locks the partial audit/report/manifest, and re-raises cancellation so no next case
runs. Interrupted responses retain the decoded prefix, labelled incomplete;
unreported costs remain unknown. An actual provider quota/payment 402 or exhausted
usage/budget 403 creates a stop marker and blocks subsequent provider requests and
cases. Other access/schema errors do not masquerade as quota exhaustion. This
reacts to the provider response and adds no model token or reasoning cap.

For separately launched model processes, mount one shared writable directory and
pass the same `--cohort-stop /reports/shared/quota-stop.json` to each. The first
confirmed limit publishes an immutable marker; every later provider request checks
it before transport. Requests already in flight can still incur charges. Use a
fresh marker path for a separately authorized cohort; a pre-existing marker stops
the cohort before any provider request. Without this option a matrix uses its own
output directory's `cohort-stop.json`. The summary records unstarted cases and the
structured stop reason rather than assigning them scientific failures.

The supplied six cases are public development regression inputs, not blind
holdouts. Their independent audit criteria are in `hierarchy_case_audit.json`,
which the runner never reads. A new cases JSON file contains only request objects:
`id`, `prompt`, optional `pdf` (relative to that file), optional `source_assets`,
and optional `target_selection` (`case_id`/`target_ids`, never target values).
Do not put source gold or expected answers in the case requests.

Paper cases use upload, hierarchical extraction, typed feature planning,
immutable revisions and published targets frozen before both mesh and solve.
Protocol v2 accepts explicit nonzero published analytical, numerical and
experimental reference values from the selected case, preserving the original
method, value, units, conditions and citations. A formula computed by the runner,
an invented target or a target from another case is ineligible. Target checks
retain their reference method; agreement with an analytical reference does not
establish reproduction of a published FE result. The policy is recorded in each
protocol and trial before execution. Earlier v1 cohorts and their numerical-only
selection remain unchanged. Missing circuit identifiers are rebound only when
exactly one native circuit exists. Ambiguities, incomplete evidence, actual
function failures and unsupported physics retain their original outcomes.
Assemblies and previews remain available when execution is unsupported.

Hierarchical drafting uses separate strict JSON schemas for inventory, physical
facts/settings, exterior geometry, each component and each custom or modified
feature. A feature request selects one typed shape contract; the full assembly
is composed and checked locally. Templates, closed paths, patterns, transforms
and unsupported intent share the same editable graph. There is no ordinary-JSON
fallback or monolithic assembly response schema.

After an actual wire or local validation error, only that stage may receive one
Jev diagnostic and one repair with the precise errors and unchanged successful
checkpoints. A diagnostic is advisory; its failure does not prevent that bounded
repair, and its answer cannot approve invalid geometry. Provider grammar,
configuration, access, budget and rate-limit failures stop without a physical
repair. Reports retain failed/successful attempts and diagnostic usage when
returned; missing usage stays unknown. Final compiler failures remain blocked
assemblies with inspectable intent. Score the two clarification/unsupported NL
cases by their expected disposition, separately from native-execution success.

Source image comparison is opt-in through `source_assets` entries such as
`{"page": 8, "purpose": "table", "label": "Comparison table"}`. Crops use
`bbox` in the source API's displayed PDF-point coordinate system. Select an
explicit shared image-capable `--vision-model MODEL`; the requested drafting
model remains unchanged. Add `review_mesh: true` to attach the actual passed
native mesh. `focused_checks: true` asks the configured Jev model for additional
image questions and labels that multimodal role experimental. These observations
report discrepancies; they do not approve physical fidelity or replace native
checks.

Each trial saves every API result, complete decoded/redacted provider request and
response, provider/model identity where returned, reasoning/token usage, cost,
latency, native artifacts and a SHA-256 manifest. HTTPX decodes compressed bodies
once and caches the unchanged application bytes; the audit does not claim TLS or
original compressed-byte capture. Unknown cost stays unknown. Reports distinguish
`execution_pipeline_passed` from `source_fidelity: pending_independent_audit`:
successful execution alone is not a paper-reproduction or intent-fidelity verdict.

## Paper replication and controlled recovery

`python -m evaluation.replication --output /reports/lubin --protocol near_field_v3`
runs three independently transcribed cases from
[Lubin, Berger and Rezzoug (2012)](https://www.jpier.org/PIERB/pier.php?paper=12051105):
centred iron-core inductance, displaced-coil axial force and air-core inductance.
Each case uses base/refined/far-boundary profiles. It freezes the paper targets
before native work, audits the actual FEMM file and compares native CSV results.
All targets use 2%; mesh/domain changes must be below 0.5%. The explicit v3 energy
diagnostic is 1e-4 relative because FEMM's axisymmetric energy and circuit-linkage
quadratures differ at finite mesh. The pinned engine source and rationale are
recorded. This correction does not change paper values or their 2% tolerances.

`--audit-existing /reports/previous/report.json --output /reports/new-audit`
verifies artifact hashes and rereads native CSV/spec files under the new protocol
without rerunning FEMM. Original failures and report hashes remain available;
re-audit time is not reported as native solve time. Gold fixtures are independent
reference inputs, not evidence that AI read the paper correctly.

The separate `python -m evaluation.paper_live --live --pdf /papers/source.pdf
--output /reports/paper-live` uses the real upload/extraction/planning APIs and
actual native execution. It makes one extraction request and one normal bounded
planning request, retaining all failures and questions. Source gold numbers are
used only after model generation for an independent audit. Numerical domain/mesh
choices are explicitly supplied as user instructions. A source-validation stop
is a failed trial, not a successful paper reproduction.
`--resume-evidence /reports/previous/report.json` revalidates and reuses that
attempt's real extraction without another extraction call. It links the original
report digest and retains the failed planning attempt. This continuation is
reported separately from a wholly fresh end-to-end trial. The runner requests
the opt-in single-winding current-density arithmetic check; a mismatch uses the
normal one-diagnostic/one-repair planning bound.

`python -m evaluation.recovery_loop --output /reports/recovery` injects a current
error into a known reference case and tests frozen-target failure and checkpoint
recovery with real native FEMM. Add `--live` and the existing env-file to exercise
one Jev diagnosis and one draft repair proposal. The deterministic mode makes no
provider calls. Neither mode establishes comparative resource savings; retain
the failed branch, invocation counts, timings and unknown usage as unknown.

These commands run inside the isolated development image with fresh report
directories and `FEMM_MESH_EXPORT=1`, as in the examples below.

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
attempts are never replaced by whole-case retries. This legacy flat-planner batch
uses at most two drafts, one logical route and one logical actual-error diagnostic
per planning request. The hierarchical matrix above instead applies its bounded
repair to the individual failing stage while retaining successful checkpoints.
Each Jev logical call can make at most two transport attempts after connection
establishment failures; there is no default overall AI read deadline. These
connection attempts do not restart a case or approve geometry. Passing execution
is not proof of numerical convergence or engineering
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
