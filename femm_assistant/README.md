# FEMM 5 Assistant

A small Linux application for describing, previewing, solving, and exporting
**2D planar DC magnetic models**. It lives alongside the book and does not
install the repository's machine-learning training dependencies.

General geometry mode builds new models from circles, rectangles and simple
polygons, or modifies an existing model plan. A separate C-core example retains
the original numeric form and solver. Axisymmetric models, AC analysis,
permanent magnets, nonlinear B-H curves, shared-edge Boolean operations and
full 3D CAD remain outside this version.

The [implementation pipeline](PIPELINE.md) records the implemented stages and
remaining validation and physics extensions.

The [evaluation guide](evaluation/README.md) runs labeled live OpenRouter checks,
and language-to-CAD execution in a separate
development Docker image while preserving deployment settings and jobs.

## What a reader can do

1. Describe a new geometry or a modification in natural language, load a round-wire
   example, or use the C-core numeric form.
2. Inspect the geometry and assumptions, then select **Check mesh**. Inspect the
   actual triangular mesh, counts, minimum angle, domain coverage and region labels.
3. Select **Solve** after the mesh check passes. Inspect energy, circuit quantities, field
   probes, optional force and a sampled magnetic-field plot.
4. Download a ZIP containing an editable Python CAD script `model.py`, native Lua
   bridge `model.lua`, compatibility launcher `run.py`, `spec.json`, previews and
   available mesh/solve artifacts, field CSV, quantities and solver log.

Source-only downloads work before a solve and without an AI credential. They do
not contain a FEMM model/solution or numerical results. The portable scripts need
FEMM 5 but need no repository checkout, third-party Python package or AI API key.
Edit the `MODEL` dictionary in `model.py`; its named Python CAD stages regenerate
the complete Lua construction, material, boundary, mesh, solve and extraction
sequence. Run `python3 model.py`, `python3 model.py --mesh-only`, or
`python3 model.py --write-only`; `python3 run.py` invokes the same pipeline.

The general-geometry backend requires a passed mesh job for the exact current
specification before accepting a solve. Any model edit requires another mesh
check. This gate detects invalid mesh structure and missing coverage; it is not
proof of field accuracy or convergence. The separate C-core numeric form retains
its original solve flow.

## Run locally, without Docker

Use Python 3.12 or newer in a separate virtual environment:

```bash
cd femm_assistant
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000 --workers 1
```

Open <http://127.0.0.1:8000>. Without FEMM installed, previews and source exports
work; the UI identifies the unavailable solver and disables solving. Without an
AI key, the numeric form remains usable and the natural-language control
is disabled. Local development has no access token by default; keep it on loopback.

For actual solves, build/install [FEMM 5](https://femm.info/doku/doku.php?id=download5)
using its [Linux instructions](https://femm.info/doku/doku.php?id=bindingsetup).
Put `femm-lua` on PATH, or set `FEMM_LUA` to its full launcher path. General-mode
mesh checks also need the opt-in native mesh export patch included by the Docker
build; stock FEMM 5 keeps this pre-solve mesh in memory. Portable model scripts
can still run with stock FEMM 5, but detailed diagnostic mesh files require the
patch. This application uses FEMM 5's bundled Lua launcher and its headless engine.
It does not use the
old PyPI `pyfemm` package, Wine, a display server, or Windows automation.

To enable natural-language requests, set the API key before starting:

```text
OPENROUTER_API_KEY=your-key-kept-on-the-server
```

Only the API key is required for the default AI configuration. The draft model
defaults to `openai/gpt-4.1-mini` and the semantic decision model to
`typesafe/jev-1.14`; blank model settings use these defaults. Optional overrides
are `AI_MODEL`, `DECISION_MODEL` and `DECISION_THRESHOLD` (default `0.90`). For
ordinary Docker use, add only the key to the existing `.env`; the supplied Compose
file forwards the key and `AI_MODEL` and uses the decision defaults. Custom
decision settings also need explicit forwarding in the Compose `environment`
section; adding them to `.env` alone does not forward them. Plain `uvicorn` uses
exported environment variables for all these settings.
`AI_BASE_URL` defaults to OpenRouter's API. Keep the key on the server, out of Git
and chat. Requests send the prompt, current plan when present, and proposed plan
for the fidelity check to the provider.

The AI returns a structured model plan rather than executable Lua or Python.
The application validates contours, region labels, material/circuit references,
physics, and mesh budgets, then compiles the accepted plan. General AI planning
requires Jev semantic preflight and a fidelity check of the resulting plan through
OpenRouter's Decisions API. Jev probabilities concern intent, not numerical
accuracy; the `0.90` threshold is provisional until calibrated on labeled cases.
Arithmetic, topology, validation and stage gates remain deterministic. Missing
physical inputs or unsupported requests produce clarification questions. A
request allows at most two Jev calls and two draft-model calls, including one
validation-guided repair capped at 6000 generated tokens per draft, within a
90 second total deadline. Review the proposed model before checking its mesh.

## Deploy on a single Linux VPS

Start with an x86-64 Ubuntu 24.04 VPS with Docker Engine and Compose installed.
2 vCPUs / 4 GB RAM is a **benchmark starting point**, not a measured capacity
guarantee. The initial native FEMM compilation can need more memory than a solve;
build the image on a larger machine and transfer it if a small VPS cannot build it.

From a checkout containing this addition:

```bash
cd femm_assistant
cp .env.example .env
```

Edit `.env`: replace `ASSISTANT_TOKEN` with a long random value. Add your API key
if you want natural-language requests; default models need no additional setting.
Keep this file out of Git.
Compose reads this file; plain `uvicorn` does not load it automatically.

```bash
docker compose --env-file .env -f deploy/compose.yaml up --build -d assistant
docker compose --env-file .env -f deploy/compose.yaml logs -f assistant
```

The service initially binds only to `127.0.0.1:8000` on the VPS. For private use,
forward that port from your computer:

```bash
ssh -L 8000:127.0.0.1:8000 your-user@your-vps
```

Then open <http://127.0.0.1:8000> locally and enter your access token.

For an HTTPS domain, point DNS at the VPS, set `DOMAIN=assistant.example.org`
in `.env`, and allow inbound ports 80 and 443. Then run:

```bash
docker compose --env-file .env -f deploy/compose.yaml --profile public up -d
```

Caddy provides HTTPS and forwards requests to the application. Add that actual
HTTPS URL to the book once the deployment is verified. The interface is public
but the model APIs, jobs, and downloads require the access token. This is a
**shared-access pilot**, not a service with separate user accounts or private
per-user histories; anyone holding the token may access a job whose ID they have.

## Update an existing installation

From the existing repository root, pull the checked-out branch:

```bash
git pull --ff-only
```

Keep the existing `.env`, AI key, and local Compose settings. Do not copy
`.env.example` again or reset local changes. If Git reports that a locally edited
Compose file would be overwritten, preserve those edits before updating instead
of forcing the checkout. The named jobs volume remains in place. On installations
where Docker requires root, prefix Docker commands with `sudo`.

### Use a tested image on an older laptop

The workflow is configured to export a Linux `amd64` image after its live tests
pass. Prefer this option on an older x86-64 laptop to avoid native compilation.
The artifact exists only after a successful run, and expires after three days.

In GitHub Actions, open a successful **FEMM 5 assistant** run for the same commit
as your updated checkout. Download **femm-linux-amd64-image** and extract its ZIP.
From the extracted folder, check the downloaded image:

```bash
sha256sum -c femm-assistant-linux-amd64.tar.gz.sha256
```

Proceed if the check reports `OK`, then load it:

```bash
sudo docker image load -i femm-assistant-linux-amd64.tar.gz
```

Return to your existing `research_extender/femm_assistant` folder and start it
with the existing settings:

```bash
sudo docker compose --env-file .env -f deploy/compose.yaml up --no-build -d assistant
```

The image contains the application version from that workflow's commit. It does
not contain your API key or local settings. Reload the browser after updating.

### Build the image yourself

From the updated repository root:

```bash
cd femm_assistant
docker compose --env-file .env -f deploy/compose.yaml up --build -d assistant
```

This version adds a native mesh-export patch, so an existing image needs one
FEMM recompilation. The initial rebuild can take substantial time on an old
laptop. Alternatively, build a Linux `amd64` image on a powerful machine and
transfer it with `docker image save` / `docker image load`, then start with
`--no-build` as above.

Changing only the key or model in `.env` needs no image rebuild; use the same
Compose command with `--no-build` in place of `--build`. Compose recreates the
service with the new environment, then reload the browser.

### What the image builds

The Dockerfile downloads the official `06oct2026_bd03d0b` source archive, checks
its published SHA-256, builds natively on Ubuntu 24.04, and installs the headless
engine, field solvers, and Lua launcher. Before compilation,
[`deploy/patch_femm_mesh.py`](deploy/patch_femm_mesh.py) adds an opt-in
`FEMM_MESH_EXPORT=1` callback export of FEMM's actual in-memory mesh. It writes
diagnostic files without changing the numerical meshing or solving algorithms.
The image retains the original upstream archive and patcher alongside the
installed program. It then installs only this app's four Python
dependencies and runs as a non-root user.

The image build supports an optional `proxy_ca` BuildKit secret for environments
with an HTTPS inspection proxy. For an ordinary VPS this is unnecessary. Never
disable TLS verification. Source/version changes belong in
`deploy/fetch_femm.py` and require rerunning the live numerical checks.

## Cost and operational limits

- One server, one application process, one simulation worker, SQLite, and local
  result storage. No separate database, queue, GPU, or local language model.
- Up to five queued/running jobs and 20 solve submissions per hour for the pilot.
- Up to 20 AI requests per hour. A general geometry request may make two Jev calls
  and two draft-model calls, capped at 6000 generated tokens per draft and
  90 seconds total, including one validation-guided repair. These in-memory limits
  reset on restart; use the provider's spending controls for an account-level budget.
- Each simulation has a 180 second timeout. The Compose container is limited to
  2 GB RAM, 1.5 CPU cores, and 128 processes; the whole process group is stopped
  on timeout.
- Completed and failed job files expire after 24 hours. Download anything you
  need to keep. The Docker volume persists across restarts; `down -v` removes it.
- Restarted in-flight jobs are marked failed; queued jobs resume. Use exactly one
  Uvicorn worker/replica with this SQLite worker design.
- The interface reports whether the launcher is detected. That is not evidence
  of a successful numerical solve; only completed jobs contain numerical results.

For initial low traffic, preconfigured forms avoid AI costs entirely. If usage
grows, measure peak memory and solve duration before increasing concurrency.

## Verification

Run the application/API tests without FEMM:

```bash
python -m unittest discover -s tests -v
node --check app/static/app.js
```

Run the live mesh, Python-export and numerical suites with the patched native
FEMM 5 installed:

```bash
FEMM_LIVE_TEST=1 FEMM_MESH_EXPORT=1 python -m unittest discover -s tests -p 'test_*live.py' -v
```

Or, after building the container:

```bash
docker run --rm --memory=2g --cpus=2 -e FEMM_LIVE_TEST=1 -e FEMM_MESH_EXPORT=1 \
  research-extender-femm:local python -m unittest discover -s tests -p 'test_*live.py' -v
```

The C-core suite checks force direction, zero current, linear flux scaling,
quadratic force scaling, increasing-gap sensitivity, mesh refinement, and the
linear energy/flux-linkage identity. The general-geometry suite builds a circular
wire in air, verifies that mesh-only mode creates no field solution, then compares
a solved field probe against Ampere's law, repeats with a finer mesh, doubles the
current, and checks field and energy scaling. The Python-export suite runs the
downloaded CAD script through mesh-only and solve stages. These tests use actual
FEMM. Without `FEMM_LIVE_TEST=1` they
are skipped, so a passing unit test run alone does not verify the native solver.

The `FEMM 5 assistant` GitHub Actions workflow includes application checks,
browser workflow checks, an image build and live suites on relevant pull requests.
Browser checks use fixture AI and job completion; they verify the UI and stage
gates rather than provider decisions or numerical physics. Review the current
workflow results before deployment; this document does not certify a CI run.
These are initial numerical checks; a formal
benchmark against an independently established reference remains future work.

## Model details

General geometry plans describe 2D planar DC magnetostatics in millimeters, with
an explicit out-of-plane depth. They support circular, rectangular and simple
polygonal contours; separate or nested labeled regions; linear isotropic
materials; and signed-turn DC circuit assignments. An outer contour encloses
background air with fixed magnetic vector potential A=0. Contours must not touch,
cross, or share edges. Every face needs a valid label and all references must
resolve. Plans outside the supported physics or bounded mesh budget are rejected.

Outputs include energy, defined circuit currents and flux linkages, specified
field probes, and optional force on an eligible region surrounded by air. Field
output is a 41 × 31 grid in tesla. Samples outside or on a nonrectangular outer
boundary are zero-filled for plotting; they are not field measurements there.
The download records the validated plan in `spec.json`, editable Python CAD
stages in `model.py`, and the compiled native sequence in `model.lua`.

The separate C-core example keeps its original parameterized geometry and
pole-edge refinement. Its paired winding cross-sections have opposite turn signs,
both pole gaps use the specified length, and core/armature permeability is
constant. Armature force uses FEMM's weighted stress tensor block integral.

Axisymmetry, AC, permanent magnets, nonlinear B-H curves, saturation, hysteresis,
losses, shared-edge Boolean geometry, and full 3D CAD are not supported. Mesh and
outer-domain convergence still matter for quantitative interpretation. Numerical
outputs come from FEMM, not from AI predictions or a surrogate model.

The existing LightRAG and MCP research tools remain independent. The assistant
does not retrieve private Drive documents or invoke the research planner.
