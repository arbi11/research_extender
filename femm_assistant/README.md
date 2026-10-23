# FEMM 5 Assistant

A small Linux application for describing, previewing, solving, and exporting
**2D planar DC magnetic models**. It lives alongside the book and does not
install the repository's machine-learning training dependencies.

General geometry mode builds new models from circles, rectangles and simple
polygons, or modifies an existing model plan. A separate C-core example retains
the original numeric form and solver. Axisymmetric models, AC analysis,
permanent magnets, nonlinear B-H curves, shared-edge Boolean operations and
full 3D CAD remain outside this version.

The [implementation pipeline](PIPELINE.md) distinguishes this working harness
from planned mesh-only stages, complete Python model exports and Jev decisions.

## What a reader can do

1. Describe a new geometry or a modification in natural language, load a round-wire
   example, or use the C-core numeric form.
2. Inspect the geometry and all assumptions before submitting a solve.
3. Run a queued FEMM 5 simulation and inspect energy, circuit quantities, field
   probes, optional force and a sampled magnetic-field plot.
4. Download a complete ZIP: the editable source script `model.lua`, a standalone
   Python launcher `run.py`, `spec.json`,
   geometry preview, FEMM model and solution, field CSV, quantities, and solver log.

Source-only downloads work before a solve and without an AI credential. They do
not contain a FEMM model/solution or numerical results. The portable scripts need
FEMM 5 but need no repository checkout or AI API key. The Lua script creates,
meshes, solves, and exports the model; the Python script launches it.

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
AI key/model, the numeric form remains usable and the natural-language control
is disabled. Local development has no access token by default; keep it on loopback.

For actual solves, build/install [FEMM 5](https://femm.info/doku/doku.php?id=download5)
using its [Linux instructions](https://femm.info/doku/doku.php?id=bindingsetup).
Put `femm-lua` on PATH, or set `FEMM_LUA` to its full launcher path. This application
uses FEMM 5's bundled Lua launcher and its headless engine. It does not use the
old PyPI `pyfemm` package, Wine, a display server, or Windows automation.

To enable natural-language requests, set these variables before starting:

```text
OPENROUTER_API_KEY=your-key-kept-on-the-server
AI_MODEL=openai/gpt-4.1-mini
```

For Docker, put them in the existing `.env`; plain `uvicorn` needs exported
variables. `AI_BASE_URL` defaults to OpenRouter's API. Keep the key out of Git and
chat. Requests send the prompt and, when modifying a model, its current geometry
plan to the provider.

The AI returns a structured model plan rather than executable Lua or Python.
The application validates contours, region labels, material/circuit references,
physics, and mesh budgets, then compiles the accepted plan into the same Lua used
by the worker and the download. New models need not use a template. Missing
physical inputs or unsupported requests produce clarification questions. General
geometry requests allow at most two provider calls: an initial proposal and one
validation-guided repair, each capped at 6000 generated tokens, within a total
60 second deadline. Review the proposed model before solving.

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
and selected model if you want natural-language requests. Keep this file out of Git.
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

From the existing repository root, pull the checked-out branch and rebuild:

```bash
git pull --ff-only
cd femm_assistant
docker compose --env-file .env -f deploy/compose.yaml up --build -d assistant
```

Keep the existing `.env`, AI key, and local Compose settings. Do not copy
`.env.example` again or reset local changes. If Git reports that a locally edited
Compose file would be overwritten, preserve those edits before updating instead
of forcing the checkout. The named jobs volume remains in place. On installations
where Docker requires root, prefix Docker commands with `sudo`.

Changing only the key or model in `.env` needs no image rebuild; use the same
Compose command with `--no-build` in place of `--build`. Compose recreates the
service with the new environment, then reload the browser.

### What the image builds

The Dockerfile downloads the official `06oct2026_bd03d0b` source archive, checks
its published SHA-256, builds natively on Ubuntu 24.04, and installs the headless
engine, field solvers, and Lua launcher. It retains the upstream source archive
alongside the installed program. It then installs only this app's four Python
dependencies and runs as a non-root user.

The image build supports an optional `proxy_ca` BuildKit secret for environments
with an HTTPS inspection proxy. For an ordinary VPS this is unnecessary. Never
disable TLS verification. Source/version changes belong in
`deploy/fetch_femm.py` and require rerunning the live numerical checks.

## Cost and operational limits

- One server, one application process, one simulation worker, SQLite, and local
  result storage. No separate database, queue, GPU, or local language model.
- Up to five queued/running jobs and 20 solve submissions per hour for the pilot.
- Up to 20 AI requests per hour. A general geometry request may make two provider
  calls, capped at 6000 generated tokens each and 60 seconds total, including one
  validation-guided repair. These in-memory limits reset on restart; use the
  provider's spending controls for an account-level budget.
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

Run both real numerical test suites with native FEMM 5 installed:

```bash
FEMM_LIVE_TEST=1 python -m unittest discover -s tests -p 'test_*live.py' -v
```

Or, after building the container:

```bash
docker run --rm --memory=2g --cpus=2 -e FEMM_LIVE_TEST=1 \
  research-extender-femm:local python -m unittest discover -s tests -p 'test_*live.py' -v
```

The C-core suite checks force direction, zero current, linear flux scaling,
quadratic force scaling, increasing-gap sensitivity, mesh refinement, and the
linear energy/flux-linkage identity. The general-geometry suite builds a circular
wire in air and compares a field probe against Ampere's law, repeats with a finer
mesh, doubles the current, and checks field and energy scaling. These tests run
actual FEMM solves and print their measurements. Without `FEMM_LIVE_TEST=1` they
are skipped, so a passing unit test run alone does not verify the native solver.

The `FEMM 5 assistant` GitHub Actions workflow runs unit checks and separately
builds the Linux image and enables the live test on relevant pull requests.
Review both jobs before deployment. These are initial numerical checks; a formal
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
The download records the validated plan in `spec.json` and its complete compiled
geometry/material/boundary/mesh/solve/export sequence in `model.lua`.

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
