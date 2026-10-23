# FEMM 5 Assistant

A small Linux application for describing, previewing, solving, and exporting a
reproducible **2D planar DC C-core actuator**. It lives alongside the book and
does not install the repository's machine-learning training dependencies.

The first release supports one parameterized geometry family. General CAD,
axisymmetric models, AC analysis, nonlinear material curves, and surrogate
predictions are future work. The UI explicitly describes the current assumptions.

## What a reader can do

1. Describe a parameter change in natural language, or edit the numeric form.
2. Inspect the geometry and all assumptions before submitting a solve.
3. Run a queued FEMM 5 simulation and inspect armature force, energy, flux linkage,
   and a sampled magnetic-field plot.
4. Download a complete ZIP: `model.lua`, a standalone `run.py`, `spec.json`,
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

To enable natural-language configuration, export `OPENROUTER_API_KEY` and
`AI_MODEL` before starting. Choose a model supporting JSON-object responses.
`AI_BASE_URL` defaults to OpenRouter's API; it is server configuration, never a
user-supplied endpoint. Requests send the user's prompt and current parameters
to that provider. AI output is validated data, not executable code.

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
- Up to 20 AI requests per hour; each request caps generated tokens at 1500.
  These in-memory limits reset on restart; use the provider's spending controls
  for an account-level budget.
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

Run the real numerical test with native FEMM 5 installed:

```bash
FEMM_LIVE_TEST=1 python -m unittest discover -s tests -p test_femm_live.py -v
```

Or, after building the container:

```bash
docker run --rm --memory=2g --cpus=2 -e FEMM_LIVE_TEST=1 \
  research-extender-femm:local python -m unittest discover -s tests -p test_femm_live.py -v
```

The live test performs five actual solves. It checks force direction, zero-current
behavior, linear flux scaling, quadratic force scaling, increasing-gap sensitivity,
mesh refinement, and the linear energy/flux-linkage identity. It prints the
measured quantities and runtimes. It is skipped unless explicitly enabled, so a
passing unit test run alone does **not** validate the solver or container build.

The `FEMM 5 assistant` GitHub Actions workflow runs unit checks and separately
builds the Linux image and enables the live test on relevant pull requests.
Review both jobs before deployment. These are initial numerical checks; a formal
benchmark against an independently established reference remains future work.

## Model details

All form lengths are mm. `depth_mm` is the out-of-plane stack length. The C-core
and armature use one explicitly specified constant relative permeability. The
winding is represented by opposite positive/negative turn counts in two
cross-sections around the left limb; conductivity is zero for this DC impressed
current model. Both pole faces have the specified gap. The air-box boundary has
zero magnetic vector potential and extends two device dimensions beyond the
geometry. Pole-facing edges are refined to at most one third of the gap.

Armature force uses FEMM's weighted stress tensor block integral. Field output is
a 41 × 31 Cartesian grid in tesla. Saturation, hysteresis, losses, detailed wire
packing, and end effects are not represented. This is an educational model, with
all these assumptions included in the download. Boundary-distance convergence
and broader mesh studies are needed for quantitative engineering use.

The existing LightRAG and MCP research tools remain independent. This first
release does not perform retrieval over private Drive documents or invoke the
research planner. Those integrations can follow once the solve/export path is
verified.
