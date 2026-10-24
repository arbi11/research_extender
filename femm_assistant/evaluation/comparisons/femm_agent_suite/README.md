# FEMM Agent Suite capability evaluation — 24 October 2026

This records an actual isolated test of
[FEMM Agent Suite](https://github.com/Zwl20085/femm-agent-suite), pinned to commit
`019d6a60bc09df013145e88c4e8deb8d91e11f6b`. The suite source was not modified.
Its maintainer is [Wentao Zhang / Zwl20085](https://github.com/Zwl20085); his
[professional website](https://zwl20085.github.io/) identifies him as a Research
Assistant at PEMC, University of Nottingham. This is an independent toolkit, not
an attribution of the project to that university or to FEMM author David Meeker.

`results.json` contains measured outcomes, dependency/script hashes and limitations.
The laptop evidence package additionally retains every model attempt, original
harness/setup failures, XML/logs, native models/solutions, plots and reports.
No global MCP/Claude installation, GitHub star, outbound message, production
deployment or change to saved settings/running assistant services occurred.

## Native capabilities actually tested

- Offline suite: **123 passed, four skipped** (Windows-specific binding checks).
- Native suite after resource setup: **19 passed, two skipped** (GUI and Windows
  Lua launcher). Native tests covered all four physics, axisymmetric solenoids and
  coilguns, nonlinear steel, permanent magnets, magnetic air-gap elements, AC
  wire impedance/power balance, actuator force/inductance, motor cogging,
  back-EMF/dq torque, MCP stdio and standalone Python replay.
- Official C-core and quick motor examples produced native results, plots and
  reports. The motor example's cogging peak-to-peak agreed within 1.63% between
  rotation and air-gap-element methods, but two other comparison columns exceeded
  their tolerance. The example is not an unconditional all-metrics pass.

The first native run had one manual-discovery failure, 17 passes and three skips.
The Linux CMake installation already contained the manual, examples and Python
bindings under `share/femm`, but the suite searches the traditional layout.
`Dockerfile.support` added path aliases and optional `pypdf`, enabling discovery
and replay without editing the suite or rebuilding/changing FEMM.

## OpenRouter-driven natural-language trials

The suite is an MCP toolkit, not a bundled chat model or browser application.
`openrouter_trials.py` is our bounded evaluation client. It supplies the suite's
modeling guidance, exposes 88 of its 89 tools (excluding the unrelated GitHub-star
action), executes tool calls sequentially, and preserves actual responses.
Credentials are not passed to the MCP server/FEMM child environment. The client
retains failed attempts and does not replace them with whole-case retries.

Gemini 3.1 Flash Lite completed four requests through actual MCP and FEMM:

| Request | Independent result |
| --- | --- |
| User's asymmetric C-core plus requested total energy | Geometry, assignments, boundary, explicit mesh and probe correct; native solve succeeded. Agent reported core-only energy as total energy. |
| Extend only the upper arm to x=14 mm | Correct edit and preserved physical properties; solve/probe correct. Same energy-scope reporting error. |
| Parallel-plate capacitor | Native geometry, dielectric, conductors, insulation, mesh, voltage, field, capacitance, charge and energy verified. |
| Copper wire at 10, 1000 and 5000 Hz | Native sweep and power balance completed, but A=0 was assigned to only one of four outer arcs. Requested boundary fidelity failed. |

The four requests used 12 model rounds, 65 tool calls (64 successful and one
correctly rejected empty block selection), costing $0.0389161 as reported by
OpenRouter. These are small guided smoke tests, not a general accuracy score.
The incorrect energy scope and arc selection came from model arguments/reporting;
the suite tools executed those supplied calls correctly.

The true C-core total energies were `3.050518389e-5 J` and `3.508982605e-5 J`;
the model instead reported `2.508437266e-7 J` and `3.741465306e-7 J` from the core
alone. Independent all-region integrals satisfy `W = 0.5 I psi` to floating-point
precision. `check_ac_boundary.py` corrects the boundary on a separate copy:
inductance then agrees with the analytic concentric-wire reference within 0.26%
over all three frequencies. These corrections do not convert original attempts
into passes.

GPT-4.1 Mini was also tried with the same client and 16-turn limit. Its C-core and
capacitor attempts reached that limit before completing, so the dependent edit
was not attempted. The AC trial omitted the boundary definition and timed out
during the native solve. These are incomplete observations under this harness,
not a reliable model ranking or evidence that the suite lacks those capabilities.
All paid attempts across both models cost **$0.2107657**, provider-reported.

## Independently reproduced stale results

`native_probes.py` solves a round wire at 1 A, changes its current to 2 A and reads
the result before re-solving. The session still reports `solved=True` and returns
the old 1 A field. Explicit re-solving produces exactly twice the field. The suite
documents that agents must re-solve after edits, but does not enforce result
invalidation. This observation used registered tools and real FEMM, without an LLM.

## Reproducing

Use a separate source checkout and image. Do not run the upstream installer on a
shared machine: it registers client configuration and copies skills globally.
The Dockerfiles assume the existing development FEMM image tag shown in their
`FROM` line. Build that tag from this repository's `femm_assistant/Dockerfile` if
needed, then:

```bash
git clone https://github.com/Zwl20085/femm-agent-suite.git /tmp/femm-suite-source
git -C /tmp/femm-suite-source checkout 019d6a60bc09df013145e88c4e8deb8d91e11f6b
docker build -f "$PWD/Dockerfile" -t femm-agent-suite:evaluation-019d6a6 /tmp/femm-suite-source
docker build -f Dockerfile.support -t femm-agent-suite:evaluation-supported-019d6a6 .
mkdir -p evaluation-results
docker run --rm --network none --user "$(id -u):$(id -g)" \
  -v "$PWD/evaluation-results:/testwork" \
  femm-agent-suite:evaluation-supported-019d6a6 \
  python -m pytest -m live -vv -p no:cacheprovider --junitxml=/testwork/native.xml
```

Paid trials are explicit opt-in, use a fresh filename, and retain their limits:

```bash
docker run --rm --user "$(id -u):$(id -g)" --memory=2g --cpus=2 \
  --env-file /path/to/existing/untracked/.env \
  -v "$PWD/evaluation-results:/testwork" -v "$PWD:/evaluation:ro" \
  femm-agent-suite:evaluation-supported-019d6a6 \
  python /evaluation/openrouter_trials.py --live --output /testwork/new-trial.json
```

Use the saved reports to distinguish model completion, native execution, requested
model fidelity and result scope. This evaluation makes no differentiation claim
or claim of reliable arbitrary CAD, transient thermal coverage or general
numerical convergence.
