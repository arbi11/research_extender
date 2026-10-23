# Development measurements, 2026-10-24

All prompts in these reports are public evaluation fixtures. The reports contain
no API keys, authorization headers, user model history or environment dumps.

* `2026-10-24/semantic-seed.json`: initial six-case live-provider baseline;
  5/6 matching gate labels, zero false accepts, one false reject.
* `2026-10-24/semantic-full.json`: all 28 cases after clarifying the separation
  between supported physics, required inputs and schema-level fidelity;
  27/28 matching gate labels, zero false accepts, one false reject, no provider
  errors, with the existing 0.90 threshold. `wire_assumptions` was conservatively
  rejected despite explicit permission to assume physical values.
* `2026-10-24/cad-final.json`: four passing real-provider/real-solver checks:
  creation, current modification, missing-depth clarification and unsupported-AC
  clarification. The generated wire's probe-vector magnitude differs from the
  analytic reference by 0.736%; double current gives exactly 4.0 times the energy
  to the recorded precision. Both solves require actual passed mesh checks and
  include verified native exports. The first draft failed deterministic validation
  for explicit air definitions and a duplicate domain region; the existing one-
  repair loop corrected it before fidelity approval or execution.

The baseline and final runs are separate stochastic measurements, not paired
samples or an independently labeled holdout. No runtime threshold was tuned.
The criteria and dataset hashes are recorded in semantic reports. Exact provider
revision and future repeatability are not established. These results do not
certify general CAD accuracy or a probability calibration guarantee.
