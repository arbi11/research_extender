# Axisymmetric language-to-CAD trials (2026-10-24)

Two prompt variants for each of seven public development cases: four creations,
two typed modifications and one unsupported-3D clarification. All 14 attempts
passed the existing independent physical-signature and production-API harness.
The twelve executable models passed native mesh, solve and ZIP export; two
clarifications produced no model/jobs. No draft repair was needed.

Dataset: `evaluation/axisymmetric_prompt_cases.json`. Reproduce in the isolated
Docker development image with `python -m evaluation.batch --live --rounds 2
--max-cases 7 --workers 2 --cases evaluation/axisymmetric_prompt_cases.json
--output /reports/new-run.json`, passing the existing key only through env-file.

This public seed set is not an independent holdout or a numerical convergence
study. Native numerical axisymmetric references and journal targets are tested
separately. Provider usage/cost are unknown because the planner transport does
not expose them. Original detailed report and ZIP artifacts remain locally;
their report digest and compact outcomes are retained here.
