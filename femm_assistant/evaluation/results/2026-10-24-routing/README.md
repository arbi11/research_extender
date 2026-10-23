# Live function-routing and native execution loops

`summary.json` records every case outcome from five complete runs of the same
28 public cases and 56 natural-language phrasings. It includes source-file,
dataset, original-report and immutable Docker image hashes. Full original reports
and selected native exports are retained in the laptop's `femm-loop-evidence.zip`.
Every report is a separate stochastic measurement; failed cases are never replaced
by a later whole-case retry. These are development seeds used to improve the
instructions, not an independent holdout.

| Run | Draft model | Passed | Failures |
| --- | --- | ---: | --- |
| Initial instructions | GPT-4.1 Mini | 48/56 | Two invented material permeabilities, four current/turn errors, two missing probes |
| Refined instructions | GPT-4.1 Mini | 52/56 | Four routing timeouts before drafting |
| Refined instructions | Gemini 3.1 Flash Lite | 54/56 | Two routing timeouts before drafting |
| One short connection retry | Gemini 3.1 Flash Lite | 54/56 | Two routing timeouts before drafting |
| Final 10-second connection allowance | Gemini 3.1 Flash Lite | 56/56 | None observed |

The final loop produced 46 models whose typed physical values matched the explicit
requests; every one passed preview, actual native FEMM meshing, solving and ZIP
export. Ten incomplete or unsupported requests produced questions with no model
or native job. A separate archive audit verified all 46 exported specifications
against submitted mesh/solve specifications and checked finite output quantities.
Median full-case latency was 6.4482 seconds and the 95th percentile was 8.3493
seconds, with two concurrent isolated application instances.

`asymmetric-routing-final.json` separately records the original asymmetric C-core
creation and a follow-up extending only its upper arm from x=6 to x=14 mm. Both
matched their independently specified physical values and passed native mesh,
solve and export. Those two focused requests are additional to the 280 loop
attempts; their solved archives are included with the laptop evidence package.

The initial mistakes prompted explicit instructions distinguishing circuit current
from signed turns, requiring requested probes in the actual typed data, and asking
for unknown steel permeability. The refined model runs had no observed physical
mismatches. Routing failures around five seconds, then 10.2 seconds with two short
connection attempts, motivated the final transport allowance. Earlier reports did
not retain the HTTPX exception subtype, so connection exhaustion was inferred;
one additional successful timing probe did not reproduce the failure. The final
transport permits at most two 10-second connection attempts within a 25-second
logical decision deadline and the existing 90-second planner deadline. It does
not replay provider responses or ambiguous read/write failures.

`vision-loop-refined.json` preserves six geometry-only fixtures for each optional
reviewer. Qwen 3 VL 30B A3B agreed with four expected outcomes, with one false flag
and one missed issue, costing $0.00423299 across six requests. Gemini agreed with
all six expected outcomes, costing $0.0052155 with mean latency 2.0109 seconds.
The earlier comparison is retained in the evidence archive. This small seeded
comparison informed the new Gemini drafting/vision defaults; explicit saved model
overrides continue to take precedence, and `VISION_MODEL=` disables visual review.

`actual-mesh-vision.json` is a separate live API test using the actual native mesh
of the asymmetric C-core: 11,213 elements, visual status `matches`, cost $0.001031.
The native mesh job was identical before and after advisory review. The UI permits
geometry-only review even after a checked mesh exists; findings never control
preview, mesh or solve.

Local verification: 144 tests on the host (139 passed, five native-only skips),
144 in Docker (143 passed, one compiler-only skip), all five actual native FEMM
reference tests, and browser smoke covering routing, stale responses, advisory
review, oversized meshes and failed-call diagnosis. Native execution does not
prove requested intent, numerical convergence or arbitrary CAD support. Scope
remains 2D planar DC magnetostatics with linear materials and supported closed
primitives/nested regions. No production or old-laptop deployment occurred.
