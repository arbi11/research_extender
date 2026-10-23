"""Browser integration smoke test, run separately from unittest discovery.

Install Playwright and Chromium, then run ``python tests/browser_smoke.py``.
The real FastAPI app supplies authentication, geometry validation, previews,
exports and queued jobs. A worker fixture completes the actual temporary jobs,
while browser routes provide AI proposals and mesh/field SVG fixtures. This test
does not contact an AI provider or run FEMM. Native solver and provider-adapter
tests cover those separate boundaries.
"""
from collections import deque
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[1]
TOKEN = "browser-pilot"
SERVER = """
import sys
from pathlib import Path
import uvicorn
from app.main import Settings, create_app
settings = Settings(data_dir=Path(sys.argv[1]), token='browser-pilot',
                    ai_key='browser-fixture', ai_model='browser-fixture',
                    ai_base_url='http://127.0.0.1:9/browser-fixture',
                    femm_lua='python3', start_worker=False)
uvicorn.run(create_app(settings), host='127.0.0.1', port=int(sys.argv[2]), log_level='warning')
"""
FIELD_FIXTURE = ('<svg xmlns="http://www.w3.org/2000/svg" width="160" height="80">'
                 '<rect width="160" height="80" fill="#528eaa"/>'
                 '<text x="8" y="45">Browser fixture</text></svg>')
MESH_FIXTURE = ('<svg xmlns="http://www.w3.org/2000/svg" width="160" height="80">'
                '<path d="M5 5H155V75H5ZM5 5L155 75M155 5L5 75M80 5V75" '
                'fill="none" stroke="#365c5a"/><text x="8" y="45">Fixture mesh</text></svg>')
MESH_REPORT = {"status": "passed", "node_count": 6, "element_count": 4, "edge_count": 9,
               "quality": {"min_angle_deg": 26.565, "max_aspect_ratio": 2.5},
               "checks": [{"name": "triangle_connectivity", "status": "passed",
                           "detail": "Browser worker fixture: connectivity passed."}],
               "warnings": ["Browser fixture; numerical convergence has not been verified."],
               "convergence_verified": False}
# Routing evidence is informational. Even modest probability/confidence must
# never suppress a valid generated plan or impose a geometry approval gate.
DECISION_TRACE = [
    {"stage": "routing", "question": "route", "answer": "create_plan", "probability": .55,
     "confidence": .54, "probabilities": {"create_plan": .55, "modify_plan": .25, "clarify": .20},
     "threshold": None, "passed": True, "model": "typesafe/jev-1.14"},
]
DIAGNOSIS = {"function_name": "mesh_plan", "arguments_match": False, "trace": [
    {"stage": "function_error", "question": "arguments_match", "answer": "no", "probability": .64,
     "confidence": None, "probabilities": {"yes": .36, "no": .64}, "threshold": None,
     "passed": True, "model": "typesafe/jev-1.14"},
]}
VISUAL_REVIEW = {"advisory": True, "status": "issues", "summary": "Inspect the conductor placement.",
                 "findings": [{"location": "Conductor", "issue": "The requested placement may differ.",
                               "suggestion": "Compare the position with your requested coordinates."}],
                 "requested_model": "browser-vision", "usage": {"total_tokens": 12}}



def wait_for_server(process, url, log):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if process.poll() is not None:
            break
        try:
            with urllib.request.urlopen(url + "/healthz", timeout=1) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            time.sleep(.1)
    log.flush()
    raise RuntimeError("Browser test server did not start: " + Path(log.name).read_text()[-4000:])


def assert_source_archive(download, folder, kind):
    destination = folder / download.suggested_filename
    download.save_as(str(destination))
    assert download.failure() is None, "Source archive download failed"
    with zipfile.ZipFile(destination) as archive:
        assert {"spec.json", "model.lua", "run.py", "geometry.svg", "README.txt"}.issubset(archive.namelist())
        assert "model.ans" not in archive.namelist(), "Source export fabricated a solution"
        assert "results.csv" not in archive.namelist(), "Source export fabricated results"
        spec = json.loads(archive.read("spec.json"))
        if kind == "plan":
            assert spec["kind"] == "magnetostatic_plan"
        else:
            assert spec["geometry"] == "c_core"


def exercise_browser(url, folder):
    from playwright.sync_api import expect, sync_playwright
    sys.path.insert(0, str(ROOT))
    from app.jobs import JobStore

    store = JobStore(folder / "data")
    proposals = deque()
    mesh_fixtures = deque()
    solve_fixtures = deque()
    visual_fixtures = deque()
    visual_requests = []
    diagnosis_requests = []
    stale_diagnosis = False
    ai_requests = []
    mesh_specs = []
    completed_specs = []
    solve_requests = []
    page_errors = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(accept_downloads=True, viewport={"width": 1440, "height": 1100})
        page = context.new_page()
        page.on("pageerror", lambda error: page_errors.append(str(error)))
        page.on("request", lambda request: solve_requests.append(request.post_data_json)
                if request.method == "POST" and request.url.endswith("/api/plan/jobs") else None)
        page.set_default_timeout(10000)

        def ai_fixture(route):
            assert route.request.headers.get("authorization") == "Bearer " + TOKEN
            ai_requests.append(route.request.post_data_json)
            assert proposals, "An unexpected AI request escaped the test sequence"
            proposal = proposals.popleft()
            edited_draft = proposal.pop("_edit_draft", None)
            if edited_draft is not None:
                # Withhold the AI answer until an actual input event changes the
                # user's draft. A delayed response must never overwrite that edit.
                page.evaluate("""draft => {
                    const editor = document.getElementById('plan-json');
                    editor.value = draft;
                    editor.dispatchEvent(new Event('input', {bubbles: true}));
                }""", edited_draft)
            route.fulfill(json=proposal)

        def config_fixture(route):
            response = route.fetch()
            value = response.json()
            value.update(visual_review_available=True, argument_diagnosis_available=True)
            route.fulfill(json=value)

        def visual_fixture(route):
            assert route.request.headers.get("authorization") == "Bearer " + TOKEN
            assert visual_fixtures, "Unexpected paid visual call escaped fixture sequence"
            visual_requests.append(route.request.post_data_json)
            behavior = visual_fixtures.popleft()
            # An advisory request must not disable normal model actions.
            expect(page.locator("#source")).to_be_enabled()
            expect(page.locator("#mesh-check")).to_be_enabled()
            if "mesh_job_id" in visual_requests[-1]:
                expect(page.locator("#solve")).to_be_enabled()
            if behavior == "stale":
                page.evaluate("""() => {
                    const prompt = document.getElementById('prompt');
                    prompt.value += ' Updated request.';
                    prompt.dispatchEvent(new Event('input', {bubbles: true}));
                }""")
            if behavior == "toggle_mesh":
                page.evaluate("""() => {
                    const checkbox = document.getElementById('visual-include-mesh');
                    checkbox.checked = !checkbox.checked;
                    checkbox.dispatchEvent(new Event('change', {bubbles: true}));
                }""")
            if behavior == "oversized_mesh":
                assert "mesh_job_id" in visual_requests[-1]
                route.fulfill(status=422, json={"detail": "The preview has too many elements for visual review. Review geometry only or use a smaller mesh."})
            elif behavior == "failed":
                route.fulfill(status=502, json={"detail": "Browser visual provider fixture unavailable."})
            else:
                route.fulfill(json=VISUAL_REVIEW)

        def diagnosis_fixture(route):
            nonlocal stale_diagnosis
            assert route.request.headers.get("authorization") == "Bearer " + TOKEN
            job_id = route.request.url.split("/")[-2]
            job = store.get(job_id)
            assert job["status"] == "failed", "Diagnosis must only follow an actual persisted function failure"
            diagnosis_requests.append(job_id)
            if stale_diagnosis:
                page.evaluate("""() => {
                    const prompt = document.getElementById('prompt');
                    prompt.value = 'Current request changed while diagnosing an earlier failure.';
                    prompt.dispatchEvent(new Event('input', {bubbles: true}));
                }""")
                stale_diagnosis = False
            value = json.loads(json.dumps(DIAGNOSIS))
            value["function_name"] = "mesh_plan" if job["kind"] == "mesh" else "solve_plan"
            route.fulfill(json=value)

        def complete_job_fixture(route):
            assert route.request.headers.get("authorization") == "Bearer " + TOKEN
            if route.request.url.endswith("/field.svg"):
                route.fulfill(content_type="image/svg+xml", body=FIELD_FIXTURE)
                return
            if route.request.url.endswith("/mesh.svg"):
                route.fulfill(content_type="image/svg+xml", body=MESH_FIXTURE)
                return
            response = route.fetch()
            assert response.status == 200
            job = response.json()
            assert job["status"] == "queued", "The browser test must not start a real solver"
            if job["kind"] == "mesh":
                assert mesh_fixtures, "Unexpected mesh job escaped the fixture sequence"
                behavior = mesh_fixtures.popleft()
                mesh_specs.append(job["spec"])
                if behavior == "stale":
                    # Finish an earlier model's real queued job only after a real
                    # editor event changes the draft; it must not enable Solve.
                    page.evaluate("""() => {
                        const editor = document.getElementById('plan-json');
                        const draft = JSON.parse(editor.value);
                        draft.title = 'Draft edited during the mesh fixture';
                        editor.value = JSON.stringify(draft);
                        editor.dispatchEvent(new Event('input', {bubbles: true}));
                    }""")
                if behavior == "failed":
                    store.update(job["id"], "failed", error="Browser worker fixture rejected this mesh.")
                else:
                    store.update(job["id"], "completed", result=MESH_REPORT)
            else:
                assert job["kind"] == "solve"
                assert solve_fixtures, "Unexpected solve escaped the fixture sequence"
                behavior = solve_fixtures.popleft()
                if behavior == "failed":
                    store.update(job["id"], "failed", error="Browser worker fixture failed the solve function.")
                else:
                    completed_specs.append(job["spec"])
                    store.update(job["id"], "completed", result={"energy": {"value": .001, "unit": "J"},
                                 "by_0": {"value": .0001, "unit": "T"}, "runtime": {"value": .02, "unit": "s"}})
            # This fixture replaces only the worker boundary. Completed status is
            # persisted so real solve POSTs still enforce exact-plan mesh gating.
            finished = route.fetch()
            assert finished.status == 200
            route.fulfill(response=finished)

        def expect_model_ready():
            expect(page.locator("#mesh-check")).to_be_enabled()
            expect(page.locator("#source")).to_be_enabled()
            expect(page.locator("#solve")).to_be_disabled()

        def check_mesh():
            mesh_fixtures.append("passed")
            page.locator("#mesh-check").click()
            expect(page.locator("#mesh-status")).to_have_text("PASSED")
            expect(page.locator("#mesh-image")).to_be_visible()
            expect(page.locator("#mesh-quantities")).to_contain_text("Nodes")
            expect(page.locator("#mesh-quantities dd")).to_have_text(["6", "4", "9", "26.57°", "2.50"])
            expect(page.locator("#solve")).to_be_enabled()

        page.route("**/api/config", config_fixture)
        page.route("**/api/plan/interpret", ai_fixture)
        page.route("**/api/plan/review", visual_fixture)
        page.route(re.compile(r".*/api/jobs/[a-f0-9]{32}/diagnose$"), diagnosis_fixture)
        page.route(re.compile(r".*/api/jobs/[a-f0-9]{32}(?:/(?:field|mesh)\.svg)?$"), complete_job_fixture)
        page.goto(url, wait_until="networkidle")
        expect(page.locator("#access")).to_be_visible()
        expect(page.locator("#mode")).to_have_value("plan")
        expect(page.locator("#solve")).to_be_disabled()

        # Login uses the real protected preview endpoint, including a failed key.
        page.locator("#token").fill("incorrect-token")
        page.locator("#access-form button").click()
        expect(page.locator("#message")).to_contain_text("access token")
        page.locator("#token").fill(TOKEN)
        page.locator("#access-form button").click()
        expect(page.locator("#message")).to_contain_text("Connected")

        page.locator("#load-example").click()
        expect_model_ready()
        expect(page.locator("#geometry")).to_be_visible()
        wire = json.loads(page.locator("#plan-json").input_value())
        assert wire["regions"][0]["shape"]["type"] == "circle"
        with page.expect_download() as download:
            page.locator("#source").click()
        assert_source_archive(download.value, folder, "plan")
        check_mesh()

        # Editing a valid preview invalidates it immediately. The server then
        # rejects an invalid material reference and the UI stays blocked.
        page.locator("#plan-editor > summary").click()
        invalid = json.loads(json.dumps(wire))
        invalid["regions"][0]["material"] = "undefined_material"
        page.locator("#plan-json").fill(json.dumps(invalid))
        expect(page.locator("#solve")).to_be_disabled()
        expect(page.locator("#mesh-check")).to_be_disabled()
        expect(page.locator("#mesh-image")).to_be_hidden()
        expect(page.locator("#source")).to_be_disabled()
        page.locator("#validate-plan").click()
        expect(page.locator("#message")).to_have_class(re.compile(r".*error.*"))
        expect(page.locator("#geometry")).to_be_hidden()
        page.locator("#plan-json").fill(json.dumps(wire))
        page.locator("#validate-plan").click()
        expect_model_ready()

        mesh_fixtures.append("stale")
        page.locator("#mesh-check").click()
        expect(page.locator("#mesh-message")).to_contain_text("earlier model")
        expect(page.locator("#solve")).to_be_disabled()
        expect(page.locator("#source")).to_be_disabled()
        assert json.loads(page.locator("#plan-json").input_value())["title"] == "Draft edited during the mesh fixture"
        page.locator("#plan-json").fill(json.dumps(wire))
        page.locator("#validate-plan").click()
        expect_model_ready()

        mesh_fixtures.append("failed")
        page.locator("#mesh-check").click()
        expect(page.locator("#mesh-status")).to_have_text("FAILED")
        expect(page.locator("#mesh-message")).to_contain_text("fixture rejected")
        expect_model_ready()
        expect(page.locator("#mesh-diagnose")).to_be_visible()
        page.locator("#mesh-diagnose").click()
        expect(page.locator("#mesh-diagnosis")).to_contain_text("arguments may not match")
        expect(page.locator("#mesh-diagnosis")).to_contain_text("Reported probability 64.0%")
        expect(page.locator("#mesh-status")).to_have_text("FAILED")
        expect_model_ready()
        stale_diagnosis = True
        with page.expect_response(re.compile(r".*/api/jobs/[a-f0-9]{32}/diagnose$")):
            page.locator("#mesh-diagnose").click()
        expect(page.locator("#mesh-diagnosis")).to_be_hidden()
        expect_model_ready()

        # Deterministically withhold an old preview response until a real editor
        # input event has invalidated its snapshot. The stale response must not
        # re-enable actions or restore the old image.
        def stale_preview_fixture(route):
            response = route.fetch()
            assert response.status == 200
            page.evaluate("""() => {
                const editor = document.getElementById('plan-json');
                editor.value += ' ';
                editor.dispatchEvent(new Event('input', {bubbles: true}));
            }""")
            route.fulfill(response=response)

        page.route("**/api/plan/preview", stale_preview_fixture, times=1)
        with page.expect_response("**/api/plan/preview"):
            page.locator("#validate-plan").click()
        expect(page.locator("#solve")).to_be_disabled()
        expect(page.locator("#source")).to_be_disabled()
        expect(page.locator("#geometry")).to_be_hidden()
        page.locator("#validate-plan").click()
        expect_model_ready()

        # A new AI-created topology uses no inherited model. Its rectangle is
        # previewed by the real compiler and later submitted through the real API.
        rectangle = json.loads(json.dumps(wire))
        rectangle["title"] = "Rectangular conductor from a browser AI fixture"
        rectangle["regions"][0]["shape"] = {"type": "rectangle", "x_min": -1, "x_max": 1,
                                             "y_min": -1, "y_max": 1}
        proposals.append({"spec": rectangle, "explanation": "Created the requested rectangle.",
                          "questions": [], "attempts": 1, "function_name": "create_plan", "decision_trace": DECISION_TRACE})
        page.locator("#request-action").select_option("new")
        page.locator("#prompt").fill("Create the fully specified rectangular conductor model.")
        page.locator("#interpret").click()
        expect_model_ready()
        expect(page.locator("#plan-summary")).to_contain_text(rectangle["title"])
        expect(page.locator("#decision-trace")).to_be_visible()
        page.locator("#decision-trace > summary").click()
        expect(page.locator("#decision-list li")).to_have_count(1)
        expect(page.locator("#decision-list li").nth(0)).to_be_visible()
        expect(page.locator("#decision-list li").nth(0)).to_contain_text("Chosen function: create_plan")
        expect(page.locator("#decision-list li").nth(0)).to_contain_text("Reported probability 55.0%")
        expect(page.locator("#decision-list li").nth(0)).to_contain_text("Provider confidence 54.0%")
        assert "Needs attention" not in page.locator("#decision-list").inner_text()
        assert "Model matches" not in page.locator("#decision-list").inner_text()
        expect(page.locator("#geometry")).to_be_visible()
        expect_model_ready()
        assert ai_requests[-1]["current"] is None

        # Optional visual observations never approve or reject this valid plan.
        # Neither concerns nor provider failure can change preview/export/mesh.
        expect(page.locator("#visual-review")).to_be_enabled()
        expect(page.locator("#visual-include-mesh")).to_be_checked()
        expect(page.locator("#visual-include-mesh")).to_be_disabled()
        expect(page.locator("#visual-mesh-help")).to_contain_text("Geometry only")
        for behavior, expected in [("issues", "Possible discrepancies"), ("failed", "Visual review unavailable")]:
            visual_fixtures.append(behavior)
            page.locator("#visual-review").click()
            expect(page.locator("#visual-review-summary")).to_contain_text(expected)
            expect(page.locator("#geometry")).to_be_visible()
            expect_model_ready()
            assert json.loads(page.locator("#plan-json").input_value()) == rectangle
        assert "mesh_job_id" not in visual_requests[-1]
        visual_fixtures.append("stale")
        with page.expect_response("**/api/plan/review"):
            page.locator("#visual-review").click()
        expect(page.locator("#visual-review-panel")).to_be_hidden()
        expect_model_ready()

        modified = json.loads(json.dumps(rectangle))
        modified["circuits"][0]["current_a"] = 2
        proposals.append({"spec": modified, "explanation": "Changed only the current to 2 A.", "questions": [], "attempts": 1})
        page.locator("#request-action").select_option("modify")
        page.locator("#prompt").fill("Set the current to 2 A and preserve all geometry.")
        page.locator("#interpret").click()
        expect(page.locator("#explanation")).to_contain_text("Changed only")
        expect_model_ready()
        assert ai_requests[-1]["current"]["title"] == rectangle["title"]
        assert ai_requests[-1]["current"]["circuits"][0]["current_a"] == 1
        assert json.loads(page.locator("#plan-json").input_value())["circuits"][0]["current_a"] == 2

        manual_edit = json.loads(json.dumps(modified))
        manual_edit["title"] = "Manual draft edited while AI was working"
        proposals.append({"spec": modified, "explanation": "This answer belongs to the older draft.",
                          "questions": [], "attempts": 1, "_edit_draft": json.dumps(manual_edit)})
        page.locator("#prompt").fill("Review the current model while I edit its title.")
        page.locator("#interpret").click()
        expect(page.locator("#message")).to_contain_text("not applied")
        expect(page.locator("#solve")).to_be_disabled()
        expect(page.locator("#source")).to_be_disabled()
        assert json.loads(page.locator("#plan-json").input_value())["title"] == manual_edit["title"]
        page.locator("#plan-json").fill(json.dumps(modified))
        page.locator("#validate-plan").click()
        expect_model_ready()

        # A valid candidate with questions remains gated until explicitly resolved.
        proposals.append({"spec": modified, "explanation": "Confirm the stated material assumption.",
                          "questions": ["Is relative permeability 1 correct for the conductor?"], "attempts": 1})
        page.locator("#prompt").fill("Review the conductor material assumption.")
        page.locator("#interpret").click()
        expect(page.locator("#questions")).to_be_visible()
        expect(page.locator("#solve")).to_be_disabled()
        expect(page.locator("#source")).to_be_disabled()
        page.locator("#resolved").check()
        expect_model_ready()

        # A clarification-only modification preserves the validated old model but
        # must still disable solving; it must not fall into the C-core code path.
        proposals.append({"spec": None, "explanation": "The requested physics needs clarification.",
                          "questions": ["Can this request use planar DC magnetics?"], "attempts": 1})
        page.locator("#prompt").fill("Modify this model using unsupported AC physics.")
        page.locator("#interpret").click()
        expect(page.locator("#question-list")).to_contain_text("planar DC")
        expect(page.locator("#resolved-label")).to_be_hidden()
        expect(page.locator("#solve")).to_be_disabled()
        expect(page.locator("#source")).to_be_disabled()
        assert json.loads(page.locator("#plan-json").input_value())["title"] == modified["title"]

        proposals.append({"spec": modified, "explanation": "Kept the supported DC model.", "questions": [], "attempts": 1})
        page.locator("#prompt").fill("Keep DC physics and the current model unchanged.")
        page.locator("#interpret").click()
        expect(page.locator("#questions")).to_be_hidden()
        expect_model_ready()
        check_mesh()
        expect(page.locator("#visual-include-mesh")).to_be_checked()
        expect(page.locator("#visual-include-mesh")).to_be_enabled()
        expect(page.locator("#visual-mesh-help")).to_contain_text("will be included")
        visual_fixtures.append("oversized_mesh")
        page.locator("#visual-review").click()
        expect(page.locator("#visual-review-summary")).to_contain_text("Review geometry only")
        expect(page.locator("#solve")).to_be_enabled()
        expect(page.locator("#visual-include-mesh")).to_be_enabled()
        # Changing the input choice while a review is loading invalidates its
        # response. A checked mesh can then be excluded for geometry-only review.
        visual_fixtures.append("toggle_mesh")
        with page.expect_response("**/api/plan/review"):
            page.locator("#visual-review").click()
        expect(page.locator("#visual-review-panel")).to_be_hidden()
        expect(page.locator("#visual-include-mesh")).not_to_be_checked()
        assert "mesh_job_id" in visual_requests[-1]
        visual_fixtures.append("issues")
        page.locator("#visual-review").click()
        expect(page.locator("#visual-review-summary")).to_contain_text("Possible discrepancies")
        expect(page.locator("#visual-review-source")).to_have_text("Inputs: geometry only.")
        assert "mesh_job_id" not in visual_requests[-1]
        expect(page.locator("#solve")).to_be_enabled()
        assert json.loads(page.locator("#plan-json").input_value()) == modified
        page.locator("#visual-include-mesh").check()
        expect(page.locator("#visual-review-panel")).to_be_hidden()
        visual_fixtures.append("issues")
        page.locator("#visual-review").click()
        expect(page.locator("#visual-review-summary")).to_contain_text("Possible discrepancies")
        expect(page.locator("#visual-review-source")).to_have_text("Inputs: geometry and the checked FEMM mesh.")
        expect(page.locator("#solve")).to_be_enabled()
        assert store.get(visual_requests[-1]["mesh_job_id"])["status"] == "completed"
        assert visual_requests[-1]["spec"] == modified
        solve_fixtures.append("failed")
        page.locator("#solve").click()
        expect(page.locator("#job-status")).to_have_text("FAILED")
        expect(page.locator("#solve-diagnose")).to_be_visible()
        page.locator("#solve-diagnose").click()
        expect(page.locator("#solve-diagnosis")).to_contain_text("solve_plan")
        expect(page.locator("#job-status")).to_have_text("FAILED")
        expect(page.locator("#solve")).to_be_enabled()
        solve_fixtures.append("passed")
        page.locator("#solve").click()
        expect(page.locator("#job-status")).to_have_text("COMPLETED")
        expect(page.locator("#solve-diagnose")).to_be_hidden()
        expect(page.locator("#result-body")).to_be_visible()
        expect(page.locator("#quantities")).to_contain_text("Probe 1")
        expect(page.locator("#job-model")).to_contain_text(modified["title"])
        assert completed_specs[-1]["kind"] == "magnetostatic_plan"
        assert completed_specs[-1]["circuits"][0]["current_a"] == 2
        assert solve_requests[-1]["spec"] == completed_specs[-1]
        mesh_job = store.get(solve_requests[-1]["mesh_job_id"])
        assert mesh_job["kind"] == "mesh" and mesh_job["status"] == "completed"
        assert mesh_job["spec"] == solve_requests[-1]["spec"]

        artifacts = os.getenv("BROWSER_SMOKE_ARTIFACTS")
        if artifacts:
            destination = Path(artifacts)
            destination.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(destination / "general-model-results.png"), full_page=True)

        # A new unsupported request clears its draft rather than silently reusing
        # the prior topology. Loading the example is an explicit recovery action.
        proposals.append({"spec": None, "explanation": "A 3D model is outside this harness.",
                          "questions": ["Can the model be expressed as a planar cross-section?"], "attempts": 1})
        page.locator("#request-action").select_option("new")
        page.locator("#prompt").fill("Create a 3D motor instead of a planar model.")
        page.locator("#interpret").click()
        expect(page.locator("#question-list")).to_contain_text("planar cross-section")
        expect(page.locator("#plan-json")).to_have_value("")
        expect(page.locator("#solve")).to_be_disabled()
        expect(page.locator("#source")).to_be_disabled()
        page.locator("#load-example").click()
        expect_model_ready()
        expect(page.locator("#questions")).to_be_hidden()

        # Original C-core controls and source exports remain functional. Existing
        # general results continue to identify the submitted general specification.
        page.locator("#mode").select_option("ccore")
        expect(page.locator("#ccore-controls")).to_be_visible()
        expect(page.locator("#solve")).to_be_enabled()
        expect(page.locator("#width_mm")).to_have_value("60")
        expect(page.locator("#job-model")).to_contain_text(modified["title"])
        page.locator("#current_a").fill("2")
        expect(page.locator("#source")).to_be_enabled()
        with page.expect_download() as download:
            page.locator("#source").click()
        assert_source_archive(download.value, folder, "ccore")
        if artifacts:
            page.screenshot(path=str(Path(artifacts) / "ccore-example.png"), full_page=True)
        assert not proposals, "Not every AI fixture was consumed"
        assert not mesh_fixtures, "Not every mesh fixture was consumed"
        assert not solve_fixtures, "Not every solve fixture was consumed"
        assert not visual_fixtures, "Not every visual fixture was consumed"
        assert len(visual_requests) == 7, "Expected advisory/failure/stale reviews, oversized-mesh recovery and explicit input choices"
        assert len(diagnosis_requests) == 3, "Expected manual failed-call diagnosis including a discarded stale response"
        assert len(mesh_specs) == 4, "Expected passed, stale, failed and current-model mesh jobs"
        assert not page_errors, "Uncaught browser errors: " + repr(page_errors)
        browser.close()


def main():
    with tempfile.TemporaryDirectory(prefix="femm-browser-") as directory:
        folder = Path(directory)
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        url = "http://127.0.0.1:" + str(port)
        with (folder / "server.log").open("w+") as log:
            process = subprocess.Popen([sys.executable, "-u", "-c", SERVER, str(folder / "data"), str(port)],
                                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            try:
                wait_for_server(process, url, log)
                exercise_browser(url, folder)
                print("Browser smoke passed: real auth/preview/export/queue/mesh gate; informational routing, advisory visual review, failed-call diagnosis; fixture AI and worker completion.")
            finally:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


if __name__ == "__main__":
    main()
