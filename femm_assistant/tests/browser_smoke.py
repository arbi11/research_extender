"""Browser integration smoke test, run separately from unittest discovery.

Install Playwright and Chromium, then run ``python tests/browser_smoke.py``.
The real FastAPI app supplies authentication, geometry validation, previews,
exports and queued jobs. Browser routes provide AI proposal and completed-job
fixtures: this test does not contact an AI provider or run FEMM. Native solver
and provider-adapter tests cover those separate boundaries.
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

    proposals = deque()
    ai_requests = []
    completed_specs = []
    page_errors = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(accept_downloads=True, viewport={"width": 1440, "height": 1100})
        page = context.new_page()
        page.on("pageerror", lambda error: page_errors.append(str(error)))
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

        def complete_job_fixture(route):
            if route.request.url.endswith("/field.svg"):
                route.fulfill(content_type="image/svg+xml", body=FIELD_FIXTURE)
                return
            response = route.fetch()
            assert response.status == 200
            job = response.json()
            assert job["status"] == "queued", "The browser test must not start a real solver"
            completed_specs.append(job["spec"])
            job["status"] = "completed"
            job["result"] = {"energy": {"value": .001, "unit": "J"},
                             "by_0": {"value": .0001, "unit": "T"},
                             "runtime": {"value": .02, "unit": "s"}}
            route.fulfill(status=200, json=job)

        page.route("**/api/plan/interpret", ai_fixture)
        page.route(re.compile(r".*/api/jobs/[a-f0-9]{32}(?:/field\.svg)?$"), complete_job_fixture)
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
        expect(page.locator("#solve")).to_be_enabled()
        expect(page.locator("#source")).to_be_enabled()
        expect(page.locator("#geometry")).to_be_visible()
        wire = json.loads(page.locator("#plan-json").input_value())
        assert wire["regions"][0]["shape"]["type"] == "circle"
        with page.expect_download() as download:
            page.locator("#source").click()
        assert_source_archive(download.value, folder, "plan")

        # Editing a valid preview invalidates it immediately. The server then
        # rejects an invalid material reference and the UI stays blocked.
        page.locator("#plan-editor > summary").click()
        invalid = json.loads(json.dumps(wire))
        invalid["regions"][0]["material"] = "undefined_material"
        page.locator("#plan-json").fill(json.dumps(invalid))
        expect(page.locator("#solve")).to_be_disabled()
        expect(page.locator("#source")).to_be_disabled()
        page.locator("#validate-plan").click()
        expect(page.locator("#message")).to_have_class(re.compile(r".*error.*"))
        expect(page.locator("#geometry")).to_be_hidden()
        page.locator("#plan-json").fill(json.dumps(wire))
        page.locator("#validate-plan").click()
        expect(page.locator("#solve")).to_be_enabled()

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
        expect(page.locator("#solve")).to_be_enabled()

        # A new AI-created topology uses no inherited model. Its rectangle is
        # previewed by the real compiler and later submitted through the real API.
        rectangle = json.loads(json.dumps(wire))
        rectangle["title"] = "Rectangular conductor from a browser AI fixture"
        rectangle["regions"][0]["shape"] = {"type": "rectangle", "x_min": -1, "x_max": 1,
                                             "y_min": -1, "y_max": 1}
        proposals.append({"spec": rectangle, "explanation": "Created the requested rectangle.", "questions": [], "attempts": 1})
        page.locator("#request-action").select_option("new")
        page.locator("#prompt").fill("Create the fully specified rectangular conductor model.")
        page.locator("#interpret").click()
        expect(page.locator("#solve")).to_be_enabled()
        expect(page.locator("#plan-summary")).to_contain_text(rectangle["title"])
        assert ai_requests[-1]["current"] is None

        modified = json.loads(json.dumps(rectangle))
        modified["circuits"][0]["current_a"] = 2
        proposals.append({"spec": modified, "explanation": "Changed only the current to 2 A.", "questions": [], "attempts": 1})
        page.locator("#request-action").select_option("modify")
        page.locator("#prompt").fill("Set the current to 2 A and preserve all geometry.")
        page.locator("#interpret").click()
        expect(page.locator("#explanation")).to_contain_text("Changed only")
        expect(page.locator("#solve")).to_be_enabled()
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
        expect(page.locator("#solve")).to_be_enabled()

        # A valid candidate with questions remains gated until explicitly resolved.
        proposals.append({"spec": modified, "explanation": "Confirm the stated material assumption.",
                          "questions": ["Is relative permeability 1 correct for the conductor?"], "attempts": 1})
        page.locator("#prompt").fill("Review the conductor material assumption.")
        page.locator("#interpret").click()
        expect(page.locator("#questions")).to_be_visible()
        expect(page.locator("#solve")).to_be_disabled()
        expect(page.locator("#source")).to_be_disabled()
        page.locator("#resolved").check()
        expect(page.locator("#solve")).to_be_enabled()

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
        expect(page.locator("#solve")).to_be_enabled()
        page.locator("#solve").click()
        expect(page.locator("#job-status")).to_have_text("COMPLETED")
        expect(page.locator("#result-body")).to_be_visible()
        expect(page.locator("#quantities")).to_contain_text("Probe 1")
        expect(page.locator("#job-model")).to_contain_text(modified["title"])
        assert completed_specs[-1]["kind"] == "magnetostatic_plan"
        assert completed_specs[-1]["circuits"][0]["current_a"] == 2

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
        expect(page.locator("#solve")).to_be_enabled()
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
                print("Browser smoke passed: real auth/preview/export/queue; fixture AI and completed-job UI.")
            finally:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


if __name__ == "__main__":
    main()
