"""Exercise the real local graph server and interactive offline viewer.

Run separately from unittest discovery after installing Playwright Chromium.
No FEMM process, LLM request or external source navigation is performed.
"""
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


ROOT = Path(__file__).resolve().parents[2]


def wait_for_server(process, url, logfile):
    for _ in range(150):
        if process.poll() is not None:
            break
        try:
            with urllib.request.urlopen(url + "/api/graph", timeout=1) as response:
                if response.status == 200:
                    return json.load(response)
        except (OSError, urllib.error.URLError):
            time.sleep(.1)
    logfile.flush()
    raise RuntimeError("Server did not start: " + Path(logfile.name).read_text()[-5000:])


def exercise_browser(url, graph, folder):
    from playwright.sync_api import expect, sync_playwright

    errors, external_requests = [], []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(accept_downloads=True, viewport={"width": 1600, "height": 1100})
        page = context.new_page()
        page.set_default_timeout(12000)
        page.on("pageerror", lambda error: errors.append(str(error)))

        def network_boundary(route):
            if not route.request.url.startswith(url + "/"):
                external_requests.append(route.request.url)
                route.abort()
            else:
                route.continue_()

        page.route("**/*", network_boundary)
        page.goto(url, wait_until="networkidle")
        expect(page.locator("#case-list button").first).to_be_visible()
        expect(page.locator("#summary")).to_contain_text("examples")
        assert page.locator("#graph canvas").count() > 0, "The actual graph renderer did not load"

        # Example navigation uses the genuine graph and explicit source evidence.
        page.locator("#case-search").fill("induction")
        page.wait_for_function("document.querySelectorAll('#case-list button').length >= 2")
        page.locator("#case-list button").first.click()
        expect(page.locator("#graph-title")).to_contain_text(re.compile("induction", re.I))
        expect(page.locator("#source-list a").first).to_be_visible()
        expect(page.locator("#node-verification")).to_contain_text(re.compile("documented|reconstructed|checked", re.I))

        # Accessible node navigation exercises expansion without relying on
        # pixel coordinates inside a canvas or hidden application state.
        page.locator("#overview").click()
        page.locator(".node-browser summary").click()
        page.locator("#collapse-node").click()
        expect(page.locator("#graph-summary")).to_contain_text("1 visible nodes")
        page.locator("#expand-node").click()
        page.wait_for_function("document.querySelectorAll('#graph-node-list button').length > 1")
        assert len(page.locator("#graph-node-list button").all()) > 1
        page.locator("#view-mode").select_option("reuse")
        page.locator("#relations").check()
        page.locator("#view-mode").select_option("hierarchy")

        # Construction is a real deterministic API request, not a browser fixture.
        page.locator("#recipe-select").select_option("stator_tooth")
        expect(page.locator("#geometry-preview svg")).to_be_visible()
        expect(page.locator("#generate-recipe")).to_be_enabled()
        expect(page.locator("#geometry-note")).to_contain_text(re.compile("reconstruct|illustrative", re.I))
        expect(page.locator("#construction-step-list li").first).to_be_visible()
        original_svg = page.locator("#geometry-preview").inner_html()
        height = page.locator('#recipe-parameters input[data-parameter="height"]')
        height.fill("10")
        page.locator("#generate-recipe").click()
        page.wait_for_function("previous => document.querySelector('#geometry-preview').innerHTML !== previous", arg=original_svg)
        expect(page.locator("#generate-recipe")).to_be_enabled()
        assert 'height' in page.locator("#download-python").get_attribute("href")

        # A primitive in the SVG and its operation are linked in both directions.
        primitive = page.locator("#geometry-preview svg [data-operation-id]").first
        operation_id = primitive.get_attribute("data-operation-id")
        primitive.dispatch_event("click")
        expect(page.locator('#operations-list button.selected')).to_have_attribute("data-operation-id", operation_id)
        selected_geometry = page.locator("#geometry-preview .geometry-selected")
        expect(selected_geometry).to_have_count(1)
        # Vertical SVG lines have zero-width geometric bounding boxes and are
        # classified as hidden by Playwright despite having a visible stroke.
        assert selected_geometry.evaluate("node => getComputedStyle(node).stroke !== 'none'")
        expect(page.locator("#node-layer")).to_contain_text("3")
        expect(page.locator("#node-signature")).to_be_visible()
        expect(page.locator("#node-signature")).to_contain_text("mi_add")
        expect(page.locator('#source-list a[href*="#page="]').first).to_be_visible()
        page.locator("#layer-3").uncheck()
        visible_ids = page.locator("#graph-node-list button").evaluate_all("nodes => nodes.map(node => node.dataset.nodeId)")
        layers = {node["id"]: node["layer"] for node in graph["nodes"]}
        assert all(layers[node_id] != 3 for node_id in visible_ids)
        page.locator("#layer-3").check()

        with page.expect_download() as downloaded:
            page.locator("#download-python").click()
        download = downloaded.value
        export = folder / download.suggested_filename
        download.save_as(str(export))
        assert download.failure() is None
        compile(export.read_text(), str(export), "exec")
        assert "from femm_brain" not in export.read_text()

        # A source preview, when present, stays visibly different from a recipe.
        page.locator("#case-search").fill("")
        page.locator("#overview").click()
        available = page.locator("#artifact-select option").all()
        assert len(available) > 1, "No bundled source-model preview is available"
        artifact_id = available[1].get_attribute("value")
        inspector_title = page.locator("#node-title").text_content()
        inspector_sources = page.locator("#source-list a").evaluate_all("nodes => nodes.map(node => node.href)")
        page.locator("#artifact-select").select_option(artifact_id)
        expect(page.locator("#recipe-select")).to_have_value("")
        expect(page.locator("#geometry-preview svg")).to_be_visible()
        expect(page.locator("#geometry-note")).to_contain_text(re.compile("source|recorded|native|original example", re.I))
        expect(page.locator("#download-python")).to_be_hidden()
        expect(page.locator("#geometry-sources a").first).to_be_visible()
        if page.locator("#node-title").text_content() == inspector_title:
            assert page.locator("#source-list a").evaluate_all("nodes => nodes.map(node => node.href)") == inspector_sources, \
                "Loading geometry changed a knowledge card's sources without changing its identity"

        artifact_directory = os.getenv("BROWSER_SMOKE_ARTIFACTS")
        if artifact_directory:
            destination = Path(artifact_directory)
            destination.mkdir(parents=True, exist_ok=True)
            page.evaluate("window.scrollTo({top: 0, left: 0, behavior: 'instant'})")
            page.screenshot(path=str(destination / "brain-source-model.png"), full_page=True)
            page.locator("#recipe-select").select_option("stator_tooth")
            expect(page.locator("#generate-recipe")).to_be_enabled()
            page.evaluate("window.scrollTo({top: 0, left: 0, behavior: 'instant'})")
            page.screenshot(path=str(destination / "brain-tooth-recipe.png"), full_page=True)

        assert not errors, "Browser JavaScript errors: " + json.dumps(errors)
        assert not external_requests, "Offline viewer requested external resources: " + json.dumps(external_requests)
        context.close()
        browser.close()
    print("Browser checks passed: source navigation, graph expansion, parameterized construction, primitive links, exports and offline resources.")


def main():
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        url = "http://127.0.0.1:" + str(port)
        with (folder / "server.log").open("w+") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", "femm_brain", "serve", "--host", "127.0.0.1",
                 "--port", str(port), "--db", str(folder / "brain.sqlite3")],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            )
            try:
                graph = wait_for_server(process, url, log)
                exercise_browser(url, graph, folder)
            finally:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


if __name__ == "__main__":
    main()
