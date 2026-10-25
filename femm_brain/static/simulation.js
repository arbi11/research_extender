"use strict";
(() => {
  const $ = id => document.getElementById(id);
  const state = {cases: [], plan: null, enabled: false, generation: 0, search: 0, preview: null};
  const el = (tag, text, cls) => { const node = document.createElement(tag); if (text != null) node.textContent = text; if (cls) node.className = cls; return node; };
  const message = (text, error = false) => { $("study-message").textContent = text; $("study-message").classList.toggle("error", error); };
  async function api(path, options) {
    const response = await fetch(path, options); const body = await response.json();
    if (!response.ok) throw new Error(body.error || "Study request failed");
    return body;
  }
  function parameters() {
    const values = {};
    $("study-parameters").querySelectorAll("input").forEach(input => {
      if (!input.checkValidity()) throw new Error("Check the parameter ranges and units");
      values[input.dataset.parameter] = Number(input.value);
    });
    return values;
  }
  function busy(value) {
    document.querySelectorAll(".study-panel input,.study-panel select,.study-panel button").forEach(n => n.disabled = value);
    if (!value) $("study-solve").disabled = !state.enabled || !state.plan;
  }
  function invalidate() {
    ++state.generation; state.plan = null;
    $("study-solve").disabled = true;
    for (const format of ["lua", "json"]) { const a = $("study-" + format); a.hidden = true; a.removeAttribute("href"); }
    for (const id of ["preview", "sources", "knowledge", "assumptions", "reference"]) $("study-" + id).replaceChildren();
    if (state.preview) { URL.revokeObjectURL(state.preview); state.preview = null; }
    $("study-result").hidden = true;
  }
  function renderParameters(item, overrides = {}) {
    $("study-parameters").replaceChildren(...item.parameters.map(p => {
      const label = el("label", p.description + " [" + p.unit + "]");
      const input = el("input"); input.type = "number"; input.step = p.integer ? "1" : "any";
      input.min = p.minimum; input.max = p.maximum; input.required = true;
      input.dataset.parameter = p.name; input.value = overrides[p.name] ?? p.default;
      input.addEventListener("input", () => { ++state.search; invalidate(); message("Generate the model again to freeze the edited parameters and checks."); });
      label.append(input); return label;
    }));
  }
  async function choose(id, overrides = {}) {
    const item = state.cases.find(c => c.id === id); if (!item) return;
    ++state.search; invalidate(); $("study-select").value = id;
    $("study-selection").replaceChildren(el("p", "Selected: " + item.title + ". The parameter fields below define the model."));
    renderParameters(item, overrides); await generate();
  }
  async function generate() {
    invalidate(); const generation = state.generation;
    try {
      const params = parameters(); message("Building from the selected knowledge and source parameters…");
      const plan = await api("/api/simulation-plan?case=" + encodeURIComponent($("study-select").value) + "&parameters=" + encodeURIComponent(JSON.stringify(params)));
      if (generation !== state.generation) return;
      state.plan = plan;
      if (state.preview) URL.revokeObjectURL(state.preview);
      state.preview = URL.createObjectURL(new Blob([plan.preview_svg], {type: "image/svg+xml"}));
      const img = el("img"); img.src = state.preview; img.alt = "Generated geometry for " + plan.title;
      $("study-preview").replaceChildren(img);
      const sources = plan.knowledge_sources || [];
      $("study-sources").replaceChildren(el("h3", "Source evidence"), ...sources.map(source => {
        const a = el("a", source.title || source.id); const url = source.url || source.source_url;
        if (url && /^https?:\/\//.test(url)) { a.href = url; a.target = "_blank"; a.rel = "noopener noreferrer"; }
        return a;
      }));
      $("study-knowledge").replaceChildren(el("h3", "Follow the knowledge used by this model"), ...plan.knowledge_node_ids.map(id => {
        const b = el("button", id, "small-button"); b.type = "button";
        b.addEventListener("click", () => window.dispatchEvent(new CustomEvent("brain:select-node", {detail: {id}}))); return b;
      }));
      const list = el("ul"); [...plan.assumptions, ...plan.limits].forEach(s => list.append(el("li", s)));
      $("study-assumptions").replaceChildren(list);
      const ref = plan.frozen_reference;
      $("study-reference").textContent = ref.applicable ? "Frozen reference: " + ref.value + " · " + ref.metric + " · " + ref.kind + ". Acceptance tolerance: " + (ref.relative_tolerance * 100) + "%." : "Reference unassessed: " + ref.reason;
      for (const fmt of ["lua", "json"]) {
        const a = $("study-" + fmt); a.hidden = false;
        a.href = "/api/simulation-download?case=" + encodeURIComponent(plan.case_id) + "&parameters=" + encodeURIComponent(JSON.stringify(plan.parameters)) + "&plan_sha256=" + encodeURIComponent(plan.plan_sha256) + "&format=" + fmt;
        a.download = plan.case_id + "." + fmt;
      }
      $("study-solve").disabled = !state.enabled;
      message("Model ready. Physics, geometry and reference are frozen before execution. No LLM calls.");
    } catch (error) { if (generation === state.generation) { state.plan = null; message(error.message, true); } }
  }
  function showResult(job) {
    const report = job.report; $("study-result").hidden = false;
    $("study-checks").replaceChildren(...Object.entries(report.checks).map(([name, value]) => {
      const p = el("p", name + ": " + value.status + (value.error ? " · " + value.error : ""), value.status === "failed" ? "error" : ""); return p;
    }));
    const table = el("table");
    Object.entries(report.metrics || {}).forEach(([name, value]) => { const row = el("tr"); row.append(el("th", name), el("td", Number(value).toPrecision(8))); table.append(row); });
    $("study-metrics").replaceChildren(table);
    const base = "/api/simulation-artifacts/" + job.id + "/";
    $("study-native-images").replaceChildren(...["mesh.svg", "field.svg"].filter(name => report.artifacts?.some(a => a.filename === name)).map(name => {
      const figure = el("figure"), img = el("img"); img.src = base + name; img.alt = "Native " + name.replace(".svg", "") + " for " + report.case_id;
      figure.append(img, el("figcaption", name === "mesh.svg" ? "Actual native mesh" : "Solved field samples")); return figure;
    }));
    const available = new Set((report.artifacts || []).map(a => a.filename));
    available.add("report.json"); available.add("job.json"); if (report.execution) available.add("execution.json");
    $("study-artifacts").replaceChildren(...["model.fem", "model.ans", "model.lua", "plan.json", "report.json", "metrics.csv", "execution.json", "job.json"].filter(name => available.has(name)).map(name => {
      const a = el("a", name, "download-button"); a.href = base + name; a.download = name; return a;
    }));
    message("Native study: " + report.status.replaceAll("_", " ") + ". Target and convergence scope are recorded in the report.", report.status === "failed");
  }
  async function poll(id) {
    try {
      const job = await api("/api/simulation-jobs/" + id);
      if (["queued", "running"].includes(job.status)) { message("Native FEMM: " + job.status + " · no network or AI key."); setTimeout(() => poll(id), 1000); return; }
      if (job.report) showResult(job); else message(job.error || "Native execution failed", true);
    } catch (error) { message(error.message, true); }
    busy(false);
  }
  $("study-form").addEventListener("submit", e => { e.preventDefault(); generate(); });
  $("study-select").addEventListener("change", () => { $("study-request").value = ""; choose($("study-select").value); });
  $("study-search-form").addEventListener("submit", async e => {
    e.preventDefault();
    invalidate();
    const search = ++state.search;
    try {
      const found = await api("/api/simulation-request?q=" + encodeURIComponent($("study-request").value));
      if (search !== state.search) return;
      $("study-selection").replaceChildren(el("p", found.status + " · " + (found.messages || []).join(" ")));
      if (found.status === "ready") await choose(found.selected_case_id, found.parameters);
      else if (found.status === "ambiguous") (found.candidates || []).forEach(c => {
        const b = el("button", "Choose " + c.title, "small-button"); b.type = "button"; b.addEventListener("click", () => choose(c.id, found.parameters)); $("study-selection").append(b);
      });
    } catch (error) { if (search === state.search) message(error.message, true); }
  });
  $("study-request").addEventListener("input", () => { ++state.search; invalidate(); message("Find a supported study, or generate the selected study explicitly, before solving."); });
  $("study-solve").addEventListener("click", async () => {
    if (!state.plan) return;
    busy(true); $("study-result").hidden = true;
    try { const job = await api("/api/simulations", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({case_id: state.plan.case_id, parameters: state.plan.parameters, plan_sha256: state.plan.plan_sha256})}); poll(job.id); }
    catch (error) { message(error.message, true); busy(false); }
  });
  api("/api/simulations").then(data => {
    state.cases = data.cases; state.enabled = data.solver_enabled;
    $("study-select").replaceChildren(...state.cases.map(c => { const o = el("option", c.title); o.value = c.id; return o; }));
    $("study-engine").textContent = state.enabled ? "Solves run in a separate Docker container with networking disabled. Existing assistant services are not used." : "Planning and downloads work offline. To solve, start this server with --enable-solver and a FEMM Docker image; see the README.";
    if (state.cases.length) choose(state.cases[0].id);
  }).catch(error => message(error.message, true));
})();
