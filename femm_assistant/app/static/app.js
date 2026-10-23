"use strict";
const $ = id => document.getElementById(id);
const fields = [
  ["width_mm", "Core width (mm)", 10, 200, 1], ["height_mm", "Core height (mm)", 10, 200, 1],
  ["limb_mm", "Limb thickness (mm)", 2, 40, .5], ["gap_mm", "Gap at each pole (mm)", .2, 20, .1],
  ["depth_mm", "Stack depth (mm)", 1, 200, 1], ["turns", "Winding turns", 1, 10000, 1],
  ["current_a", "Current (A)", 0, 100, .1], ["relative_permeability", "Linear relative permeability", 1, 10000, 1],
  ["mesh_mm", "Core mesh size (mm)", .5, 5, .1],
];
let config, currentJob, polling, previewTimer, previewRequest, busy = false;
let token = sessionStorage.getItem("femm_token") || "";
const urls = new Map();
function message(text, error = false) { $("message").textContent = text; $("message").classList.toggle("error", error); }
async function api(path, body) {
  const response = await fetch(path, { method: body === undefined ? "GET" : "POST", headers: {"Authorization": "Bearer " + token, ...(body === undefined ? {} : {"Content-Type": "application/json"})}, ...(body === undefined ? {} : {body: JSON.stringify(body)}) });
  if (!response.ok) {
    let detail;
    try { const data = await response.json(); detail = typeof data.detail === "string" ? data.detail : data.detail?.map(x => `${x.loc.slice(1).join(".")}: ${x.msg}`).join("; "); } catch (_) { /* plain error response */ }
    throw new Error(detail || `Request failed (${response.status}).`);
  }
  return response;
}
function spec() {
  if (!$("parameters").reportValidity()) throw new Error("Check the highlighted model parameters.");
  const values = {geometry: "c_core"};
  fields.forEach(([key]) => { values[key] = Number($(key).value); });
  return values;
}
function populate(values) { fields.forEach(([key]) => { $(key).value = values[key]; }); }
function imageBlob(id, blob) { if (urls.has(id)) URL.revokeObjectURL(urls.get(id)); const url = URL.createObjectURL(blob); urls.set(id, url); $(id).src = url; }
async function preview() {
  const serial = Symbol(); previewRequest = serial;
  try {
    const values = spec(); const response = await api("/api/preview", values); const blob = await response.blob();
    if (previewRequest !== serial) return;
    imageBlob("geometry", blob); $("dimensions").textContent = `${values.width_mm} × ${values.height_mm} mm`; message("Geometry preview ready. Review parameters before solving.");
  } catch (error) { if (previewRequest === serial) message(error.message, true); }
}
async function saveResponse(response, name) {
  const url = URL.createObjectURL(await response.blob()); const a = document.createElement("a"); a.href = url; a.download = name; document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function controls() {
  const unresolved = !$("questions").hidden && !$("resolved").checked;
  $("solve").disabled = busy || unresolved || !config?.solver_detected;
  $("interpret").disabled = busy || !config?.ai_available;
}
$("access-form").addEventListener("submit", async event => { event.preventDefault(); token = $("token").value.trim(); sessionStorage.setItem("femm_token", token); await preview(); });
$("resolved").addEventListener("change", controls);
$("interpret").addEventListener("click", async () => {
  busy = true; controls(); message("Interpreting your request…");
  try {
    const response = await api("/api/interpret", {prompt: $("prompt").value, current: spec()}); const proposal = await response.json();
    populate(proposal.spec); $("explanation").textContent = proposal.explanation; $("explanation").hidden = false;
    $("questions").hidden = proposal.questions.length === 0; $("resolved").checked = false;
    $("question-list").replaceChildren(...proposal.questions.map(question => { const li = document.createElement("li"); li.textContent = question; return li; }));
    await preview();
  } catch (error) { message(error.message, true); }
  finally { busy = false; controls(); }
});
$("source").addEventListener("click", async () => { try { await saveResponse(await api("/api/export", spec()), "femm-c-core-source.zip"); message("Source downloaded. FEMM 5 is required to run it."); } catch (error) { message(error.message, true); } });
$("solve").addEventListener("click", async () => {
  busy = true; controls();
  try {
    const job = await (await api("/api/jobs", spec())).json(); currentJob = job.id; sessionStorage.setItem("femm_job", currentJob);
    $("results").hidden = false; $("result-body").hidden = true; $("log").hidden = true; $("results").scrollIntoView({behavior: "smooth", block: "start"});
    message("Simulation submitted. You can leave this tab open while the job runs."); await poll();
  } catch (error) { message(error.message, true); busy = false; controls(); }
});
async function poll() {
  clearTimeout(polling);
  try {
    const job = await (await api(`/api/jobs/${currentJob}`)).json(); $("job-status").textContent = job.status.toUpperCase();
    $("job-message").textContent = job.status === "queued" ? "Queued — one model is solved at a time to keep hosting costs low." : job.status === "running" ? "FEMM is meshing, solving and extracting the field." : job.status === "failed" ? job.error : "Finite-element solution completed.";
    if (["queued", "running"].includes(job.status)) { busy = true; controls(); polling = setTimeout(poll, 2000); return; }
    busy = false; controls(); $("log").hidden = false;
    if (job.status === "completed") {
      const labels = {force_x: "Armature force, x", force_y: "Armature force, y", energy: "Magnetic energy", flux_linkage: "Flux linkage", current: "Coil current", runtime: "Solve & extraction time"};
      $("quantities").replaceChildren(...Object.entries(job.result).map(([key, quantity]) => { const row = document.createElement("tr"); const label = document.createElement("td"); const value = document.createElement("td"); label.textContent = labels[key] || key; value.textContent = `${Number(quantity.value).toPrecision(5)} ${quantity.unit}`; row.append(label, value); return row; }));
      imageBlob("field", await (await api(`/api/jobs/${currentJob}/field.svg`)).blob()); $("result-body").hidden = false;
    }
  } catch (error) { $("job-message").textContent = error.message; busy = false; controls(); }
}
$("download").addEventListener("click", async () => { try { await saveResponse(await api(`/api/jobs/${currentJob}/download.zip`), `femm-${currentJob}.zip`); } catch (error) { message(error.message, true); } });
$("log").addEventListener("click", async () => { try { await saveResponse(await api(`/api/jobs/${currentJob}/solver.log`), "solver.log"); } catch (error) { message(error.message, true); } });
async function init() {
  try {
    config = await (await api("/api/config")).json(); $("access").hidden = !config.auth_required; $("token").value = token;
    $("availability").textContent = config.solver_detected ? "FEMM 5 launcher detected" : "Source export available · Solver not installed";
    $("ai-note").textContent = config.ai_available ? "AI proposes parameters. You review the model and choose when to solve." : "Natural-language configuration is not enabled on this server. Edit the parameters below.";
    fields.forEach(([key, label, min, max, step]) => { const element = document.createElement("label"); element.htmlFor = key; element.textContent = label; const input = document.createElement("input"); Object.assign(input, {id: key, type: "number", min, max, step, required: true}); input.value = config.defaults[key]; element.append(input); $("parameters").append(element); });
    $("parameters").addEventListener("submit", event => event.preventDefault());
    $("parameters").addEventListener("input", () => { clearTimeout(previewTimer); previewTimer = setTimeout(preview, 350); });
    $("assumptions").replaceChildren(...config.assumptions.map(text => { const li = document.createElement("li"); li.textContent = text; return li; })); controls();
    if (!config.auth_required || token) await preview(); else message("Enter your access token to preview, export or solve a model.");
    const saved = sessionStorage.getItem("femm_job"); if (saved && (!config.auth_required || token)) { currentJob = saved; $("results").hidden = false; await poll(); }
  } catch (error) { message(error.message, true); }
}
init();
