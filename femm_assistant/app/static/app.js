"use strict";
const $ = id => document.getElementById(id);
const fields = [
  ["width_mm", "Core width (mm)", 10, 200, 1], ["height_mm", "Core height (mm)", 10, 200, 1],
  ["limb_mm", "Limb thickness (mm)", 2, 40, .5], ["gap_mm", "Gap at each pole (mm)", .2, 20, .1],
  ["depth_mm", "Stack depth (mm)", 1, 200, 1], ["turns", "Winding turns", 1, 10000, 1],
  ["current_a", "Current (A)", 0, 100, .1], ["relative_permeability", "Linear relative permeability", 1, 10000, 1],
  ["mesh_mm", "Core mesh size (mm)", .5, 5, .1],
];
const generalPlaceholder = "Create a round wire of radius 1 mm carrying 1 A in the +z direction, centered in a circular air domain of radius 10 mm. Use linear relative permeability 1, planar depth 20 mm, mesh 0.5 mm and A=0 at the outer boundary. Probe the field at (2, 0) mm.";
const corePlaceholder = "Use a 1.5 mm gap, 600 turns and 2 A. Keep the core 60 × 60 mm with 10 mm limbs.";
let config, currentJob, polling, previewTimer, previewRequest, busy = false;
let mode = sessionStorage.getItem("femm_mode") === "ccore" ? "ccore" : "plan";
let token = sessionStorage.getItem("femm_token") || "";
let validatedPlan = null, validatedEditor = null, validatedCore = null;
const questions = {
  plan: {items: [], explanation: "", resolved: false, hasSpec: true},
  ccore: {items: [], explanation: "", resolved: false, hasSpec: true},
};
const urls = new Map();

function message(text, error = false) {
  $("message").textContent = text;
  $("message").classList.toggle("error", error);
}
async function api(path, body) {
  const response = await fetch(path, {
    method: body === undefined ? "GET" : "POST",
    headers: {"Authorization": "Bearer " + token, ...(body === undefined ? {} : {"Content-Type": "application/json"})},
    ...(body === undefined ? {} : {body: JSON.stringify(body)}),
  });
  if (!response.ok) {
    let detail;
    try {
      const data = await response.json();
      if (typeof data.detail === "string") detail = data.detail;
      else if (Array.isArray(data.detail)) detail = data.detail.map(item => (item.loc || []).slice(1).join(".") + ": " + item.msg).join("; ");
    } catch (_) { /* A plain-text response has no structured error details. */ }
    throw new Error(detail || "Request failed (" + response.status + ").");
  }
  return response;
}
function coreSpec(check = true) {
  if (check && !$("parameters").reportValidity()) throw new Error("Check the highlighted model parameters.");
  const values = {geometry: "c_core"};
  fields.forEach(([key]) => { values[key] = Number($(key).value); });
  return values;
}
function editorPlan() {
  if (!$("plan-json").value.trim()) throw new Error("Describe a model or load an example first.");
  let value;
  try { value = JSON.parse($("plan-json").value); }
  catch (_) { throw new Error("The model plan is not valid JSON. Check the model plan or apply your request again."); }
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("The model plan must be a JSON object.");
  return value;
}
function hasUnresolvedQuestions() {
  const state = questions[mode];
  return state.items.length > 0 && (!state.hasSpec || !state.resolved);
}
function isReady() {
  return mode === "plan"
    ? validatedPlan !== null && validatedEditor === $("plan-json").value
    : validatedCore !== null && validatedCore === JSON.stringify(coreSpec(false));
}
function readySpec() {
  if (hasUnresolvedQuestions()) throw new Error("Address the clarification questions before exporting or solving.");
  if (!isReady()) throw new Error("Validate and preview the current model before exporting or solving.");
  return mode === "plan" ? structuredClone(validatedPlan) : coreSpec();
}
function requestFingerprint() {
  return JSON.stringify({
    mode,
    prompt: $("prompt").value,
    draft: mode === "plan" ? $("plan-json").value : coreSpec(false),
    action: mode === "plan" ? $("request-action").value : "modify",
  });
}
function imageBlob(id, blob) {
  if (urls.has(id)) URL.revokeObjectURL(urls.get(id));
  const url = URL.createObjectURL(blob);
  urls.set(id, url);
  $(id).src = url;
  $(id).hidden = false;
}
function emptyPreview(text) {
  $("geometry").hidden = true;
  $("preview-empty").textContent = text;
  $("preview-empty").hidden = false;
  $("dimensions").textContent = "";
}
function invalidateModel(text) {
  previewRequest = Symbol();
  if (mode === "plan") {
    validatedPlan = null;
    validatedEditor = null;
  } else validatedCore = null;
  if (questions[mode].items.length) {
    questions[mode].resolved = false;
    $("resolved").checked = false;
  }
  emptyPreview(text || "Validate the current model to update its preview.");
  controls();
}
function renderQuestions() {
  const state = questions[mode];
  $("explanation").textContent = state.explanation;
  $("explanation").hidden = !state.explanation;
  $("questions").hidden = state.items.length === 0;
  $("question-list").replaceChildren(...state.items.map(question => {
    const item = document.createElement("li"); item.textContent = question; return item;
  }));
  $("resolved").checked = state.resolved;
  $("resolved-label").hidden = !state.hasSpec;
  $("question-help").textContent = state.hasSpec
    ? "Add these details to your request and apply it again, or address them in the model below."
    : "Add the missing details to your description, then apply the request again to create a model.";
}
function resetQuestions() {
  questions[mode] = {items: [], explanation: "", resolved: false, hasSpec: true};
  renderQuestions();
}
function summarizePlan(plan) {
  const shapes = plan.regions || [];
  const circuits = plan.circuits || [];
  const parts = [
    plan.title || "Magnetic model",
    shapes.length + (shapes.length === 1 ? " region" : " regions"),
    circuits.length + (circuits.length === 1 ? " circuit" : " circuits"),
    "depth " + plan.depth_mm + " mm",
    "mesh " + plan.mesh_mm + " mm",
  ];
  $("plan-summary").textContent = parts.join(" · ");
  $("model-tag").textContent = "CUSTOM MODEL";
  $("dimensions").textContent = shapes.length + " regions";
}
function setPlan(value) {
  $("plan-json").value = value ? JSON.stringify(value, null, 2) : "";
  invalidateModel(value ? "Model received. Checking geometry and assignments…" : "Add the missing details to create a model.");
  $("plan-summary").textContent = value ? "Checking the proposed model…" : "No model has been created yet.";
}
function controls() {
  const ready = Boolean(config) && isReady() && !hasUnresolvedQuestions();
  const needsCurrent = mode === "plan" && $("request-action").value === "modify";
  $("solve").disabled = busy || !ready || !config?.solver_detected;
  $("source").disabled = busy || !ready;
  $("interpret").disabled = busy || !config?.ai_available || $("prompt").value.trim().length < 5 || (needsCurrent && !isReady());
  $("validate-plan").disabled = busy || !$("plan-json").value.trim();
  $("load-example").disabled = busy || !config?.plan_example;
  $("mode").disabled = busy;
  $("request-action").disabled = busy;
  $("interpret").textContent = mode === "ccore" ? "Apply request ↗" : needsCurrent ? "Update model ↗" : "Create model ↗";
}
async function preview() {
  const serial = Symbol();
  previewRequest = serial;
  const requestedMode = mode;
  let snapshot;
  try {
    const values = mode === "plan" ? editorPlan() : coreSpec();
    snapshot = mode === "plan" ? $("plan-json").value : JSON.stringify(values);
    const response = await api(mode === "plan" ? "/api/plan/preview" : "/api/preview", values);
    const blob = await response.blob();
    if (previewRequest !== serial || requestedMode !== mode) return false;
    const now = mode === "plan" ? $("plan-json").value : JSON.stringify(coreSpec(false));
    if (snapshot !== now) return false;
    if (mode === "plan") {
      validatedPlan = structuredClone(values);
      validatedEditor = snapshot;
      summarizePlan(values);
      if (questions[mode].items.length && !questions[mode].hasSpec) {
        questions[mode].hasSpec = true;
        renderQuestions();
      }
    } else {
      validatedCore = snapshot;
      $("dimensions").textContent = values.width_mm + " × " + values.height_mm + " mm";
    }
    imageBlob("geometry", blob);
    $("preview-empty").hidden = true;
    message("Geometry and assignments validated. Review the model, then solve.");
    controls();
    return true;
  } catch (error) {
    if (previewRequest === serial && requestedMode === mode) {
      if (mode === "plan") { validatedPlan = null; validatedEditor = null; $("plan-summary").textContent = "The model needs attention before it can be solved."; }
      else validatedCore = null;
      emptyPreview("The current model could not be validated.");
      message(error.message, true);
      controls();
    }
    return false;
  }
}
async function showMode() {
  clearTimeout(previewTimer);
  previewRequest = Symbol();
  $("mode").value = mode;
  sessionStorage.setItem("femm_mode", mode);
  const general = mode === "plan";
  $("plan-controls").hidden = !general;
  $("plan-request-controls").hidden = !general;
  $("ccore-controls").hidden = general;
  $("prompt").placeholder = general ? generalPlaceholder : corePlaceholder;
  $("mode-note").textContent = general
    ? "Create new geometry from closed circles, rectangles and polygons. Assign linear materials, circuits and a surrounding air region."
    : "Use the original parameterized actuator to explore force, energy and flux linkage.";
  $("canvas-note").textContent = general
    ? "Closed contours and region assignments are checked before solving."
    : "Steel core & armature · Copper-colored winding";
  $("preview-scope").textContent = general
    ? "Linear materials · Zero magnetic vector potential at the outer boundary"
    : "Two pole gaps · Fixed linear permeability";
  $("model-tag").textContent = general ? "NEW MODEL" : "C-CORE";
  $("assumptions").replaceChildren(...(general ? config.plan_capabilities || [
    "2D planar DC magnetostatics; all lengths in millimeters.",
    "Linear isotropic materials and a zero-A outer boundary.",
    "Closed circle, rectangle and polygon contours; no touching or crossing contours.",
  ] : config.assumptions).map(text => { const item = document.createElement("li"); item.textContent = text; return item; }));
  $("ai-note").textContent = config.ai_available
    ? general ? "The AI proposes geometry and physical assignments. The validated model generates the FEMM script." : "AI proposes parameters. You review the model and choose when to solve."
    : "Natural-language requests are not enabled on this server. You can load an example or edit the model manually.";
  renderQuestions();
  emptyPreview(general ? "Describe a model or load an example to preview its geometry." : "Preparing the C-core preview…");
  controls();
  if ((!config.auth_required || token) && (!general || $("plan-json").value.trim())) await preview();
  else message(config.auth_required && !token ? "Enter your access token to preview, export or solve a model." : "Describe a new model or load the wire example to get started.");
}
async function saveResponse(response, name) {
  const url = URL.createObjectURL(await response.blob());
  const link = document.createElement("a"); link.href = url; link.download = name;
  document.body.append(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
$("access-form").addEventListener("submit", async event => {
  event.preventDefault();
  token = $("token").value.trim();
  sessionStorage.setItem("femm_token", token);
  if (mode === "plan" && !$("plan-json").value.trim()) {
    try {
      await api("/api/plan/preview", config.plan_example);
      message("Connected. Describe a new model or load the wire example.");
    } catch (error) { message(error.message, true); }
  } else await preview();
});
$("mode").addEventListener("change", async () => { mode = $("mode").value; await showMode(); });
$("request-action").addEventListener("change", controls);
$("prompt").addEventListener("input", controls);
$("resolved").addEventListener("change", () => { questions[mode].resolved = $("resolved").checked; controls(); });
$("plan-json").addEventListener("input", () => {
  if (mode !== "plan") return;
  invalidateModel("The model plan changed. Choose Validate & preview to check it.");
  $("plan-summary").textContent = "Unvalidated draft — changes have not been applied to the preview.";
  message("Model plan changed. Validate it before solving or downloading a script.");
});
$("validate-plan").addEventListener("click", preview);
$("load-example").addEventListener("click", async () => {
  resetQuestions();
  setPlan(config.plan_example);
  $("request-action").value = "modify";
  await preview();
});
$("interpret").addEventListener("click", async () => {
  const requestedMode = mode;
  const fingerprint = requestFingerprint();
  const prompt = $("prompt").value;
  busy = true; controls(); message(mode === "plan" ? "Preparing and validating the model…" : "Interpreting your request…");
  try {
    const modifyingPlan = mode === "plan" && $("request-action").value === "modify";
    if (modifyingPlan && !isReady()) throw new Error("Validate the current model before asking the AI to modify it.");
    const current = mode === "plan" ? modifyingPlan ? structuredClone(validatedPlan) : null : coreSpec();
    const response = await api(mode === "plan" ? "/api/plan/interpret" : "/api/interpret", {prompt, current});
    const proposal = await response.json();
    if (requestedMode !== mode || fingerprint !== requestFingerprint()) {
      message("Your request or model changed while the AI was working. Its response was not applied; submit your current request again.");
      return;
    }
    if (mode === "plan") {
      if (proposal.spec || !modifyingPlan) setPlan(proposal.spec);
    } else {
      fields.forEach(([key]) => { $(key).value = proposal.spec[key]; });
      invalidateModel();
    }
    questions[mode] = {
      items: Array.isArray(proposal.questions) ? proposal.questions : [],
      explanation: proposal.explanation || "",
      resolved: false,
      hasSpec: Boolean(proposal.spec),
    };
    renderQuestions();
    if (proposal.spec) await preview();
    else message("More information is needed. Update your description with the clarification details.");
  } catch (error) { message(error.message, true); }
  finally { busy = false; controls(); }
});
$("source").addEventListener("click", async () => {
  try {
    const requestedMode = mode;
    const values = readySpec();
    await saveResponse(await api(requestedMode === "plan" ? "/api/plan/export" : "/api/export", values), requestedMode === "plan" ? "femm-geometry-source.zip" : "femm-c-core-source.zip");
    message("Source downloaded. FEMM 5 is required to run the portable script.");
  } catch (error) { message(error.message, true); }
});
$("solve").addEventListener("click", async () => {
  busy = true; controls();
  try {
    const values = readySpec();
    const job = await (await api(mode === "plan" ? "/api/plan/jobs" : "/api/jobs", values)).json();
    currentJob = job.id; sessionStorage.setItem("femm_job", currentJob);
    $("results").hidden = false; $("result-body").hidden = true; $("log").hidden = true;
    $("results").scrollIntoView({behavior: "smooth", block: "start"});
    message("Simulation submitted. Results will use this submitted model, even if you edit the draft.");
    await poll();
  } catch (error) { message(error.message, true); busy = false; controls(); }
});
function quantityLabel(key, general) {
  const labels = {force_x: general ? "Region force, x" : "Armature force, x", force_y: general ? "Region force, y" : "Armature force, y", energy: "Magnetic energy", flux_linkage: "Flux linkage", current: "Coil current", runtime: "Solve & extraction time"};
  if (labels[key]) return labels[key];
  if (key.startsWith("current_")) return "Current · " + key.slice(8);
  if (key.startsWith("flux_linkage_")) return "Flux linkage · " + key.slice(13);
  if (key.startsWith("bx_")) return "Probe " + (Number(key.slice(3)) + 1) + " · Bx";
  if (key.startsWith("by_")) return "Probe " + (Number(key.slice(3)) + 1) + " · By";
  return key.replaceAll("_", " ");
}
async function poll() {
  clearTimeout(polling);
  const jobId = currentJob;
  try {
    const job = await (await api("/api/jobs/" + jobId)).json();
    if (jobId !== currentJob) return;
    const general = job.spec?.kind === "magnetostatic_plan";
    $("job-model").textContent = "Submitted model: " + (general ? job.spec.title : "C-core actuator");
    $("job-status").textContent = job.status.toUpperCase();
    $("job-message").textContent = job.status === "queued" ? "Queued — one model is solved at a time to keep hosting costs low."
      : job.status === "running" ? "FEMM is meshing, solving and extracting the field."
      : job.status === "failed" ? job.error : "Finite-element solution completed.";
    if (["queued", "running"].includes(job.status)) {
      busy = true; controls(); polling = setTimeout(poll, 2000); return;
    }
    busy = false; controls(); $("log").hidden = false;
    if (job.status === "completed") {
      $("quantities").replaceChildren(...Object.entries(job.result).map(([key, quantity]) => {
        const row = document.createElement("tr"), label = document.createElement("td"), value = document.createElement("td");
        label.textContent = quantityLabel(key, general);
        value.textContent = Number(quantity.value).toPrecision(5) + " " + quantity.unit;
        row.append(label, value); return row;
      }));
      const image = await (await api("/api/jobs/" + jobId + "/field.svg")).blob();
      if (jobId !== currentJob) return;
      imageBlob("field", image); $("result-body").hidden = false;
    }
  } catch (error) {
    if (jobId === currentJob) { $("job-message").textContent = error.message; busy = false; controls(); }
  }
}
$("download").addEventListener("click", async () => {
  try { await saveResponse(await api("/api/jobs/" + currentJob + "/download.zip"), "femm-" + currentJob + ".zip"); }
  catch (error) { message(error.message, true); }
});
$("log").addEventListener("click", async () => {
  try { await saveResponse(await api("/api/jobs/" + currentJob + "/solver.log"), "solver.log"); }
  catch (error) { message(error.message, true); }
});
async function init() {
  try {
    config = await (await api("/api/config")).json();
    $("access").hidden = !config.auth_required; $("token").value = token;
    $("availability").textContent = config.solver_detected ? "FEMM 5 launcher detected" : "Source export available · Solver not installed";
    if (!config.plan_example) { $("mode").querySelector('option[value="plan"]').disabled = true; mode = "ccore"; }
    fields.forEach(([key, label, min, max, step]) => {
      const element = document.createElement("label"); element.htmlFor = key; element.textContent = label;
      const input = document.createElement("input");
      Object.assign(input, {id: key, type: "number", min, max, step: key === "turns" ? 1 : "any", required: true});
      input.value = config.defaults[key]; element.append(input); $("parameters").append(element);
    });
    $("parameters").addEventListener("submit", event => event.preventDefault());
    $("parameters").addEventListener("input", () => {
      if (mode !== "ccore") return;
      clearTimeout(previewTimer); invalidateModel("The parameters changed. Updating the geometry preview…");
      previewTimer = setTimeout(preview, 350);
    });
    await showMode();
    const saved = sessionStorage.getItem("femm_job");
    if (saved && (!config.auth_required || token)) {
      currentJob = saved; $("results").hidden = false; await poll();
    }
  } catch (error) { message(error.message, true); }
}
init();
