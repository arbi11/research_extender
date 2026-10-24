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
let meshJob = null, meshPolling, meshBusy = false, meshValidation = null, planSolve = null, planRevision = 0;
let recordedRevision = null, pendingEdits = null;
let recoveryJob = null, automaticRecovery = null;
const questions = {
  plan: {items: [], explanation: "", resolved: false, hasSpec: true, decisions: []},
  ccore: {items: [], explanation: "", resolved: false, hasSpec: true, decisions: []},
};
const urls = new Map();
let visualRequest, visualBusy = false;
const failedJobs = {mesh: null, solve: null};
const diagnosisRequests = {mesh: null, solve: null};

function message(text, error = false) {
  $("message").textContent = text;
  $("message").classList.toggle("error", error);
}
async function api(path, body, extraHeaders = {}) {
  const response = await fetch(path, {
    method: body === undefined ? "GET" : "POST",
    headers: {"Authorization": "Bearer " + token, ...(body === undefined ? {} : {"Content-Type": "application/json"}), ...extraHeaders},
    ...(body === undefined ? {} : {body: JSON.stringify(body)}),
  });
  if (!response.ok) {
    let detail;
    try {
      const data = await response.json();
      if (typeof data.detail === "string") detail = data.detail;
      else if (data.detail && typeof data.detail.message === "string") {
        const summaries = [...(Array.isArray(data.detail.validation_summary) ? data.detail.validation_summary : []), ...(Array.isArray(data.detail.validation_issues) ? data.detail.validation_issues : [])];
        detail = data.detail.message + (summaries.length ? " " + summaries.map(item => typeof item === "string" ? item : item.message || item.msg || "").filter(Boolean).join("; ") : "") +
          (data.detail.audit_id ? " Audit: " + data.detail.audit_id : "");
      }
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
function planKey() {
  return mode === "plan" && isReady() ? JSON.stringify(validatedPlan) : null;
}
function hasCurrentMesh() {
  return meshValidation !== null && meshValidation.key === planKey() && meshValidation.revision === planRevision;
}
function isCurrentMeshSubmission(submitted) {
  return submitted.key === planKey() && submitted.revision === planRevision;
}
function renderPipeline() {
  const key = planKey(), ready = key !== null && !hasUnresolvedQuestions();
  const solved = planSolve?.key === key ? planSolve.status : null;
  const stages = [
    ["stage-model", ready ? "complete" : "current"],
    ["stage-mesh", hasCurrentMesh() ? "complete" : ready ? "current" : "pending"],
    ["stage-solve", solved === "completed" ? "complete" : hasCurrentMesh() ? "current" : "pending"],
    ["stage-results", solved === "completed" ? "complete" : "pending"],
  ];
  stages.forEach(([id, state]) => {
    $(id).dataset.state = state;
    if (state === "current") $(id).setAttribute("aria-current", "step");
    else $(id).removeAttribute("aria-current");
  });
}
function invalidateMesh() {
  clearVisualReview();
  clearDiagnosis("mesh");
  planRevision += 1;
  meshValidation = null;
  planSolve = null;
  $("mesh-image").hidden = true;
  $("mesh-quantities").replaceChildren();
  $("mesh-checks").replaceChildren();
  $("mesh-warnings").replaceChildren();
  $("mesh-scope").hidden = true;
  $("mesh-download").hidden = true;
  $("mesh-log").hidden = true;
  $("mesh-panel").hidden = !meshBusy;
  if (meshBusy) $("mesh-message").textContent = "The model changed while this mesh check was running. Check the updated model again when it finishes.";
}
function requestFingerprint() {
  return JSON.stringify({
    mode,
    prompt: $("prompt").value,
    draft: mode === "plan" ? $("plan-json").value : coreSpec(false),
    action: mode === "plan" ? $("request-action").value : "modify",
    paper: typeof paperFingerprint === "function" ? paperFingerprint() : null,
    aiModel: typeof selectedAiModel === "function" ? selectedAiModel() : null,
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
function clearRecoveryOffer() {
  recoveryJob = null; automaticRecovery = null;
  $("recovery-controls").hidden = true;
  $("recovery-status").textContent = "";
}
function invalidateModel(text) {
  clearRecoveryOffer();
  if (typeof clearAssemblyBinding === "function") clearAssemblyBinding();
  if (typeof clearPaperModelBinding === "function") clearPaperModelBinding();
  clearVisualReview();
  previewRequest = Symbol();
  if (mode === "plan") {
    validatedPlan = null;
    validatedEditor = null;
    invalidateMesh();
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
  const decisions = state.decisions || [];
  const rows = decisions.map(decisionText);
  if (state.functionName && !decisions.some(decision => decision.question === "route")) rows.unshift("Chosen function: " + state.functionName);
  $("decision-list").replaceChildren(...rows.map(text => {
    const item = document.createElement("li"); item.textContent = text; return item;
  }));
  $("decision-trace").hidden = !rows.length;
}
function decisionText(decision) {
  const label = decision.question === "route" ? "Chosen function" : decision.question === "arguments_match"
    ? "Reported argument match" : decision.description || decision.question || "Function evidence";
  const metrics = [];
  if (Number.isFinite(decision.probability)) metrics.push("Reported probability " + (decision.probability * 100).toFixed(1) + "%");
  if (Number.isFinite(decision.confidence)) metrics.push("Provider confidence " + (decision.confidence * 100).toFixed(1) + "%");
  return label + ": " + decision.answer + (metrics.length ? " · " + metrics.join(" · ") : "");
}
function clearVisualReview() {
  visualRequest = Symbol();
  visualBusy = false;
  $("visual-review-panel").hidden = true;
  $("visual-review-summary").textContent = "";
  $("visual-review-findings").replaceChildren();
}
function includesReviewMesh() {
  return $("visual-include-mesh").checked && hasCurrentMesh();
}
function visualFingerprint() {
  return JSON.stringify({request: requestFingerprint(), revision: planRevision,
    includeMesh: $("visual-include-mesh").checked, mesh: includesReviewMesh() ? meshValidation.id : null});
}
function clearDiagnosis(kind) {
  failedJobs[kind] = null;
  diagnosisRequests[kind] = Symbol();
  $(kind + "-diagnose").hidden = true;
  $(kind + "-diagnose").disabled = false;
  $(kind + "-diagnosis").hidden = true;
  $(kind + "-diagnosis").textContent = "";
}
function offerDiagnosis(kind, job) {
  clearDiagnosis(kind);
  failedJobs[kind] = job;
  $(kind + "-diagnose").hidden = !(config?.argument_diagnosis_available ?? config?.ai_available);
}
async function diagnoseJob(kind) {
  const job = failedJobs[kind];
  if (!job) return;
  const serial = Symbol(), fingerprint = requestFingerprint();
  diagnosisRequests[kind] = serial;
  const button = $(kind + "-diagnose"), output = $(kind + "-diagnosis");
  button.disabled = true; output.hidden = false; output.textContent = "Diagnosing the failed function's arguments…";
  try {
    const result = await (await api("/api/jobs/" + job.id + "/diagnose", {})).json();
    if (diagnosisRequests[kind] !== serial || failedJobs[kind] !== job) return;
    if (fingerprint !== requestFingerprint()) { output.hidden = true; return; }
    const trace = Array.isArray(result.trace) ? result.trace : result.trace ? [result.trace] : [];
    output.textContent = "Advisory argument diagnosis for " + result.function_name + ": "
      + (result.arguments_match ? "arguments appear to match the function." : "arguments may not match the function.")
      + (trace.length ? " " + trace.map(decisionText).join("; ") + "." : "")
      + " Review the failure log; this diagnosis does not change the job or model.";
  } catch (error) {
    if (diagnosisRequests[kind] === serial && fingerprint === requestFingerprint()) output.textContent = "Argument diagnosis unavailable: " + error.message;
  } finally {
    if (diagnosisRequests[kind] === serial) {
      if (fingerprint !== requestFingerprint()) output.hidden = true;
      button.disabled = false;
    }
  }
}
$("mesh-diagnose").addEventListener("click", () => diagnoseJob("mesh"));
$("solve-diagnose").addEventListener("click", () => diagnoseJob("solve"));
$("visual-include-mesh").addEventListener("change", () => { clearVisualReview(); controls(); });
$("visual-review").addEventListener("click", async () => {
  const serial = Symbol();
  visualRequest = serial;
  try {
    const values = readySpec();
    if (mode !== "plan") return;
    const fingerprint = visualFingerprint(), includeMesh = includesReviewMesh();
    visualBusy = true; controls();
    $("visual-review-panel").hidden = false;
    $("visual-review-summary").textContent = "Reviewing the current geometry" + (includeMesh ? " and actual FEMM mesh" : " only") + "…";
    $("visual-review-source").textContent = includeMesh ? "Inputs: geometry and the checked FEMM mesh." : "Inputs: geometry only.";
    $("visual-review-findings").replaceChildren();
    try {
      const result = await (await api("/api/plan/review", {prompt: $("prompt").value, spec: values,
        ...(includeMesh ? {mesh_job_id: meshValidation.id} : {})})).json();
      if (visualRequest !== serial || fingerprint !== visualFingerprint()) return;
      const labels = {matches: "Appears consistent", issues: "Possible discrepancies", uncertain: "Uncertain"};
      $("visual-review-summary").textContent = (labels[result.status] || "Visual observations") + ": " + result.summary;
      $("visual-review-findings").replaceChildren(...(result.findings || []).map(finding => {
        const item = document.createElement("li");
        item.textContent = finding.location + ": " + finding.issue + " Suggestion: " + finding.suggestion;
        return item;
      }));
    } catch (error) {
      if (visualRequest === serial && fingerprint === visualFingerprint()) $("visual-review-summary").textContent = "Visual review unavailable: " + error.message;
    }
  } catch (error) { message(error.message, true); }
  finally { if (visualRequest === serial) { visualBusy = false; controls(); } }
});
function resetQuestions() {
  questions[mode] = {items: [], explanation: "", resolved: false, hasSpec: true, decisions: []};
  renderQuestions();
}
function summarizePlan(plan) {
  const shapes = plan.regions || [];
  const circuits = plan.circuits || [];
  const parts = [
    plan.title || "Magnetic model",
    shapes.length + (shapes.length === 1 ? " region" : " regions"),
    circuits.length + (circuits.length === 1 ? " circuit" : " circuits"),
    plan.physics === "axisymmetric_dc_magnetics" ? "axisymmetric · r/z coordinates" : "depth " + plan.depth_mm + " mm",
    "mesh " + plan.mesh_mm + " mm",
  ];
  $("plan-summary").textContent = parts.join(" · ");
  $("model-tag").textContent = "CUSTOM MODEL";
  $("dimensions").textContent = shapes.length + " regions";
}
function setPlan(value, {fresh = false, patch = null} = {}) {
  if (fresh) recordedRevision = null;
  pendingEdits = patch;
  $("plan-json").value = value ? JSON.stringify(value, null, 2) : "";
  invalidateModel(value ? "Model received. Checking geometry and assignments…" : "Add the missing details to create a model.");
  $("plan-summary").textContent = value ? "Checking the proposed model…" : "No model has been created yet.";
}
async function recordCurrentRevision(values, serial, requestedMode, snapshot) {
  if (!config.revision_history_available) return true;
  const parent = recordedRevision;
  const patch = pendingEdits;
  const body = {spec: values, evidence: [{source: "user", reference: "Model editor", detail: $("prompt").value.slice(0, 3000)}]};
  if (typeof paperSourceEvidence === "function") body.evidence.push(...paperSourceEvidence(values));
  if (typeof assemblySourceEvidence === "function") body.evidence.push(...assemblySourceEvidence(values));
  if (typeof paperRevisionInput === "function") Object.assign(body, paperRevisionInput(values));
  if (parent) Object.assign(body, {parent_id: parent.id, branch: parent.branch});
  if (patch?.edits?.length && parent) Object.assign(body, {edits: patch.edits,
    allowed_paths: patch.allowed_paths, reason: patch.explanation || "Requested model modification"});
  const revision = await (await api("/api/revisions", body)).json();
  if (previewRequest !== serial || requestedMode !== mode || snapshot !== $("plan-json").value || recordedRevision !== parent) return false;
  recordedRevision = revision; pendingEdits = null;
  if (typeof paperModelRecorded === "function") paperModelRecorded(revision, values);
  $("model-history").hidden = false;
  $("revision-summary").textContent = "Revision " + revision.id.slice(0, 12) + " · " + revision.branch;
  $("revision-changes").replaceChildren(...revision.changes.map(change => {
    const row = document.createElement("li");
    row.textContent = change.path === "/" ? "Created this model." : change.path + ": " + JSON.stringify(change.before) + " → " + JSON.stringify(change.after);
    return row;
  }));
  await refreshRevisionHistory();
  return previewRequest === serial && requestedMode === mode && snapshot === $("plan-json").value;
}
async function refreshRevisionHistory() {
  const id = recordedRevision?.id;
  if (!id) return;
  const revision = await (await api("/api/revisions/" + id)).json();
  if (recordedRevision?.id !== id) return;
  $("revision-checks").replaceChildren(...(revision.checks || []).map(check => {
    const row = document.createElement("li");
    row.textContent = check.stage + ": " + check.status + " · job " + check.job_id.slice(0, 12);
    return row;
  }));
}
function controls() {
  const ready = Boolean(config) && isReady() && !hasUnresolvedQuestions();
  const working = busy || meshBusy || (typeof paperIsBusy === "function" && paperIsBusy());
  const needsCurrent = mode === "plan" && $("request-action").value === "modify";
  $("solve").disabled = working || !ready || !config?.solver_detected || (mode === "plan" && !hasCurrentMesh());
  $("source").disabled = working || !ready;
  $("mesh-check").disabled = working || mode !== "plan" || !ready || !config?.solver_detected;
  $("mesh-check").textContent = meshBusy ? "Checking mesh…" : hasCurrentMesh() ? "Check mesh again →" : "Check mesh with FEMM →";
  $("interpret").disabled = working || !config?.ai_available || $("prompt").value.trim().length < 5 || (needsCurrent && !isReady());
  $("validate-plan").disabled = working || !$("plan-json").value.trim();
  $("load-example").disabled = working || !config?.plan_example;
  $("load-axisymmetric").disabled = working || !config?.axisymmetric_example;
  $("visual-review-controls").hidden = mode !== "plan" || !config?.visual_review_available;
  $("visual-review").disabled = working || visualBusy || !ready || !$("prompt").value.trim();
  $("visual-review").textContent = visualBusy ? "Reviewing geometry…" : "Review geometry visually";
  $("visual-include-mesh").disabled = !hasCurrentMesh();
  $("visual-mesh-help").textContent = !hasCurrentMesh() ? "Geometry only. Check the current mesh to include it."
    : includesReviewMesh() ? "The current checked mesh will be included." : "Geometry only. The checked mesh will be left out.";
  $("mode").disabled = working;
  $("request-action").disabled = working;
  $("interpret").textContent = mode === "ccore" ? "Apply request ↗" : needsCurrent ? "Update model ↗" : "Create model ↗";
  if (typeof updatePaperControls === "function") updatePaperControls();
  if (typeof updateHierarchyControls === "function") updateHierarchyControls();
  renderPipeline();
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
      if (!await recordCurrentRevision(values, serial, requestedMode, snapshot)) return false;
      if (previewRequest !== serial || requestedMode !== mode || snapshot !== $("plan-json").value) return false;
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
    message(mode === "plan" ? "Geometry, materials and boundary assignments validated. Check the mesh next." : "Geometry and assignments validated. Review the model, then solve.");
    controls();
    return true;
  } catch (error) {
    if (previewRequest === serial && requestedMode === mode) {
      if (mode === "plan") { validatedPlan = null; validatedEditor = null; invalidateMesh(); $("plan-summary").textContent = "The model needs attention before it can be solved."; }
      else validatedCore = null;
      emptyPreview("The current model could not be validated.");
      message(error.message, true);
      controls();
    }
    return false;
  }
}
async function showMode() {
  clearRecoveryOffer();
  clearVisualReview();
  clearTimeout(previewTimer);
  previewRequest = Symbol();
  $("mode").value = mode;
  sessionStorage.setItem("femm_mode", mode);
  const general = mode === "plan";
  $("plan-controls").hidden = !general;
  $("plan-request-controls").hidden = !general;
  $("ccore-controls").hidden = general;
  $("pipeline").hidden = !general;
  $("mesh-controls").hidden = !general;
  $("mesh-panel").hidden = !general || (!meshBusy && !hasCurrentMesh());
  $("inspect-title").textContent = general ? "Inspect, mesh & solve" : "Inspect & solve";
  $("prompt").placeholder = general ? generalPlaceholder : corePlaceholder;
  $("mode-note").textContent = general
    ? "Create new geometry from closed circles, rectangles and polygons. Assign linear materials, circuits and a surrounding air region."
    : "Use the original parameterized actuator to explore force, energy and flux linkage.";
  $("canvas-note").textContent = general
    ? "Geometry, materials and boundary assignments are checked before meshing."
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
$("request-action").addEventListener("change", () => { clearVisualReview(); controls(); });
$("prompt").addEventListener("input", () => { clearVisualReview(); controls(); });
$("resolved").addEventListener("change", () => { questions[mode].resolved = $("resolved").checked; controls(); });
$("plan-json").addEventListener("input", () => {
  if (mode !== "plan") return;
  pendingEdits = null;
  invalidateModel("The model plan changed. Choose Validate & preview to check it.");
  $("plan-summary").textContent = "Unvalidated draft — changes have not been applied to the preview.";
  message("Model plan changed. Validate it before solving or downloading a script.");
});
$("validate-plan").addEventListener("click", preview);
$("load-example").addEventListener("click", async () => {
  resetQuestions();
  setPlan(config.plan_example, {fresh: true});
  $("request-action").value = "modify";
  await preview();
});
$("load-axisymmetric").addEventListener("click", async () => {
  resetQuestions(); setPlan(config.axisymmetric_example, {fresh: true});
  $("request-action").value = "modify"; await preview();
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
    const response = await api(mode === "plan" ? "/api/plan/interpret" : "/api/interpret", {prompt, current,
      ...(mode === "plan" ? {hierarchical: true, model: typeof selectedAiModel === "function" ? selectedAiModel() : null} : {}),
      ...(mode === "plan" && typeof paperPlanningInput === "function" ? paperPlanningInput() : {})});
    const proposal = await response.json();
    if (requestedMode !== mode || fingerprint !== requestFingerprint()) {
      message("Your request or model changed while the AI was working. Its response was not applied; submit your current request again.");
      return;
    }
    if (mode === "plan") {
      invalidateMesh();
      if (proposal.spec || !modifyingPlan) {
        setPlan(proposal.spec, {fresh: !modifyingPlan, patch: modifyingPlan ? proposal : null});
        if (proposal.spec && typeof applyPaperProposal === "function") applyPaperProposal(proposal);
      }
    } else {
      fields.forEach(([key]) => { $(key).value = proposal.spec[key]; });
      invalidateModel();
    }
    if (mode === "plan" && typeof adoptAssemblyProposal === "function") await adoptAssemblyProposal(proposal);
    questions[mode] = {
      items: Array.isArray(proposal.questions) ? proposal.questions : [],
      explanation: proposal.explanation || "",
      resolved: false,
      hasSpec: Boolean(proposal.spec),
      decisions: Array.isArray(proposal.decision_trace) ? proposal.decision_trace : [],
      functionName: proposal.function_name,
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
function renderMeshReport(report) {
  const metrics = [
    ["Nodes", report.node_count], ["Elements", report.element_count], ["Edges", report.edge_count],
    ["Smallest angle", Number.isFinite(report.quality?.min_angle_deg) ? report.quality.min_angle_deg.toFixed(2) + "°" : "Not reported"],
    ["Largest aspect ratio", Number.isFinite(report.quality?.max_aspect_ratio) ? report.quality.max_aspect_ratio.toFixed(2) : "Not reported"],
  ];
  $("mesh-quantities").replaceChildren(...metrics.flatMap(([name, value]) => {
    const label = document.createElement("dt"), quantity = document.createElement("dd");
    label.textContent = name; quantity.textContent = value ?? "Not reported";
    return [label, quantity];
  }));
  $("mesh-checks").replaceChildren(...(report.checks || []).map(check => {
    const item = document.createElement("li");
    item.textContent = (check.name || "Mesh check").replaceAll("_", " ") + ": " + (check.detail || check.status);
    return item;
  }));
  $("mesh-warnings").replaceChildren(...(report.warnings || []).map(warning => {
    const item = document.createElement("li"); item.textContent = warning; return item;
  }));
}
$("mesh-check").addEventListener("click", async () => {
  try {
    if (mode !== "plan") return;
    const values = readySpec(), key = JSON.stringify(values);
    invalidateMesh(); const revision = planRevision; meshBusy = true; controls();
    $("mesh-panel").hidden = false; $("mesh-status").textContent = "SUBMITTING";
    $("mesh-message").textContent = "Submitting the validated model for a mesh check…";
    const job = await (await api("/api/plan/mesh", values, recordedRevision ? {"X-Model-Revision": recordedRevision.id} : {})).json();
    meshJob = {id: job.id, key, revision};
    message("Mesh check submitted. No field solution is being calculated at this stage.");
    await pollMesh();
  } catch (error) {
    meshBusy = false; controls();
    $("mesh-status").textContent = "FAILED"; $("mesh-message").textContent = error.message;
    message(error.message, true);
  }
});
async function pollMesh() {
  clearTimeout(meshPolling);
  const submitted = meshJob;
  if (!submitted) return;
  try {
    const job = await (await api("/api/jobs/" + submitted.id)).json();
    if (meshJob !== submitted) return;
    if (job.kind !== "mesh") throw new Error("The server returned a different job type. Run the mesh check again.");
    $("mesh-status").textContent = job.status.toUpperCase();
    const current = isCurrentMeshSubmission(submitted);
    if (["queued", "running"].includes(job.status)) {
      $("mesh-message").textContent = !current ? "The model changed. This check belongs to the earlier model and cannot enable Solve."
        : job.status === "queued" ? "Queued — mesh checks and solves share one worker."
        : "FEMM is generating the mesh and checking its triangles. No field solve is running.";
      meshPolling = setTimeout(pollMesh, 2000); return;
    }
    $("mesh-log").hidden = false;
    await refreshRevisionHistory();
    if (!current) {
      $("mesh-message").textContent = "This check belongs to an earlier model. Validate and check the current mesh before solving.";
      meshBusy = false; controls(); return;
    }
    if (job.status === "failed") { offerDiagnosis("mesh", {id: submitted.id}); offerRecovery(job); automaticRecovery = null; }
    if (job.status !== "completed") throw new Error(job.error || "Mesh generation failed. Review the model or use a coarser mesh and try again.");
    if (job.result?.status !== "passed") throw new Error("The mesh did not pass its checks. Review the mesh log, adjust the model and check it again.");
    const blob = await (await api("/api/jobs/" + submitted.id + "/mesh.svg")).blob();
    if (meshJob !== submitted) return;
    if (!isCurrentMeshSubmission(submitted)) {
      $("mesh-message").textContent = "The model changed while the mesh was loading. Check the updated model again.";
      meshBusy = false; controls(); return;
    }
    clearVisualReview();
    imageBlob("mesh-image", blob); renderMeshReport(job.result);
    meshValidation = {id: submitted.id, key: submitted.key, revision: submitted.revision};
    $("mesh-status").textContent = "PASSED";
    $("mesh-message").textContent = "Mesh checks passed for the current model. Inspect the mesh, then solve.";
    $("mesh-scope").hidden = false; $("mesh-download").hidden = false;
    message("Mesh checks passed. The current model is ready to solve.");
    meshBusy = false; controls();
    if (automaticRecovery?.mesh_id === submitted.id && hasCurrentMesh()) {
      const recovery = automaticRecovery; automaticRecovery = null;
      await submitSolve(recovery.target_contract_id || null);
    }
  } catch (error) {
    if (meshJob !== submitted) return;
    meshValidation = null; meshBusy = false; controls();
    $("mesh-status").textContent = "FAILED"; $("mesh-message").textContent = error.message;
    message(error.message, true);
  }
}
$("mesh-download").addEventListener("click", async () => {
  try { if (meshJob) await saveResponse(await api("/api/jobs/" + meshJob.id + "/download.zip"), "femm-mesh-" + meshJob.id + ".zip"); }
  catch (error) { message(error.message, true); }
});
$("mesh-log").addEventListener("click", async () => {
  try { if (meshJob) await saveResponse(await api("/api/jobs/" + meshJob.id + "/solver.log"), "mesh-solver.log"); }
  catch (error) { message(error.message, true); }
});
async function submitSolve(targetOverride = undefined) {
  busy = true; controls();
  try {
    const values = readySpec();
    const general = mode === "plan";
    const fingerprint = requestFingerprint(), revisionId = recordedRevision?.id, meshId = meshValidation?.id;
    clearDiagnosis("solve");
    if (general && !hasCurrentMesh()) throw new Error("Run a successful mesh check for this exact model before solving.");
    const targetContract = targetOverride !== undefined ? targetOverride :
      general && typeof freezePaperTargets === "function" ? await freezePaperTargets() : null;
    if (fingerprint !== requestFingerprint() || (general && (recordedRevision?.id !== revisionId || meshValidation?.id !== meshId || !hasCurrentMesh()))) {
      throw new Error("The model, paper case or targets changed while preparing the solve. Submit the current model again.");
    }
    const job = await (await api(general ? "/api/plan/jobs" : "/api/jobs", general ? {spec: values, mesh_job_id: meshValidation.id,
      ...(recordedRevision ? {revision_id: recordedRevision.id} : {}), ...(targetContract ? {target_contract_id: targetContract} : {})} : values)).json();
    planSolve = general ? {key: JSON.stringify(values), status: job.status, id: job.id} : null;
    currentJob = job.id; sessionStorage.setItem("femm_job", currentJob);
    $("results").hidden = false; $("result-body").hidden = true; $("log").hidden = true;
    $("results").scrollIntoView({behavior: "smooth", block: "start"});
    message("Simulation submitted. Results will use this submitted model, even if you edit the draft.");
    await poll();
  } catch (error) { message(error.message, true); busy = false; controls(); }
}
$("solve").addEventListener("click", () => submitSolve());
function offerRecovery(job) {
  if (!config?.controlled_recovery_available || !job.evidence?.failure_id || !job.revision_id ||
      mode !== "plan" || !isReady() || recordedRevision?.id !== job.revision_id || JSON.stringify(job.spec) !== planKey()) return;
  recoveryJob = job; $("recovery-controls").hidden = false;
  $("recovery-status").textContent = "Failure recorded for revision " + job.revision_id.slice(0, 12) + ". Select a repair scope.";
}
$("recover-model").addEventListener("click", async () => {
  if (!recoveryJob || busy || meshBusy) return;
  if (mode !== "plan" || !isReady() || recordedRevision?.id !== recoveryJob.revision_id || JSON.stringify(recoveryJob.spec) !== planKey()) {
    clearRecoveryOffer(); message("This failure belongs to another model revision. Open its model before requesting recovery.", true); return;
  }
  const failed = recoveryJob, fingerprint = requestFingerprint();
  busy = true; controls(); $("recover-model").disabled = true;
  $("recovery-status").textContent = "Tracing the failed call and preparing one bounded repair…";
  try {
    const scope = $("recovery-scope").value;
    const body = {scope, instruction: $("recovery-instruction").value,
      ...(scope === "custom" ? {allowed_paths: $("recovery-paths").value.split(",").map(x => x.trim()).filter(Boolean)} : {})};
    const response = await (await api("/api/jobs/" + failed.id + "/recover", body)).json();
    const frozenRecord = response.target_contract_id ? await (await api("/api/targets/" + response.target_contract_id)).json() : null;
    if (frozenRecord && frozenRecord.revision_id !== response.revision?.id) throw new Error("The recovery targets belong to another model revision.");
    if (fingerprint !== requestFingerprint()) throw new Error("The current request changed. The recovery branch is retained; its model was not applied to this draft.");
    if (!response.revision) {
      $("recovery-status").textContent = [response.proposal.explanation, ...(response.proposal.questions || [])].join(" ");
      return;
    }
    setPlan(response.revision.plan, {fresh: true}); recordedRevision = response.revision;
    if (typeof adoptPaperRevision === "function") adoptPaperRevision(response.revision, frozenRecord);
    resetQuestions();
    if (!await preview()) throw new Error("The recovery branch could not be previewed. Its native job and evidence remain available.");
    const submitted = response.mesh_job;
    automaticRecovery = {mesh_id: submitted.id, target_contract_id: response.target_contract_id};
    meshJob = {id: submitted.id, key: planKey(), revision: planRevision};
    meshBusy = true; $("mesh-panel").hidden = false; $("recovery-status").textContent = response.proposal.explanation + " Checking the repaired mesh…";
    busy = false; controls(); await pollMesh();
  } catch (error) { $("recovery-status").textContent = error.message; message(error.message, true); }
  finally { busy = false; controls(); $("recover-model").disabled = false; }
});
function quantityLabel(key, general) {
  const labels = {force_x: general ? "Region force, x" : "Armature force, x", force_y: general ? "Region force, y" : "Armature force, y", energy: "Magnetic energy", flux_linkage: "Flux linkage", current: "Coil current", runtime: "Solve & extraction time"};
  if (labels[key]) return labels[key];
  if (key.startsWith("current_")) return "Current · " + key.slice(8);
  if (key.startsWith("flux_linkage_")) return "Flux linkage · " + key.slice(13);
  if (key.startsWith("bx_")) return "Probe " + (Number(key.slice(3)) + 1) + " · Bx";
  if (key.startsWith("by_")) return "Probe " + (Number(key.slice(3)) + 1) + " · By";
  if (key.startsWith("br_")) return "Probe " + (Number(key.slice(3)) + 1) + " · Br";
  if (key.startsWith("bz_")) return "Probe " + (Number(key.slice(3)) + 1) + " · Bz";
  if (key === "force_z") return "Region force, axial z";
  return key.replaceAll("_", " ");
}
async function poll() {
  clearTimeout(polling);
  const jobId = currentJob;
  try {
    const job = await (await api("/api/jobs/" + jobId)).json();
    if (jobId !== currentJob) return;
    const general = job.spec?.kind === "magnetostatic_plan";
    if (planSolve?.id === jobId) planSolve.status = job.status;
    $("job-model").textContent = "Submitted model: " + (general ? job.spec.title : "C-core actuator");
    $("job-status").textContent = job.status.toUpperCase();
    $("job-message").textContent = job.status === "queued" ? "Queued — one model is solved at a time to keep hosting costs low."
      : job.status === "running" ? general ? "FEMM is solving the checked model and extracting the field." : "FEMM is meshing, solving and extracting the field."
      : job.status === "failed" ? job.error : "Finite-element solution completed.";
    if (["queued", "running"].includes(job.status)) {
      busy = true; controls(); polling = setTimeout(poll, 2000); return;
    }
    busy = false; controls(); $("log").hidden = false;
    await refreshRevisionHistory();
    if (job.status === "failed") { offerDiagnosis("solve", {id: jobId}); offerRecovery(job); }
    if (job.status === "completed") {
      if (typeof renderTargetResults === "function") renderTargetResults(job);
      if (job.evidence?.target_report?.status !== "passed") offerRecovery(job);
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
    $("load-axisymmetric").hidden = !config.axisymmetric_example;
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
