"use strict";
let uploadedPaper = null, selectedPaper = null, paperEvidence = null, paperBusy = false;
let paperEpoch = 0, paperModelBinding = null, paperRecoveryContract = null, paperExtractionNotice = "", paperExcitationNotice = "";

function paperIsBusy() { return paperBusy; }
function paperPlanningInput() {
  return selectedPaper ? {evidence_id: selectedPaper.evidence_id, case_id: $("paper-case").value,
    preserve_paper_excitation: $("paper-preserve-excitation").checked} : {};
}
function paperBindingMatches(values) {
  return Boolean(paperModelBinding && selectedPaper && paperModelBinding.evidence_id === selectedPaper.evidence_id &&
    paperModelBinding.case_id === $("paper-case").value &&
    paperModelBinding.preserve_paper_excitation === $("paper-preserve-excitation").checked && paperModelBinding.specKey === JSON.stringify(values));
}
function paperRevisionInput(values) {
  if (!paperBindingMatches(values)) return {};
  const {planning_id, evidence_id, case_id, preserve_paper_excitation} = paperModelBinding;
  return {planning_id, evidence_id, case_id, preserve_paper_excitation};
}
function paperSourceEvidence(values) {
  return paperBindingMatches(values) ? [{source: "paper", reference: "Evidence " + selectedPaper.evidence_id,
    detail: paperEvidence.title + " · case " + $("paper-case").value + " · planning " + paperModelBinding.planning_id,
    sha256: uploadedPaper.document.sha256}] : [];
}
function clearPaperModelBinding() {
  paperModelBinding = null; paperRecoveryContract = null; paperExcitationNotice = "";
  $("target-json").readOnly = false;
  $("compare-targets").checked = false;
  $("compare-targets").disabled = true;
}
function applyPaperProposal(proposal) {
  clearPaperModelBinding();
  if (!selectedPaper || !proposal.planning_id || !proposal.spec) return;
  paperExcitationNotice = typeof proposal.paper_excitation_check?.message === "string" ? proposal.paper_excitation_check.message : "";
  paperModelBinding = {...paperPlanningInput(), planning_id: proposal.planning_id, specKey: JSON.stringify(proposal.spec)};
  $("compare-targets").checked = paperEvidence.targets.some(target => target.case_id === paperModelBinding.case_id);
}
function recoveredContractMatches(values) {
  return Boolean(paperRecoveryContract && recordedRevision?.id === paperRecoveryContract.revision_id &&
    paperRecoveryContract.specKey === JSON.stringify(values) &&
    paperRecoveryContract.preserve_paper_excitation === $("paper-preserve-excitation").checked &&
    (!paperRecoveryContract.evidence_id || (selectedPaper?.evidence_id === paperRecoveryContract.evidence_id &&
      $("paper-case").value === paperRecoveryContract.contract.case_id)));
}
function adoptPaperRevision(revision, frozenRecord = null) {
  clearPaperModelBinding();
  if (frozenRecord) {
    if (frozenRecord.revision_id !== revision.id) throw new Error("The recovery targets belong to another model revision.");
    paperRecoveryContract = {...frozenRecord, specKey: JSON.stringify(revision.plan),
      preserve_paper_excitation: $("paper-preserve-excitation").checked};
    $("target-json").value = JSON.stringify(frozenRecord.contract.targets, null, 2);
    $("target-json").readOnly = true;
    $("compare-targets").checked = true;
    paperState("Recovery retains the original frozen target definitions and tolerances for this exact revision. Further unchanged-model solves use the same contract.");
  }
  if (!selectedPaper || !revision.planning_id || revision.evidence_id !== selectedPaper.evidence_id || revision.case_id !== $("paper-case").value || revision.preserve_paper_excitation !== $("paper-preserve-excitation").checked) return;
  paperModelBinding = {planning_id: revision.planning_id, evidence_id: revision.evidence_id,
    case_id: revision.case_id, preserve_paper_excitation: $("paper-preserve-excitation").checked,
    specKey: JSON.stringify(revision.plan)};
  $("compare-targets").checked = paperEvidence.targets.some(target => target.case_id === revision.case_id);
}
function paperFingerprint() {
  return JSON.stringify({epoch: paperEpoch, input: paperPlanningInput(), compare: $("compare-targets").checked, targets: $("target-json").value});
}
function updatePaperControls() {
  const working = paperBusy || busy || meshBusy;
  $("paper-upload").disabled = working;
  $("paper-extract").disabled = working || !uploadedPaper || !config?.ai_available;
  $("paper-case").disabled = working;
  $("paper-preserve-excitation").disabled = working;
  let bound = false;
  try {
    bound = Boolean(mode === "plan" && recordedRevision && paperBindingMatches(editorPlan()) &&
      recordedRevision.planning_id === paperModelBinding.planning_id &&
      recordedRevision.evidence_id === paperModelBinding.evidence_id && recordedRevision.case_id === paperModelBinding.case_id &&
      recordedRevision.preserve_paper_excitation === paperModelBinding.preserve_paper_excitation);
  } catch (_) { /* An empty or malformed draft has no paper binding. */ }
  let recovered = false;
  try { recovered = mode === "plan" && recoveredContractMatches(editorPlan()); } catch (_) { /* No valid draft. */ }
  $("compare-targets").disabled = recovered || working || !bound || !paperEvidence?.targets.some(target => target.case_id === $("paper-case").value);
  if (recovered) $("compare-targets").checked = true;
  if (typeof updateHierarchyControls === "function") updateHierarchyControls();
}
function paperModelRecorded(revision, values) {
  if (paperBusy || paperRecoveryContract || !paperBindingMatches(values) ||
      revision.planning_id !== paperModelBinding.planning_id || revision.evidence_id !== paperModelBinding.evidence_id ||
      revision.case_id !== paperModelBinding.case_id || revision.preserve_paper_excitation !== paperModelBinding.preserve_paper_excitation) return;
  paperState("Model ready for " + revision.case_id + ". Compare the original conditions before solving." +
    (paperExcitationNotice ? " " + paperExcitationNotice : "") +
    (paperExtractionNotice ? " " + paperExtractionNotice : ""));
}
function paperState(text, error = false) {
  $("paper-status").textContent = text; $("paper-status").classList.toggle("error", error);
  updatePaperControls();
}
function paperContextChanged(text) {
  clearPaperModelBinding();
  invalidateModel(text || "Paper context changed. Create a model for the selected paper case before comparing its targets.");
  emitPaperContext();
}
function emitPaperContext() {
  document.dispatchEvent(new CustomEvent("femm:paper-context", {detail: {
    paper_id: uploadedPaper?.paper_id || null,
    case_id: selectedPaper ? $("paper-case").value : "baseline"
  }}));
}
function renderPaperCase() {
  if (!paperEvidence) return;
  paperContextChanged();
  const caseId = $("paper-case").value;
  const parameters = paperEvidence.parameters.filter(item => item.case_id === "baseline" || item.case_id === caseId);
  $("paper-parameters").replaceChildren(...parameters.map(item => {
    const row = document.createElement("li");
    row.textContent = item.description + ": " + item.value + " " + item.unit + " · " + item.evidence_kind +
      (item.citations.length ? " · PDF page " + item.citations.map(c => c.page).join(", ") : "");
    return row;
  }));
  const targets = paperEvidence.targets.filter(item => item.case_id === caseId);
  $("target-json").value = JSON.stringify(targets, null, 2);
  let conditions = $("paper-target-conditions");
  if (!conditions) {
    conditions = document.createElement("ul"); conditions.id = "paper-target-conditions";
    $("paper-questions").after(conditions);
  }
  conditions.replaceChildren(...targets.map(target => {
    const row = document.createElement("li"); row.textContent = target.description + " · Published conditions: " + target.conditions;
    return row;
  }));
  paperState("Selected case " + caseId + ". Create its model to enable linked target comparison. Explicit instruction changes are variations; agreement alone does not establish reproduction of the original conditions." +
    (paperExtractionNotice ? " " + paperExtractionNotice : ""));
  controls();
}
async function freezePaperTargets() {
  if (paperRecoveryContract) {
    if (!recoveredContractMatches(readySpec())) throw new Error("The frozen recovery targets no longer match this exact model and paper case.");
    return paperRecoveryContract.contract_id;
  }
  if (!selectedPaper || !$("compare-targets").checked) return null;
  const values = readySpec();
  if (!recordedRevision || !paperBindingMatches(values) || recordedRevision.planning_id !== paperModelBinding.planning_id ||
      recordedRevision.evidence_id !== paperModelBinding.evidence_id || recordedRevision.case_id !== paperModelBinding.case_id ||
      recordedRevision.preserve_paper_excitation !== paperModelBinding.preserve_paper_excitation) {
    throw new Error("Create a model for this paper and operating case before freezing linked targets. Previewing an older model cannot attach a new paper case.");
  }
  let targets;
  try { targets = JSON.parse($("target-json").value); }
  catch (_) { throw new Error("The target definitions are not valid JSON."); }
  if (!Array.isArray(targets) || !targets.length) throw new Error("Select at least one target quantity before solving.");
  const record = await (await api("/api/targets", {revision_id: recordedRevision.id,
    case_id: $("paper-case").value, targets, evidence_id: selectedPaper.evidence_id})).json();
  return record.contract_id;
}
function renderTargetResults(job) {
  const report = job.evidence?.target_report;
  $("target-results").hidden = !report;
  if (!report) return;
  $("target-status").textContent = report.status === "passed" ? "Numerical target comparison passed — verify the published operating conditions" :
    report.status === "failed" ? "Target mismatch — the solver completed" : "Target comparison is incomplete";
  $("target-checks").replaceChildren(...report.checks.map(check => {
    const row = document.createElement("li");
    row.textContent = check.target_id + ": " + check.status + " · expected " + check.target + " " + check.unit +
      (check.actual === null ? " · " + check.error : " · computed " + Number(check.actual).toPrecision(6) + " " + check.unit +
       " · permitted error " + Number(check.permitted_error).toPrecision(3) + " " + check.unit);
    return row;
  }));
}
$("paper-upload").addEventListener("click", async () => {
  const file = $("paper-file").files[0];
  if (!file) { paperState("Choose a PDF first.", true); return; }
  if (file.size > 8 * 1024 * 1024) { paperState("The PDF exceeds 8 MiB.", true); return; }
  const epoch = ++paperEpoch;
  uploadedPaper = null; selectedPaper = null; paperEvidence = null; paperExtractionNotice = "";
  $("paper-evidence").hidden = true; paperContextChanged();
  paperBusy = true; controls(); paperState("Reading PDF text locally…");
  try {
    const response = await fetch("/api/papers", {method: "POST", body: file,
      headers: {"Authorization": "Bearer " + token, "Content-Type": "application/pdf", "X-Filename": file.name.replace(/[^\x20-\x7E]/g, "_")}});
    if (!response.ok) { const data = await response.json(); throw new Error(typeof data.detail === "string" ? data.detail : "Could not read this PDF."); }
    const received = await response.json();
    if (epoch !== paperEpoch || $("paper-file").files[0] !== file) return;
    uploadedPaper = received;
    emitPaperContext();
    paperState(uploadedPaper.document.page_count + " pages read locally. " + uploadedPaper.document.missing_information.join(" "));
  } catch (error) { if (epoch === paperEpoch) paperState(error.message, true); }
  finally { if (epoch === paperEpoch) { paperBusy = false; controls(); } }
});
$("paper-extract").addEventListener("click", async () => {
  if (!uploadedPaper || paperBusy) return;
  const epoch = ++paperEpoch, source = uploadedPaper;
  paperExtractionNotice = "";
  paperContextChanged(); paperBusy = true; controls(); paperState("Extracting cited parameters and numerical references…");
  try {
    const chosenModel = typeof selectedAiModel === "function" ? selectedAiModel() : null;
    const received = await (await api("/api/papers/" + source.paper_id + "/extract", {instructions: $("prompt").value,
      ...(chosenModel ? {model: chosenModel} : {})})).json();
    if (epoch !== paperEpoch || uploadedPaper !== source) return;
    selectedPaper = received; paperEvidence = selectedPaper.evidence;
    if (received.extraction_status === "partial") {
      const count = Number.isInteger(received.rejected_claim_count) ? received.rejected_claim_count : (received.validation_summary || []).length;
      paperExtractionNotice = "Partial extraction: " + count + " claims excluded; review questions and supply missing inputs." +
        (received.audit_id ? " Audit: " + received.audit_id : "");
    }
    $("paper-title").textContent = paperEvidence.title;
    const cases = [...new Set([...paperEvidence.targets, ...paperEvidence.parameters].map(item => item.case_id))];
    $("paper-case").replaceChildren(...cases.map(id => { const option = document.createElement("option"); option.value = id; option.textContent = id; return option; }));
    $("paper-questions").replaceChildren(...[...paperEvidence.questions, ...paperEvidence.limitations].map(text => {
      const row = document.createElement("li"); row.textContent = text; return row;
    }));
    $("paper-evidence").hidden = false; renderPaperCase();
    if (!$("prompt").value.trim()) $("prompt").value = "Reproduce the selected paper case. Preserve the cited physical inputs and ask for missing information.";
    $("prompt").dispatchEvent(new Event("input", {bubbles: true}));
  } catch (error) { if (epoch === paperEpoch && uploadedPaper === source) paperState(error.message, true); }
  finally { if (epoch === paperEpoch) { paperBusy = false; controls(); } }
});
$("paper-case").addEventListener("change", renderPaperCase);
$("paper-preserve-excitation").addEventListener("change", () => {
  paperContextChanged("Excitation policy changed. Create the selected case again before comparing its targets.");
  paperState($("paper-preserve-excitation").checked
    ? "Paper excitation selected. Create the model again to apply the cited current or current density."
    : "Excitation variation selected. Your instructions may change the current; published targets still describe their original conditions.");
  controls();
});
function removePaperContext() {
  paperEpoch += 1; paperBusy = false;
  uploadedPaper = null; selectedPaper = null; paperEvidence = null; paperExtractionNotice = "";
  $("paper-preserve-excitation").checked = true;
  $("paper-evidence").hidden = true;
  paperContextChanged("Paper context removed. Validate your model to continue with natural-language or manual input.");
  paperState("Paper context removed. Natural-language modelling remains available."); controls();
}
$("paper-clear").addEventListener("click", removePaperContext);
$("paper-file").addEventListener("change", () => {
  removePaperContext();
  paperState("Read the selected PDF to use its parameters and targets.");
});
