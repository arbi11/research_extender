"use strict";

let activeAssembly = null, assemblyDraftBase = null, assemblyUiBusy = false;
let sourceInventory = null, sourceAssetBusy = false, sourceReviewBusy = false, sourceContextEpoch = 0;
let shownSourceAsset = null, sourceDrag = null;
const selectedSourceAssets = new Set(), hierarchyImageUrls = new Map();

function selectedAiModel() { return $("ai-model").value.trim() || null; }
function stableJson(value) {
  if (Array.isArray(value)) return "[" + value.map(stableJson).join(",") + "]";
  if (value && typeof value === "object") return "{" + Object.keys(value).sort().map(key => JSON.stringify(key) + ":" + stableJson(value[key])).join(",") + "}";
  return JSON.stringify(value);
}
function hierarchyText(tag, text, className = "") {
  const node = document.createElement(tag); node.textContent = text;
  if (className) node.className = className;
  return node;
}
function hierarchyStatus(id, text, error = false) {
  $(id).textContent = text; $(id).classList.toggle("error", error);
}
function hierarchyImage(element, blob) {
  const previous = hierarchyImageUrls.get(element.id);
  if (previous) URL.revokeObjectURL(previous);
  const url = URL.createObjectURL(blob); hierarchyImageUrls.set(element.id, url);
  element.src = url; element.hidden = false;
}
function clearSourceReview(text = "") {
  $("source-review-status").textContent = text;
  $("source-review-findings").replaceChildren();
  $("source-review-images").replaceChildren();
  for (const [id, url] of hierarchyImageUrls) if (id.startsWith("source-review-image-")) {
    URL.revokeObjectURL(url); hierarchyImageUrls.delete(id);
  }
}
function clearAssemblyBinding() {
  activeAssembly = null;
  if (assemblyDraftBase && !assemblyUiBusy) hierarchyStatus("assembly-status",
    "This hierarchy is retained for inspection. The current model draft is no longer bound to it.");
  clearSourceReview();
}
function assemblySourceEvidence(values) {
  const record = activeAssembly;
  if (!record?.assembly_id || !record.compilation?.plan || stableJson(record.compilation.plan) !== stableJson(values)) return [];
  return [{source: "artifact", reference: "assembly-planning:" + record.assembly_id, sha256: record.assembly_id,
    detail: "Source-linked component hierarchy and feature dependencies for this exact native model."}];
}
function renderMachineInventory(inventory) {
  const panel = $("assembly-inventory"); panel.replaceChildren();
  if (!inventory) return;
  panel.append(hierarchyText("p", inventory.title + " · " + inventory.family + " · " + inventory.physics, "inventory-title"));
  if (inventory.explanation) panel.append(hierarchyText("p", inventory.explanation, "muted"));
  if (inventory.cases?.length) {
    const list = document.createElement("ul"); list.className = "inventory-cases";
    for (const item of inventory.cases) list.append(hierarchyText("li", item.label + " (" + item.id + ")" +
      (inventory.selected_case_id === item.id ? " · selected" : "") +
      (item.missing_details?.length ? " · missing: " + item.missing_details.join("; ") : "")));
    panel.append(list);
  }
}
function renderComponentTree(assembly, compilation = null) {
  const tree = $("assembly-tree"); tree.replaceChildren();
  const components = assembly?.components || [];
  function branch(component) {
    const row = document.createElement("li"), details = document.createElement("details"); details.open = true;
    const summary = document.createElement("summary");
    summary.append(document.createTextNode(component.name + " "));
    summary.append(hierarchyText("span", component.mode.replaceAll("_", " "), "component-kind"));
    details.append(summary, hierarchyText("p", component.role + " · " + component.family + " · " + component.id, "muted"));
    const features = compilation?.expanded_features?.filter(item => item.component_id === component.id);
    const list = document.createElement("ul"); list.className = "component-features";
    if (features?.length) for (const feature of features) {
      list.append(hierarchyText("li", feature.feature_id + " · " + feature.shape.type +
        (feature.circuit_id ? " · " + feature.turns + " turns · " + feature.circuit_id : "") +
        (feature.fact_ids?.length ? " · facts: " + feature.fact_ids.join(", ") : "")));
    } else for (const feature of component.required_features || component.features || []) {
      list.append(hierarchyText("li", typeof feature === "string" ? feature : feature.id + " · " + feature.shape.type));
    }
    if (component.template) list.append(hierarchyText("li", "Template: " + component.template.kind));
    for (const missing of component.missing_details || []) list.append(hierarchyText("li", "Missing: " + missing));
    if (list.children.length) details.append(list);
    const children = components.filter(item => item.parent_id === component.id);
    if (children.length) { const nested = document.createElement("ul"); children.forEach(child => nested.append(branch(child))); details.append(nested); }
    row.append(details); return row;
  }
  components.filter(item => !item.parent_id).forEach(component => tree.append(branch(component)));
}
async function adoptAssemblyProposal(proposal) {
  const record = proposal?.assembly_record;
  if (proposal?.inventory) renderMachineInventory(proposal.inventory);
  if (!record) {
    clearAssemblyBinding();
    if (proposal?.inventory) renderComponentTree(proposal.inventory);
    updateHierarchyControls(); return false;
  }
  activeAssembly = record; assemblyDraftBase = record;
  $("assembly-json").value = JSON.stringify(record.assembly, null, 2);
  $("assembly-panel").open = true;
  renderMachineInventory(record.inventory || proposal.inventory);
  renderComponentTree(record.assembly, record.compilation);
  $("assembly-issues").replaceChildren(...(record.compilation.issues || []).map(issue =>
    hierarchyText("li", issue.code + " · " + issue.path + ": " + issue.message)));
  const supported = Boolean(record.compilation.execution_supported && record.compilation.plan);
  hierarchyStatus("assembly-status", record.assembly.family + " · " + record.assembly.components.length + " components · " +
    record.compilation.expanded_features.length + " expanded features · " + (supported
      ? "Compiled for native FEMM checks. Mesh, solver and target checks remain separate."
      : "Preview available. Native execution is unavailable for the issues listed below."));
  try {
    const blob = await (await api("/api/assemblies/" + record.assembly_id + "/preview.svg")).blob();
    if (activeAssembly?.assembly_id === record.assembly_id) hierarchyImage($("assembly-preview"), blob);
  } catch (error) { if (activeAssembly?.assembly_id === record.assembly_id) hierarchyStatus("assembly-status", error.message, true); }
  updateHierarchyControls(); return true;
}
function currentPaperCase() { return selectedPaper ? $("paper-case").value : "baseline"; }
function sourcePageInfo() { return sourceInventory?.candidates.find(item => String(item.page) === $("source-page").value); }
function fullPageAsset(record) {
  return record && record.bbox.x0 === 0 && record.bbox.y0 === 0 &&
    record.bbox.x1 === record.page_width_pt && record.bbox.y1 === record.page_height_pt;
}
function showCropOverlay() {
  const overlay = $("source-crop-overlay"), page = sourcePageInfo();
  overlay.hidden = $("source-full-page").checked || !page || !fullPageAsset(shownSourceAsset) || shownSourceAsset.page !== page.page;
  if (overlay.hidden) return;
  const [x0, y0, x1, y1] = ["x0", "y0", "x1", "y1"].map(key => Number($("source-crop-" + key).value));
  if (x1 <= x0 || y1 <= y0) { overlay.hidden = true; return; }
  overlay.style.left = (100 * x0 / page.width_pt) + "%"; overlay.style.top = (100 * y0 / page.height_pt) + "%";
  overlay.style.width = (100 * (x1 - x0) / page.width_pt) + "%"; overlay.style.height = (100 * (y1 - y0) / page.height_pt) + "%";
}
function updateSourcePage() {
  const page = sourcePageInfo();
  if (!page) return;
  $("source-page-info").textContent = page.width_pt.toFixed(1) + " × " + page.height_pt.toFixed(1) + " PDF points" +
    (page.rotation ? " · rotated " + page.rotation + "°" : "") +
    (page.caption_hints.length ? " · text hints: " + page.caption_hints.join("; ") : " · no caption hints in extracted text");
  for (const [key, value] of Object.entries({x0: 0, y0: 0, x1: page.width_pt, y1: page.height_pt})) $("source-crop-" + key).value = value;
  shownSourceAsset = null; $("source-page-frame").hidden = true; showCropOverlay(); updateHierarchyControls();
}
function renderSourceAssets() {
  const list = $("source-asset-list"); list.replaceChildren();
  for (const asset of sourceInventory?.assets || []) {
    if (asset.case_id !== currentPaperCase()) continue;
    const row = document.createElement("div"); row.className = "source-asset-row";
    const label = document.createElement("label"), checkbox = document.createElement("input");
    checkbox.type = "checkbox"; checkbox.checked = selectedSourceAssets.has(asset.asset_id); checkbox.dataset.assetId = asset.asset_id;
    checkbox.addEventListener("change", () => {
      if (checkbox.checked && selectedSourceAssets.size >= 4) { checkbox.checked = false; hierarchyStatus("source-asset-status", "Select at most four source images for one comparison.", true); return; }
      if (checkbox.checked) selectedSourceAssets.add(asset.asset_id); else selectedSourceAssets.delete(asset.asset_id);
      clearSourceReview(); updateHierarchyControls();
    });
    const title = (asset.label || asset.purpose) + " · PDF page " + asset.page;
    label.append(checkbox, document.createTextNode(title));
    const view = document.createElement("button"); view.type = "button"; view.textContent = "View";
    view.addEventListener("click", () => showSourceAsset(asset).catch(error => hierarchyStatus("source-asset-status", error.message, true)));
    row.append(label, view, hierarchyText("small", asset.purpose + " · " + asset.case_id + " · crop (" +
      [asset.bbox.x0, asset.bbox.y0, asset.bbox.x1, asset.bbox.y1].map(value => Number(value).toFixed(1)).join(", ") + ") pt", "muted"));
    list.append(row);
  }
  if (!list.children.length) list.append(hierarchyText("p", "No saved source selections for " + currentPaperCase() + ".", "muted"));
}
async function loadSourceInventory() {
  const epoch = ++sourceContextEpoch, source = uploadedPaper;
  selectedSourceAssets.clear(); sourceInventory = null; shownSourceAsset = null; clearSourceReview();
  $("source-assets").hidden = !source;
  $("source-asset-list").replaceChildren(); $("source-page-frame").hidden = true;
  if (!source) { updateHierarchyControls(); return; }
  hierarchyStatus("source-asset-status", "Reading source-page inventory…");
  try {
    const inventory = await (await api("/api/papers/" + source.paper_id + "/assets")).json();
    if (epoch !== sourceContextEpoch || uploadedPaper !== source) return;
    sourceInventory = inventory;
    $("source-page").replaceChildren(...inventory.candidates.map(page => {
      const option = document.createElement("option"); option.value = page.page;
      option.textContent = "PDF page " + page.page + (page.caption_hints.length ? " · figure/table text hints" : ""); return option;
    }));
    updateSourcePage(); renderSourceAssets();
    hierarchyStatus("source-asset-status", "Select original pages or explicit crops for " + currentPaperCase() + ". Source pixels remain bound to the original PDF hash.");
  } catch (error) { if (epoch === sourceContextEpoch) hierarchyStatus("source-asset-status", error.message, true); }
  finally { updateHierarchyControls(); }
}
async function showSourceAsset(asset) {
  const epoch = sourceContextEpoch;
  const blob = await (await api("/api/paper-assets/" + asset.asset_id + "/image")).blob();
  if (epoch !== sourceContextEpoch) return;
  shownSourceAsset = asset;
  hierarchyImage($("source-page-image"), blob); $("source-page-frame").hidden = false;
  showCropOverlay();
}
async function createSourceAsset(fullPage = false) {
  const page = sourcePageInfo(), source = uploadedPaper;
  if (!page || !source) return;
  const epoch = sourceContextEpoch, usePage = fullPage || $("source-full-page").checked;
  const body = {page: page.page, case_id: currentPaperCase(), purpose: fullPage ? "page" : $("source-purpose").value,
    label: fullPage ? "PDF page " + page.page : $("source-asset-label").value.trim()};
  if (!usePage) {
    body.bbox = Object.fromEntries(["x0", "y0", "x1", "y1"].map(key => [key, Number($("source-crop-" + key).value)]));
    const {x0, y0, x1, y1} = body.bbox;
    if (![x0, y0, x1, y1].every(Number.isFinite) || x0 < 0 || y0 < 0 || x1 <= x0 || y1 <= y0 || x1 > page.width_pt || y1 > page.height_pt)
      throw new Error("Enter crop bounds inside the selected page with positive width and height.");
  }
  sourceAssetBusy = true; updateHierarchyControls(); hierarchyStatus("source-asset-status", "Rendering the original local PDF selection…");
  try {
    const asset = await (await api("/api/papers/" + source.paper_id + "/assets", body)).json();
    if (epoch !== sourceContextEpoch || uploadedPaper !== source) return;
    if (!sourceInventory.assets.some(item => item.asset_id === asset.asset_id)) sourceInventory.assets.push(asset);
    if (!fullPage) {
      if (selectedSourceAssets.size < 4) selectedSourceAssets.add(asset.asset_id);
      else hierarchyStatus("source-asset-status", "Selection saved. Deselect another image to include it in a comparison.");
    }
    renderSourceAssets(); await showSourceAsset(asset);
    hierarchyStatus("source-asset-status", fullPage ? "Original page shown. Drag to select a crop or use the coordinate fields."
      : "Source selection saved for " + asset.case_id + " · PDF page " + asset.page + " · " + asset.png_sha256.slice(0, 12));
  } finally { sourceAssetBusy = false; updateHierarchyControls(); }
}
function sourceComparisonModel(includeMesh = false) {
  if (activeAssembly && (!activeAssembly.compilation.plan ||
      stableJson(activeAssembly.compilation.plan) === stableJson(editorPlan()))) {
    if (!includeMesh) return {assembly_id: activeAssembly.assembly_id};
    return {spec: editorPlan(), feature_map: activeAssembly.feature_map};
  }
  if (mode !== "plan") throw new Error("Source comparison uses a general model or an assembly preview.");
  return {spec: editorPlan()};
}
function updateHierarchyControls() {
  const working = busy || meshBusy || paperBusy;
  $("ai-model").disabled = working || assemblyUiBusy;
  $("assembly-discover").disabled = working || assemblyUiBusy || !config?.ai_available || $("prompt").value.trim().length < 5;
  $("assembly-save").disabled = working || assemblyUiBusy || !$("assembly-json").value.trim();
  $("source-page-load").disabled = working || sourceAssetBusy || !sourcePageInfo();
  $("source-asset-save").disabled = working || sourceAssetBusy || !sourcePageInfo();
  for (const key of ["x0", "y0", "x1", "y1"]) $("source-crop-" + key).disabled = $("source-full-page").checked || sourceAssetBusy;
  const mesh = typeof hasCurrentMesh === "function" && hasCurrentMesh();
  $("source-review-mesh").disabled = !mesh;
  if (!mesh) $("source-review-mesh").checked = false;
  let modelAvailable = false;
  try { sourceComparisonModel(); modelAvailable = true; } catch (_) { /* No current typed preview. */ }
  $("source-review").disabled = working || sourceReviewBusy || !modelAvailable || selectedSourceAssets.size === 0 ||
    !(config?.visual_review_available || config?.source_review_available);
  $("source-review").textContent = sourceReviewBusy ? "Comparing source images…" : "Compare model with source";
}
function manualAssemblyVariation(value) {
  const note = $("assembly-change-note").value.trim(), before = new Map((assemblyDraftBase?.assembly.facts || []).map(fact => [fact.id, fact]));
  let count = 0;
  for (const fact of value.facts || []) {
    const original = before.get(fact.id);
    if (original?.status !== "paper" || (stableJson(fact.value) === stableJson(original.value) && fact.unit === original.unit && fact.status === original.status)) continue;
    if (!note) throw new Error("Describe the manual change before saving a variation of a cited physical fact.");
    fact.status = "user"; fact.source_parameter_id = null; fact.calculation = null;
    fact.source_refs = [...new Set([...(fact.source_refs || []), "user:manual-assembly-edit"])];
    fact.explanation = "User variation: " + note + ". Original cited value: " + JSON.stringify(original.value) + " " + original.unit + ".";
    count += 1;
  }
  return count;
}
$("assembly-save").addEventListener("click", async () => {
  let value;
  try { value = JSON.parse($("assembly-json").value); }
  catch (_) { hierarchyStatus("assembly-status", "The assembly is not valid JSON.", true); return; }
  assemblyUiBusy = true; updateHierarchyControls();
  try {
    const changed = manualAssemblyVariation(value), previous = assemblyDraftBase;
    const body = {assembly: value, instructions: $("assembly-change-note").value.trim(), case_id: previous?.case_id || currentPaperCase(),
      ...(previous ? {parent_id: previous.assembly_id} : {}),
      ...(previous?.evidence_id ? {evidence_id: previous.evidence_id} : {})};
    const record = await (await api("/api/assemblies", body)).json();
    clearPaperModelBinding();
    setPlan(record.compilation.plan, {fresh: true});
    await adoptAssemblyProposal({assembly_record: record});
    if (record.compilation.plan) await preview();
    if (changed) hierarchyStatus("assembly-status", changed + " cited facts saved as explicit user overrides. Published targets were not inherited. " + $("assembly-status").textContent);
    else if (record.compilation.plan) hierarchyStatus("assembly-status", $("assembly-status").textContent + " Manual assembly saves require a fresh paper binding before linked target comparisons.");
  } catch (error) { hierarchyStatus("assembly-status", error.message, true); }
  finally { assemblyUiBusy = false; updateHierarchyControls(); controls(); }
});
$("assembly-discover").addEventListener("click", async () => {
  assemblyUiBusy = true; updateHierarchyControls(); hierarchyStatus("assembly-status", "Discovering the machine, components and operating cases…");
  const prompt = $("prompt").value, epoch = paperEpoch;
  try {
    const model = selectedAiModel(), source = paperPlanningInput();
    const discovery = await (await api("/api/assemblies/discover", {prompt, ...(model ? {model} : {}),
      ...(source.evidence_id ? {evidence_id: source.evidence_id, case_id: source.case_id} : {})})).json();
    if (epoch !== paperEpoch || prompt !== $("prompt").value) return;
    clearAssemblyBinding();
    renderMachineInventory(discovery.inventory); renderComponentTree(discovery.inventory);
    $("assembly-panel").open = true;
    hierarchyStatus("assembly-status", "Component inventory discovered with " + discovery.requested_model + ". Create the model to detail and compile its features.");
  } catch (error) { hierarchyStatus("assembly-status", error.message, true); }
  finally { assemblyUiBusy = false; updateHierarchyControls(); }
});
$("assembly-json").addEventListener("input", () => {
  activeAssembly = null; clearPaperModelBinding(); clearSourceReview();
  invalidateModel("Assembly draft changed. Save and compile it before meshing or solving.");
  hierarchyStatus("assembly-status", "Unsaved assembly draft. Cited physical changes must be recorded as user variations."); updateHierarchyControls();
});
$("source-page-load").addEventListener("click", () => createSourceAsset(true).catch(error => hierarchyStatus("source-asset-status", error.message, true)));
$("source-asset-save").addEventListener("click", () => createSourceAsset(false).catch(error => hierarchyStatus("source-asset-status", error.message, true)));
$("source-page").addEventListener("change", updateSourcePage);
$("source-full-page").addEventListener("change", () => { showCropOverlay(); updateHierarchyControls(); });
for (const key of ["x0", "y0", "x1", "y1"]) $("source-crop-" + key).addEventListener("input", showCropOverlay);
$("prompt").addEventListener("input", updateHierarchyControls);
$("plan-json").addEventListener("input", clearAssemblyBinding);
$("ai-model").addEventListener("input", updateHierarchyControls);
document.addEventListener("femm:paper-context", () => { clearAssemblyBinding(); loadSourceInventory(); });

function cropPointer(event) {
  const page = sourcePageInfo(), bounds = $("source-page-image").getBoundingClientRect();
  return {x: Math.max(0, Math.min(page.width_pt, (event.clientX - bounds.left) / bounds.width * page.width_pt)),
    y: Math.max(0, Math.min(page.height_pt, (event.clientY - bounds.top) / bounds.height * page.height_pt))};
}
$("source-page-image").addEventListener("pointerdown", event => {
  if (!fullPageAsset(shownSourceAsset) || shownSourceAsset.page !== sourcePageInfo()?.page) return;
  sourceDrag = cropPointer(event); event.preventDefault(); event.currentTarget.setPointerCapture(event.pointerId);
});
$("source-page-image").addEventListener("pointermove", event => {
  if (!sourceDrag) return;
  const end = cropPointer(event); $("source-full-page").checked = false;
  for (const [key, value] of Object.entries({x0: Math.min(sourceDrag.x, end.x), y0: Math.min(sourceDrag.y, end.y),
    x1: Math.max(sourceDrag.x, end.x), y1: Math.max(sourceDrag.y, end.y)})) $("source-crop-" + key).value = value.toFixed(2);
  showCropOverlay(); updateHierarchyControls();
});
for (const name of ["pointerup", "pointercancel"]) $("source-page-image").addEventListener(name, () => { sourceDrag = null; });

$("source-review").addEventListener("click", async () => {
  sourceReviewBusy = true; updateHierarchyControls(); clearSourceReview("Comparing the selected original source pixels with this model view…");
  let fingerprint;
  function snapshot() { return stableJson({model: sourceComparisonModel($("source-review-mesh").checked),
    assets: [...selectedSourceAssets].sort(), case_id: currentPaperCase(), prompt: $("prompt").value, ai_model: selectedAiModel(),
    focus: $("source-review-focus").checked, mesh_job_id: $("source-review-mesh").checked ? meshValidation?.id : null}); }
  function stillCurrent() { try { return fingerprint === snapshot(); } catch (_) { return false; } }
  try {
    fingerprint = snapshot();
    const model = selectedAiModel(), body = {...sourceComparisonModel($("source-review-mesh").checked),
      asset_ids: [...selectedSourceAssets], case_id: currentPaperCase(), prompt: $("prompt").value.trim() || "Compare the selected source design with the model view.",
      focused_checks: $("source-review-focus").checked, ...(model ? {model} : {}),
      ...($("source-review-mesh").checked && hasCurrentMesh() ? {mesh_job_id: meshValidation.id} : {})};
    const report = await (await api("/api/source-review", body)).json();
    if (!stillCurrent()) { hierarchyStatus("source-review-status", "The model or source selection changed. Review " + report.review_id + " was retained, but belongs to the earlier view."); return; }
    hierarchyStatus("source-review-status", report.summary + " · View: " + report.view_alignment.replaceAll("_", " ") + " · " + report.requested_model + " · Advisory observations.");
    const focus = new Map(report.focused_checks.map(item => [item.finding_index, item]));
    $("source-review-findings").replaceChildren(...report.findings.map((item, index) => {
      const row = document.createElement("li"), check = focus.get(index);
      row.append(hierarchyText("strong", item.check + " · " + (item.feature_id || "source object missing from model")),
        hierarchyText("p", item.observation), hierarchyText("small", "Uncertainty: " + item.uncertainty + " · " + item.uncertainty_reason),
        hierarchyText("p", item.suggested_inspection, "muted"), hierarchyText("small", "Source assets: " + item.source_asset_ids.map(id => id.slice(0, 12)).join(", "), "muted"));
      if (check) row.append(hierarchyText("p", "Jev focused observation: " + check.answer + " · " + Math.round(check.probability * 100) + "% probability (informational)", "muted"));
      return row;
    }));
    if (report.focused_check_status === "failed") $("source-review-findings").append(hierarchyText("li", "Focused image check unavailable: " + report.focused_check_error + " The initial visual observations are preserved."));
    for (const [index, metadata] of report.images.entries()) {
      const figure = document.createElement("figure"), image = document.createElement("img"); image.id = "source-review-image-" + index;
      image.alt = metadata.role === "source" ? "Original source selection used in this comparison" : metadata.role === "geometry" ? "Exact geometry view used in this comparison" : "Checked native mesh used in this comparison";
      figure.append(image, hierarchyText("figcaption", metadata.role.replaceAll("_", " ") + (metadata.padded_for_multimodal_limits ? " · white border added for image limits" : "")));
      $("source-review-images").append(figure);
      const blob = await (await api("/api/source-reviews/" + report.review_id + "/images/" + metadata.transmitted_png_sha256)).blob();
      if (!stillCurrent()) { clearSourceReview("Model or source changed. Previous comparison retained in its audit record."); return; }
      hierarchyImage(image, blob);
    }
  } catch (error) { hierarchyStatus("source-review-status", error.message, true); }
  finally { sourceReviewBusy = false; updateHierarchyControls(); }
});
updateHierarchyControls();
