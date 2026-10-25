"use strict";
(() => {
  const $ = id => document.getElementById(id);
  const state = {graph: null, nodes: new Map(), sources: new Map(), examples: [], artifacts: [], recipes: [], visible: new Set(), selected: null, example: null, root: null, geometry: null, recipe: null, parameters: {}, request: 0, statusTimer: null};
  let cy;
  const hierarchyRelations = new Set(["decomposes_into", "implemented_by", "reuses"]);
  const layerNames = {1: "Workflow", 2: "Components & methods", 3: "Exact operations"};

  function status(message, error = false, persistent = false) {
    clearTimeout(state.statusTimer);
    $("status").hidden = false;
    $("status").textContent = message;
    $("status").classList.toggle("error", error);
    if (!persistent) state.statusTimer = setTimeout(() => { $("status").hidden = true; }, error ? 10000 : 5000);
  }
  async function api(path) {
    const response = await fetch(path, {headers: {Accept: "application/json"}});
    if (!response.ok) {
      let message;
      try { const body = await response.json(); message = body.error || body.detail || body.message; } catch (_) { /* Preserve the status below. */ }
      throw new Error(typeof message === "string" ? message : "Request failed (" + response.status + ").");
    }
    return response.json();
  }
  function element(tag, text, className) {
    const value = document.createElement(tag);
    if (text !== undefined && text !== null) value.textContent = text;
    if (className) value.className = className;
    return value;
  }
  function text(value) { return typeof value === "string" ? value : value == null ? "" : JSON.stringify(value); }
  function array(value) { return Array.isArray(value) ? value : value ? [value] : []; }
  function safeUrl(value) {
    try { const url = new URL(value, location.origin); return ["https:", "http:"].includes(url.protocol) ? url.href : null; }
    catch (_) { return null; }
  }
  function verification(value) {
    const item = typeof value === "string" ? {status: value} : value || {status: "documented"};
    const fragment = document.createDocumentFragment();
    fragment.append(element("span", item.status.replaceAll("_", " "), "pill " + item.status));
    if (item.note) fragment.append(element("p", item.note, "muted"));
    return fragment;
  }
  function renderSummary() {
    const sourceCount = state.sources.size, nodes = state.nodes.size, examples = state.examples.length;
    $("summary").replaceChildren();
    [[examples, "worked examples"], [sourceCount, "source documents"], [nodes, "knowledge nodes"], [3, "abstraction layers"]].forEach(([value, label]) => {
      const item = element("span"); item.append(element("strong", value.toLocaleString()), document.createTextNode(" " + label)); $("summary").append(item);
    });
    $("summary").append(element("span", "Source-backed · Local-first · No AI key needed", "summary-suffix"));
  }
  function renderCoverage(data = {}) {
    const sources = [...state.sources.values()];
    const successful = sources.filter(source => ["fetched", "ok", "downloaded", "available", "success"].includes(source.status) || source.sha256).length;
    $("coverage-count").textContent = successful + "/" + sources.length;
    const counts = new Map();
    sources.forEach(source => { const key = source.status || "catalogued"; counts.set(key, (counts.get(key) || 0) + 1); });
    const children = [...counts].map(([name, count]) => { const row = element("div", null, "coverage-item"); row.append(element("span", name.replaceAll("_", " ")), element("strong", count)); return row; });
    children.push(element("p", state.artifacts.filter(a => a.geometry_available).length + " native model previews available. Sources retain retrieval hashes and status."));
    if (data.coverage?.note) children.push(element("p", data.coverage.note));
    $("coverage-details").replaceChildren(...children);
  }
  function renderCases() {
    const query = $("case-search").value.toLowerCase().trim(), family = $("family-filter").value;
    const filtered = state.examples.filter(example => (!family || example.family === family) && (!query || [example.title, example.summary, example.family, ...(example.physics || []), ...(example.tags || []), ...(example.node_ids || []).map(id => state.nodes.get(id)?.title || "")].join(" ").toLowerCase().includes(query)));
    $("case-count").textContent = filtered.length + (filtered.length === 1 ? " example" : " examples");
    $("case-list").replaceChildren(...filtered.map(example => {
      const button = element("button", null, "case-button" + (state.example?.id === example.id ? " active" : ""));
      button.type = "button"; button.dataset.exampleId = example.id;
      button.append(element("strong", example.title), element("small", (example.family || "Worked example").replaceAll("_", " ")));
      if (array(example.reproductions).some(item => item.status === "passed")) button.append(element("span", "reproduced", "pill numerically_checked"));
      button.addEventListener("click", () => selectExample(example));
      return button;
    }));
    if (!filtered.length) $("case-list").append(element("p", "No matching example. Try a component, method or another family.", "muted"));
  }
  function graphEdges() {
    const mode = $("view-mode").value;
    const edges = state.scopeEdgeIds ? state.graph.edges.filter(e => state.scopeEdgeIds.has(e.id)) : state.graph.edges;
    if (mode === "execution") return edges.filter(edge => ["depends_on", "precedes", "implemented_by"].includes(edge.relation));
    if (mode === "reuse") return edges.filter(edge => edge.relation !== "precedes");
    return edges.filter(edge => hierarchyRelations.has(edge.relation) || $("relations").checked);
  }
  function neighbors(id, both = false) {
    const ids = new Set();
    graphEdges().forEach(edge => { if (edge.source === id) ids.add(edge.target); if (both && edge.target === id) ids.add(edge.source); });
    return [...ids].filter(value => state.nodes.has(value));
  }
  function initGraph() {
    if (typeof cytoscape !== "function") throw new Error("The bundled graph renderer could not load. Reload this page or inspect /api/graph.");
    cy = cytoscape({container: $("graph"), elements: [], minZoom: .12, maxZoom: 2.5, wheelSensitivity: .22, selectionType: "single", boxSelectionEnabled: false,
      style: [
        {selector: "node", style: {"label": "data(title)", "shape": "round-rectangle", "background-color": "#e6eee0", "border-color": "#b7ccaf", "border-width": 1.3, "color": "#345241", "text-wrap": "wrap", "text-max-width": "130px", "font-size": "10px", "font-family": "system-ui, sans-serif", "text-valign": "center", "text-halign": "center", "width": 148, "height": 52, "padding": "7px", "overlay-opacity": 0}},
        {selector: "node[layer = 1]", style: {"background-color": "#286e60", "border-color": "#286e60", "color": "#ffffff", "font-weight": 600}},
        {selector: "node[layer = 2]", style: {"background-color": "#f6eddd", "border-color": "#d8bd91", "color": "#82643b"}},
        {selector: "node[layer = 3]", style: {"background-color": "#eaf0f7", "border-color": "#aec4da", "color": "#456d92", "shape": "round-rectangle", "font-family": "monospace", "font-size": "9px"}},
        {selector: "edge", style: {"curve-style": "bezier", "width": 1.2, "line-color": "#bccdbb", "target-arrow-color": "#bccdbb", "target-arrow-shape": "triangle", "arrow-scale": .6, "label": "", "font-size": "8px", "text-rotation": "autorotate", "text-background-opacity": .95, "text-background-color": "#fffefb", "text-background-padding": "2px"}},
        {selector: 'edge[relation = "reuses"]', style: {"line-style": "dashed", "line-color": "#a7bfcf", "target-arrow-color": "#a7bfcf"}},
        {selector: 'edge[relation = "depends_on"], edge[relation = "precedes"]', style: {"line-style": "dashed", "line-color": "#c6b18b", "target-arrow-color": "#c6b18b"}},
        {selector: 'edge[relation = "alternative_to"]', style: {"line-style": "dotted", "line-color": "#bca2c0", "target-arrow-color": "#bca2c0", "target-arrow-shape": "none"}},
        {selector: ".selected", style: {"border-color": "#e08c4b", "border-width": 3, "shadow-blur": 9, "shadow-color": "#dfb17a", "shadow-opacity": .3}},
        {selector: "edge:selected", style: {"line-color": "#65896c", "target-arrow-color": "#65896c", "width": 2.5, "label": "data(label)"}}
      ]});
    cy.on("tap", "node", event => { selectNode(event.target.id()); });
    cy.on("dbltap", "node", event => { selectNode(event.target.id()); expandSelected(); });
    cy.on("tap", "edge", event => {
      const edge = event.target.data();
      const original = state.graph.edges.find(item => item.id === edge.id);
      if (original?.when) status(original.relation.replaceAll("_", " ") + ": " + original.when, false, true);
    });
  }
  function drawGraph(fit = true) {
    const enabled = new Set([1, 2, 3].filter(layer => $("layer-" + layer).checked));
    const nodes = [...state.visible].map(id => state.nodes.get(id)).filter(node => node && enabled.has(Number(node.layer)));
    const ids = new Set(nodes.map(node => node.id));
    const edges = graphEdges().filter(edge => ids.has(edge.source) && ids.has(edge.target));
    cy.batch(() => {
      cy.elements().remove();
      cy.add(nodes.map(node => ({data: {id: node.id, title: node.title || node.name || node.id, layer: Number(node.layer), order: Number.isFinite(node.order) ? node.order : 999}})));
      cy.add(edges.map((edge, index) => ({data: {id: edge.id || "edge-" + index, source: edge.source, target: edge.target, relation: edge.relation, label: edge.relation.replaceAll("_", " ")}})));
      if (state.selected && ids.has(state.selected)) cy.getElementById(state.selected).addClass("selected");
    });
    $("graph-empty").hidden = nodes.length > 0;
    const roots = state.root && ids.has(state.root) ? cy.getElementById(state.root) : undefined;
    if (nodes.length) cy.layout({name: "breadthfirst", directed: true, padding: 28, spacingFactor: 1.15, avoidOverlap: true, roots, depthSort: (a, b) => a.data("order") - b.data("order") || a.data("title").localeCompare(b.data("title")), animate: false, fit}).run();
    $("graph-caption").textContent = nodes.length + " visible nodes · " + edges.length + " relationships · Select, then expand to drill down";
    $("graph-node-list").replaceChildren(...nodes.slice().sort((a,b) => a.layer-b.layer || (a.order ?? 999)-(b.order ?? 999) || a.title.localeCompare(b.title)).map(node => {
      const button = element("button", "L" + node.layer + " · " + node.title, "graph-node-button"); button.type = "button"; button.dataset.nodeId = node.id;
      button.setAttribute("aria-pressed", node.id === state.selected ? "true" : "false");
      button.addEventListener("click", () => selectNode(node.id)); return button;
    }));
  }
  function listSection(label, values) {
    const items = array(values).filter(Boolean);
    if (!items.length) return null;
    const section = element("section"); section.append(element("h3", label));
    const ul = element("ul", null, "detail-list"); items.forEach(item => ul.append(element("li", typeof item === "object" ? item.description || item.name || JSON.stringify(item) : item)));
    section.append(ul); return section;
  }
  function sourceReferences(ids, fallback) {
    const references = [...new Set(array(ids))].map(id => state.sources.get(id) || {id, title: id, url: fallback});
    if (!references.length && fallback) references.push({title: "Original example", url: fallback});
    return references;
  }
  function sourceLink(source, locator) {
    const url = safeUrl(source.url); if (!url) return null;
    const page = Number(locator?.page);
    const hasPage = Number.isInteger(page) && page > 0 && (locator?.source_id === source.id || source.id === "manual5" || source.content_type === "application/pdf");
    const target = new URL(url); if (hasPage) target.hash = "page=" + page;
    const link = element("a", (source.title || source.id) + (hasPage ? " · PDF page " + page : "") + " ↗");
    link.href = target.href; link.target = "_blank"; link.rel = "noopener noreferrer";
    if (hasPage) link.title = "Physical PDF page " + page;
    return link;
  }
  function showSources(ids, fallback, locator) {
    const references = sourceReferences(ids, fallback);
    $("source-list").replaceChildren(...references.map(source => {
      const card = element("div", null, "source-card");
      const link = sourceLink(source, locator);
      card.append(link || element("strong", source.title || source.id));
      card.append(element("small", [source.status, source.retrieved_at ? "Retrieved " + source.retrieved_at.slice(0, 10) : null].filter(Boolean).join(" · ")));
      if (source.sha256) card.append(element("small", "SHA-256 " + source.sha256.slice(0, 16) + "…"));
      if (source.summary) card.append(element("p", source.summary));
      return card;
    }));
    if (!references.length) $("source-list").append(element("p", "No source reference attached to this item.", "muted"));
  }
  function showGeometrySources(ids) {
    const references = sourceReferences(ids);
    $("geometry-sources").replaceChildren(element("span", "Construction evidence:"), ...references.map(source => sourceLink(source) || element("span", source.title || source.id)));
    $("geometry-sources").hidden = references.length === 0;
  }
  function sourceContextSection(contexts) {
    const section = element("section", null, "source-contexts"); section.append(element("h3", "Source-specific conditions"));
    section.append(element("p", "Read each method with the conditions supplied by its own source.", "muted"));
    contexts.forEach(context => {
      const card = element("div", null, "source-context");
      const references = sourceReferences(context.source_ids);
      const heading = element("div", null, "context-sources");
      references.forEach(source => heading.append(sourceLink(source) || element("strong", source.title || source.id)));
      if (!references.length) heading.append(element("strong", "Source context"));
      card.append(heading);
      if (context.description) card.append(element("p", context.description));
      [["Applicability & constraints", context.constraints], ["Required inputs", context.inputs], ["Produced outputs", context.outputs], ["Required capabilities", context.capabilities]].forEach(([label, values]) => {
        const details = listSection(label, values); if (details) card.append(details);
      });
      section.append(card);
    });
    return section;
  }
  function selectNode(id, options = {}) {
    const node = state.nodes.get(id); if (!node) return;
    state.selected = id;
    cy.nodes().removeClass("selected"); cy.getElementById(id).addClass("selected");
    $("graph-node-list").querySelectorAll("button").forEach(button => button.setAttribute("aria-pressed", button.dataset.nodeId === id ? "true" : "false"));
    $("node-title").textContent = node.title || node.id;
    $("node-layer").textContent = "L" + node.layer + " · " + (layerNames[node.layer] || node.kind);
    $("node-description").textContent = node.description || "";
    $("node-verification").replaceChildren(verification(node.verification));
    const contexts = array(node.source_contexts);
    const sections = contexts.length ? [sourceContextSection(contexts)] : [listSection("Required inputs", node.inputs), listSection("Produced outputs", node.outputs), listSection("Applicability & constraints", node.constraints), listSection("Required capabilities", node.capabilities), listSection("Source-specific details", node.additional_descriptions)].filter(Boolean);
    if (node.parameters?.length) {
      const section = element("section"); section.append(element("h3", "Parameters"));
      const table = element("table", null, "parameter-table");
      node.parameters.forEach(parameter => { const row = element("tr"); row.append(element("td", parameter.name + (parameter.unit ? " [" + parameter.unit + "]" : "")), element("td", parameter.description || "")); table.append(row); });
      section.append(table); sections.unshift(section);
    }
    if (node.signature) {
      const section = element("section"); section.append(element("h3", "Exact function signature"));
      const signature = element("pre", null, "function-signature"); signature.id = "node-signature"; signature.append(element("code", node.signature));
      section.append(signature); sections.unshift(section);
    }
    if (node.executable_component) {
      const component = node.executable_component;
      const section = element("section"); section.append(element("h3", "Executable component · " + component.id));
      section.append(element("p", "Applies to " + array(component.physics).join(", ") + " · " + array(component.problem_types).join(", "), "muted"));
      const table = element("table", null, "parameter-table");
      array(component.parameters).forEach(parameter => { const row = element("tr"); row.append(element("td", parameter.name + (parameter.unit ? " [" + parameter.unit + "]" : "")), element("td", (parameter.description || "") + (parameter.required ? " (required)" : ""))); table.append(row); });
      section.append(table);
      if (array(node.verification?.case_ids).length) section.append(element("p", "Checked natively by: " + node.verification.case_ids.join(", "), "muted"));
      sections.push(section);
    }
    if (node.function || node.operation) sections.push(listSection("FEMM operation", [node.function || node.operation]));
    if (node.tags?.length) { const tags = element("div", null, "tag-list"); node.tags.forEach(tag => tags.append(element("span", tag, "tag"))); sections.push(tags); }
    $("node-details").replaceChildren(...sections);
    showSources(node.source_ids, null, node.source_locator);
    const related = state.graph.edges.filter(edge => edge.source === id || edge.target === id).slice(0, 18);
    $("related-list").replaceChildren(...related.map(edge => {
      const other = state.nodes.get(edge.source === id ? edge.target : edge.source);
      if (!other) return document.createTextNode("");
      const button = element("button", null, "related-button"); button.type = "button";
      button.append(element("span", other.title || other.id), element("small", edge.relation.replaceAll("_", " ") + (edge.when ? " · conditional" : "")));
      button.addEventListener("click", () => { state.visible.add(other.id); drawGraph(); selectNode(other.id); });
      return button;
    }));
    const examples = state.examples.filter(example => example.workflow_id === id || array(example.node_ids).includes(id));
    if (examples.length) {
      $("related-list").append(element("p", "Used in " + examples.length + (examples.length === 1 ? " example" : " examples"), "view-note"));
      examples.slice(0, 8).forEach(example => {
        const button = element("button", example.title, "related-button"); button.type = "button"; button.dataset.relatedExample = example.id;
        button.addEventListener("click", () => selectExample(example)); $("related-list").append(button);
      });
    }
    $("related-section").hidden = related.length === 0 && examples.length === 0;
    $("expand-node").disabled = neighbors(id, $("view-mode").value === "reuse").length === 0;
    $("collapse-node").disabled = false;
    const recipeId = node.recipe_id;
    if (recipeId && !options.noRecipe && state.recipes.some(recipe => recipe.id === recipeId)) chooseRecipe(recipeId);
  }
  function overview() {
    state.exampleRequest = (state.exampleRequest || 0) + 1; state.scopeIds = null; state.scopeEdgeIds = null;
    state.example = null;
    state.root = state.nodes.has("workflow:magnetics") ? "workflow:magnetics" : state.graph.nodes.find(node => node.layer === 1)?.id || state.graph.nodes[0]?.id;
    state.visible = new Set(state.root ? [state.root, ...neighbors(state.root)] : []);
    $("graph-title").textContent = "The shared engineering workflow";
    renderCases(); renderArtifacts(); drawGraph();
    if (state.root) selectNode(state.root, {noRecipe: true});
  }
  async function selectExample(example) {
    const request = state.exampleRequest = (state.exampleRequest || 0) + 1;
    let scoped;
    try { scoped = await api("/api/examples/" + encodeURIComponent(example.id)); }
    catch (error) { status(error.message, true); return; }
    if (request !== state.exampleRequest) return;
    state.scopeIds = new Set(scoped.nodes.map(node => node.id));
    state.scopeEdgeIds = new Set(scoped.edges.map(edge => edge.id));
    state.example = example;
    state.root = state.nodes.has(example.workflow_id) ? example.workflow_id : "workflow:magnetics";
    state.visible = new Set([state.root, ...array(example.node_ids)].filter(id => state.nodes.has(id)));
    for (const id of [...state.visible]) if (state.nodes.get(id)?.layer === 1) neighbors(id).forEach(child => state.visible.add(child));
    $("graph-title").textContent = example.title;
    renderCases(); const artifacts = renderArtifacts(); drawGraph();
    const workflow = state.nodes.get(state.root);
    if (workflow) selectNode(workflow.id, {noRecipe: true});
    $("node-title").textContent = example.title;
    $("node-description").textContent = example.summary || workflow?.description || "";
    const reproductions = array(example.reproductions).map(item => item.title + " — " + item.status.replaceAll("_", " ") +
      (array(item.levels_passed).length ? " (passed: " + item.levels_passed.join(", ") + ")" : ""));
    const details = [listSection("Native reproduction", reproductions), listSection("Assumptions", example.assumptions),
                     listSection("Limits of this example", example.limitations), listSection("Physics", example.physics)].filter(Boolean);
    $("node-details").replaceChildren(...details);
    showSources(example.source_ids, example.source_url);
    clearConstruction("Choose a parameterized reconstruction or inspect an available original model for this example.");
    if (artifacts.length) { $("artifact-select").value = artifacts[0].id; loadArtifact(artifacts[0].id); }
    status("Opened " + example.title + ". Expand a component to inspect its exact operations.");
  }
  function expandSelected() {
    if (!state.selected) return;
    const adjacent = neighbors(state.selected, $("view-mode").value === "reuse");
    const additions = adjacent.filter(id => !state.visible.has(id));
    additions.slice(0, 80).forEach(id => state.visible.add(id));
    drawGraph();
    status(additions.length ? "Expanded " + Math.min(additions.length, 80) + " related nodes." + (additions.length > 80 ? " Expand again to reveal more." : "") : "All immediate relationships are already visible.");
  }
  function collapseSelected() {
    if (!state.selected) return;
    const descendants = new Set(), queue = [...neighbors(state.selected)];
    while (queue.length && descendants.size < 5000) { const id = queue.shift(); if (id === state.selected || descendants.has(id)) continue; descendants.add(id); queue.push(...neighbors(id)); }
    descendants.forEach(id => { if (id !== state.root) state.visible.delete(id); });
    state.visible.add(state.selected); drawGraph();
  }
  function renderRecipes() {
    $("recipe-select").replaceChildren(element("option", "Choose a construction recipe…"));
    state.recipes.forEach(recipe => { const option = element("option", recipe.title || recipe.name || recipe.id); option.value = recipe.id; $("recipe-select").append(option); });
  }
  function parameterDefinitions(recipe) {
    const parameters = Array.isArray(recipe.parameters) ? recipe.parameters : Object.entries(recipe.parameters || {}).map(([name, value]) => typeof value === "object" ? {name, ...value} : {name, default: value});
    if (parameters.length) return parameters;
    return Object.entries(recipe.defaults || {}).map(([name, value]) => ({name, default: value}));
  }
  function renderParameters(recipe) {
    const defaults = recipe.defaults || {};
    $("recipe-parameters").replaceChildren(...parameterDefinitions(recipe).map(parameter => {
      const label = element("label");
      label.append(document.createTextNode(parameter.label || parameter.name.replaceAll("_", " ")));
      if (parameter.unit) label.append(element("span", " · " + parameter.unit, "parameter-unit"));
      const input = document.createElement("input"); input.type = "number"; input.dataset.parameter = parameter.name;
      input.value = state.parameters[parameter.name] ?? parameter.default ?? parameter.value ?? defaults[parameter.name] ?? "";
      input.step = parameter.step ?? (parameter.type === "integer" || parameter.integer ? 1 : "any");
      if (parameter.min !== undefined || parameter.minimum !== undefined) input.min = parameter.min ?? parameter.minimum;
      if (parameter.max !== undefined || parameter.maximum !== undefined) input.max = parameter.max ?? parameter.maximum;
      input.required = true; if (parameter.description) { input.title = parameter.description; label.title = parameter.description; }
      label.append(input); return label;
    }));
  }
  async function chooseRecipe(id) {
    const recipe = state.recipes.find(item => item.id === id); if (!recipe) return;
    state.recipe = recipe; state.parameters = {}; $("recipe-select").value = id;
    renderParameters(recipe); $("generate-recipe").disabled = false;
    await generateRecipe(true);
  }
  function getParameters() {
    const parameters = {};
    for (const input of $("recipe-parameters").querySelectorAll("input")) {
      if (input.value !== "") { const value = Number(input.value); if (!Number.isFinite(value)) throw new Error("Construction parameters must be finite numbers."); parameters[input.dataset.parameter] = value; }
    }
    return parameters;
  }
  async function generateRecipe(useDefaults = false) {
    if (!state.recipe) return;
    const request = ++state.request;
    try {
      const parameters = useDefaults ? {} : getParameters();
      $("generate-recipe").disabled = true;
      const result = await api("/api/geometry?recipe=" + encodeURIComponent(state.recipe.id) + "&parameters=" + encodeURIComponent(JSON.stringify(parameters)));
      if (request !== state.request) return;
      state.parameters = result.parameters || parameters;
      renderParameters(state.recipe); renderGeometry(result);
      const note = typeof result.verification === "object" ? result.verification.note : "";
      $("geometry-note").textContent = note || state.recipe.description || "Parameterized reconstruction. Defaults are illustrative unless a source explicitly establishes them. Preview checks do not establish numerical convergence.";
      $("artifact-select").value = "";
      ["python", "lua", "fem", "svg"].forEach(format => {
        const anchor = $("download-" + format);
        anchor.href = "/api/download?recipe=" + encodeURIComponent(state.recipe.id) + "&parameters=" + encodeURIComponent(JSON.stringify(state.parameters)) + "&format=" + format;
        anchor.setAttribute("download", state.recipe.id + "." + (format === "python" ? "py" : format)); anchor.hidden = false;
      });
      status("Construction regenerated. Select a line or arc to inspect its operation.");
    } catch (error) {
      if (request === state.request) { $("geometry-validation").textContent = error.message; $("geometry-validation").classList.add("error"); status(error.message, true); }
    } finally { if (request === state.request) $("generate-recipe").disabled = !state.recipe; }
  }
  function sanitizeSvg(svg) {
    const doc = new DOMParser().parseFromString(svg, "image/svg+xml");
    if (doc.querySelector("parsererror") || doc.documentElement.localName !== "svg") throw new Error("The geometry renderer returned an invalid SVG.");
    doc.querySelectorAll("script,foreignObject,iframe,object,embed,style,animate,set,animateTransform").forEach(node => node.remove());
    doc.querySelectorAll("*").forEach(node => [...node.attributes].forEach(attribute => {
      const name = attribute.name.toLowerCase(), value = attribute.value.trim();
      if (name.startsWith("on") || name === "style" && /url\s*\(|expression\s*\(/i.test(value) || /(?:^|:)href$/.test(name) && !value.startsWith("#")) node.removeAttribute(attribute.name);
    }));
    const root = document.importNode(doc.documentElement, true); root.setAttribute("role", "img"); root.setAttribute("aria-label", "Inspectable geometry construction");
    return root;
  }
  function renderGeometry(result) {
    state.geometry = result;
    showGeometrySources(result.source_ids);
    $("geometry-preview").replaceChildren(sanitizeSvg(result.svg));
    const operations = result.operations || [];
    const steps = result.construction_steps || [];
    $("construction-step-list").replaceChildren(...steps.map(step => element("li", step)));
    $("construction-steps").hidden = steps.length === 0;
    $("operation-count").textContent = operations.length.toLocaleString();
    $("operations-list").replaceChildren(...operations.slice(0, 3000).map((operation, index) => {
      const button = element("button", null, "operation-button"); button.type = "button"; button.dataset.operationId = operation.id || "op-" + index;
      const functionName = operation.function || operation.operation || operation.op || "operation";
      button.append(element("strong", String(index + 1).padStart(2, "0") + " · " + functionName));
      const args = operation.arguments ?? operation.args ?? operation.parameters;
      if (args !== undefined) button.append(document.createTextNode(text(args)));
      button.addEventListener("click", () => selectOperation(operation)); return button;
    }));
    const validation = result.validation;
    $("geometry-validation").classList.remove("error");
    if (Array.isArray(validation)) $("geometry-validation").textContent = validation.map(item => typeof item === "string" ? item : item.detail || item.message || item.name).join(" · ");
    else if (validation && typeof validation === "object") $("geometry-validation").textContent = validation.note || validation.summary || validation.message || "Geometry checks: " + (validation.status || "reported") + (validation.node_count ? " · " + validation.node_count + " nodes" : "");
    else $("geometry-validation").textContent = text(validation) || "Inspect the construction and its parameter constraints before using it in an analysis.";
    $("geometry-preview").querySelector("svg").addEventListener("click", event => {
      let target = event.target;
      while (target && target !== event.currentTarget) {
        const explicit = target.dataset.operationId || target.dataset.operation || target.dataset.op;
        const operation = operations.find(op => op.id === explicit || op.id === target.id || array(op.geometry_ids).includes(target.id));
        if (operation) { selectOperation(operation); return; }
        target = target.parentElement;
      }
    });
  }
  function selectOperation(operation) {
    $("operations-list").querySelectorAll("button").forEach(button => button.classList.toggle("selected", button.dataset.operationId === operation.id));
    $("geometry-preview").querySelectorAll(".geometry-selected").forEach(item => item.classList.remove("geometry-selected"));
    const ids = new Set([operation.id, ...array(operation.geometry_ids)]);
    $("geometry-preview").querySelectorAll("[id],[data-operation-id],[data-operation],[data-op]").forEach(item => {
      if (ids.has(item.id) || item.dataset.operationId === operation.id || item.dataset.operation === operation.id || item.dataset.op === operation.id) item.classList.add("geometry-selected");
    });
    const functionName = operation.function || operation.operation || operation.op;
    const primitive = state.nodes.get("primitive:" + functionName) || [...state.nodes.values()].find(node => node.layer === 3 && (node.function === functionName || node.title === functionName));
    if (primitive) { state.visible.add(primitive.id); drawGraph(false); selectNode(primitive.id, {noRecipe: true}); }
    const selected = [...$("operations-list").querySelectorAll("button")].find(button => button.dataset.operationId === operation.id);
    selected?.scrollIntoView({block: "nearest", behavior: "smooth"});
    status("Selected " + (operation.id || functionName) + " · " + functionName);
  }
  function renderArtifacts() {
    const sourceIds = new Set(state.example?.source_ids || []);
    const artifacts = state.artifacts.filter(artifact => artifact.geometry_available && (!state.example || array(artifact.source_ids).some(id => sourceIds.has(id))));
    $("artifact-select").replaceChildren(element("option", artifacts.length ? "Available native FEMM files…" : "No native preview for this selection"));
    artifacts.slice(0, 200).forEach(artifact => { const option = element("option", artifact.filename || artifact.title || artifact.id); option.value = artifact.id; $("artifact-select").append(option); });
    $("artifact-select").disabled = artifacts.length === 0;
    return artifacts;
  }
  function clearConstruction(note) {
    state.request += 1; state.geometry = null; state.recipe = null; state.parameters = {};
    $("recipe-select").value = ""; $("recipe-parameters").replaceChildren(); $("generate-recipe").disabled = true;
    $("geometry-preview").replaceChildren(element("div", "Select a recipe or original source model to inspect its construction.", "empty-state"));
    $("operations-list").replaceChildren(); $("operation-count").textContent = ""; $("geometry-validation").textContent = "";
    $("construction-steps").hidden = true; $("construction-step-list").replaceChildren();
    showGeometrySources([]);
    $("geometry-note").textContent = note;
    ["python", "lua", "fem", "svg"].forEach(format => { $("download-" + format).hidden = true; });
  }
  async function loadArtifact(id) {
    const request = ++state.request;
    if (!id) return;
    try {
      const result = await api("/api/geometry?artifact=" + encodeURIComponent(id));
      if (request !== state.request) return;
      renderGeometry(result); state.recipe = null; $("recipe-select").value = ""; $("recipe-parameters").replaceChildren(); $("generate-recipe").disabled = true;
      ["python", "lua", "fem", "svg"].forEach(format => { $("download-" + format).hidden = true; });
      $("geometry-note").textContent = "Original example geometry. Primitive operations reconstruct the saved model; they do not establish its original authoring sequence or numerical validity.";
      status("Source model loaded. Select a primitive to inspect its construction.");
    } catch (error) { if (request === state.request) status(error.message, true); }
  }

  async function start() {
    try {
      const [graph, recipes] = await Promise.all([api("/api/graph"), api("/api/recipes")]);
      state.graph = graph; state.nodes = new Map(graph.nodes.map(node => [node.id, node])); state.examples = [...(graph.examples || [])].sort((a, b) => a.title.localeCompare(b.title));
      state.sources = new Map((graph.sources || []).map(source => [source.id, source])); state.artifacts = graph.artifacts || [];
      state.recipes = Array.isArray(recipes) ? recipes : recipes.recipes || [];
      [...new Set(state.examples.map(example => example.family).filter(Boolean))].sort().forEach(family => { const option = element("option", family.replaceAll("_", " ")); option.value = family; $("family-filter").append(option); });
      initGraph(); renderSummary(); renderCoverage(); renderRecipes(); overview();
      const initialRecipe = state.recipes.find(recipe => recipe.id === "stator_tooth") || state.recipes[0];
      if (initialRecipe) await chooseRecipe(initialRecipe.id);
      status("Atlas ready. Choose an example or expand a workflow node.");
      if (!state.artifacts.length) api("/api/sources").then(data => { state.artifacts = data.artifacts || []; renderArtifacts(); renderCoverage(data); }).catch(() => {});
    } catch (error) { status(error.message, true, true); $("summary").textContent = "The atlas could not load. " + error.message; }
  }
  $("case-search").addEventListener("input", renderCases); $("family-filter").addEventListener("change", renderCases);
  $("overview").addEventListener("click", overview); $("fit-graph").addEventListener("click", () => cy?.fit(undefined, 30));
  $("expand-node").addEventListener("click", expandSelected); $("collapse-node").addEventListener("click", collapseSelected);
  ["layer-1", "layer-2", "layer-3", "relations"].forEach(id => $(id).addEventListener("change", () => drawGraph()));
  $("view-mode").addEventListener("change", () => { if ($("view-mode").value === "reuse" && state.selected) neighbors(state.selected, true).slice(0, 100).forEach(id => state.visible.add(id)); drawGraph(); });
  $("recipe-select").addEventListener("change", () => { if ($("recipe-select").value) chooseRecipe($("recipe-select").value); });
  $("recipe-form").addEventListener("submit", event => { event.preventDefault(); if ($("recipe-form").reportValidity()) generateRecipe(); });
  $("artifact-select").addEventListener("change", () => loadArtifact($("artifact-select").value));
  $("graph").addEventListener("keydown", event => { if (event.key === "Enter" || event.key === "ArrowRight") { event.preventDefault(); expandSelected(); } else if (event.key === "ArrowLeft") { event.preventDefault(); collapseSelected(); } });
  window.addEventListener("brain:select-node", event => {
    const id = event.detail?.id;
    if (state.nodes.has(id)) {
      if (state.scopeIds && !state.scopeIds.has(id)) { state.scopeIds = null; state.scopeEdgeIds = null; state.example = null; state.visible = new Set(); }
      state.visible.add(id); drawGraph(); selectNode(id, {noRecipe: true});
    }
  });
  start();
})();
