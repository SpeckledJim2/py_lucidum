const TREE_PALETTES = {
  divergent: ["#00441b", "#1b7837", "#5aae61", "#a6dba0", "#d9f0d3", "#fddbc7", "#f4a582", "#d6604d", "#b2182b", "#67001f"],
  spectral: ["#2c7bb6", "#00a6ca", "#00ccbc", "#90eb9d", "#ffff8c", "#f9d057", "#f29e2e", "#e76818", "#d7191c", "#a50026"],
  viridis: ["#fde725", "#b5de2b", "#6ece58", "#35b779", "#1f9e89", "#26828e", "#31688e", "#3e4989", "#482878", "#440154"],
};

const D3_SRC = "/static/vendor/d3/d3.min.js";
const NODE_MIN_WIDTH = 124;
const NODE_MAX_WIDTH = 220;
const NODE_LINE_HEIGHT = 18;
const NODE_VERTICAL_PADDING = 14;
const NODE_HORIZONTAL_PADDING = 18;
const EDGE_LABEL_WRAP_CHARS = 34;
const CATEGORICAL_EDGE_LABEL_WRAP_CHARS = 20;
const CATEGORICAL_EDGE_LABEL_POSITION = 0.36;
const CATEGORICAL_EDGE_LABEL_X_OFFSET = 32;
const DEFAULT_SUMMARY_WIDTH = 560;
const MIN_SUMMARY_WIDTH = 420;
const MIN_DIAGRAM_WIDTH = 360;

let d3Promise = null;

export function createGbmTreeViewer({ api, escapeHtml, loadTabulator, setGbmNotice }) {
  let currentModelId = "";
  let selectedTree = null;
  let summaryRows = [];
  let summaryTable = null;
  let selectedDetail = null;
  let palette = "plain";
  let resizeObserver = null;
  let zoomBehavior = null;
  let zoomSvg = null;
  let resetTransform = null;
  let treeBounds = null;
  let resizeFrame = null;
  let summaryWidth = DEFAULT_SUMMARY_WIDTH;
  let resizePointerId = null;
  let highlightedNodeId = "";
  let renderToken = 0;

  async function render(modelId) {
    const nextModelId = String(modelId || "");
    if (nextModelId !== currentModelId) {
      selectedTree = null;
      selectedDetail = null;
      highlightedNodeId = "";
    }
    currentModelId = nextModelId;
    const root = document.getElementById("gbmTreeViewer");
    if (!root) return;
    applySummaryWidth(root, summaryWidth);
    bindPaletteControls();
    bindZoomControls();
    bindTreeResizer(root);
    bindSearch();
    if (!currentModelId) {
      selectedTree = null;
      summaryRows = [];
      selectedDetail = null;
      highlightedNodeId = "";
      clearSummaryTable();
      updateTreeDetailSummary(null);
      renderEmpty("Select a saved GBM model to inspect its trees.");
      return;
    }

    const token = ++renderToken;
    try {
      const payload = await api(`/api/gbm/models/${encodeURIComponent(currentModelId)}/trees`, { method: "GET" });
      if (token !== renderToken) return;
      summaryRows = Array.isArray(payload.trees) ? payload.trees : [];
      await renderSummaryTable(summaryRows);
      const available = new Set(summaryRows.map((row) => Number(row.tree)));
      const nextTree = available.has(Number(selectedTree)) ? Number(selectedTree) : Number(summaryRows[0]?.tree);
      if (Number.isFinite(nextTree)) {
        await selectTree(nextTree, { preserveNotice: true });
      } else {
        selectedTree = null;
        selectedDetail = null;
        highlightedNodeId = "";
        updateTreeDetailSummary(null);
        renderEmpty("No tree artifacts are available for this GBM.");
      }
    } catch (error) {
      if (token !== renderToken) return;
      clearSummaryTable();
      updateTreeDetailSummary(null);
      renderEmpty("Unable to load tree artifacts.");
      setGbmNotice(error.message);
    }
  }

  function dispose() {
    renderToken += 1;
    resizeObserver?.disconnect();
    resizeObserver = null;
    cancelResizeFrame();
    zoomBehavior = null;
    zoomSvg = null;
    resetTransform = null;
    treeBounds = null;
    highlightedNodeId = "";
    clearSummaryTable();
    selectedDetail = null;
    updateTreeDetailSummary(null);
    const target = chartTarget();
    if (target) target.innerHTML = "";
  }

  function refreshTheme() {
    if (selectedDetail) drawTree(selectedDetail);
  }

  async function renderSummaryTable(rows) {
    const target = document.getElementById("gbmTreeSummaryGrid");
    const fallback = document.getElementById("gbmTreeSummaryFallback");
    if (!target) return;
    fallback?.replaceChildren();
    try {
      const Tabulator = await loadTabulator();
      if (summaryTable) {
        await summaryTable.replaceData(rows);
      } else {
        summaryTable = new Tabulator(target, {
          data: rows,
          index: "tree",
          layout: "fitColumns",
          height: "100%",
          selectableRows: 1,
          placeholder: "No trees",
          columns: [
            { title: "tree", field: "tree", width: 46, minWidth: 46, hozAlign: "right", headerSortStartingDir: "asc" },
            { title: "dim", field: "dim", width: 42, minWidth: 42, hozAlign: "right" },
            { title: "features", field: "features", widthGrow: 2, formatter: treeFeaturesFormatter },
            { title: "gain", field: "gain", width: 96, hozAlign: "right", formatter: treeGainFormatter },
          ],
        });
        summaryTable.on("rowClick", (_, row) => selectTree(Number(row.getData().tree)));
        await waitForTableBuilt(summaryTable);
      }
      filterSummaryTable();
      selectSummaryRow(selectedTree);
    } catch (_) {
      clearSummaryTable();
      renderFallbackSummary(rows, fallback || target);
    }
  }

  function renderFallbackSummary(rows, target) {
    if (!target) return;
    target.innerHTML = `
      <table class="gbm-model-table gbm-tree-fallback-table">
        <thead><tr><th>tree</th><th>dim</th><th>features</th><th>gain</th></tr></thead>
        <tbody>
          ${rows.map((row) => `
            <tr data-gbm-tree-row="${escapeHtml(row.tree)}">
              <td class="numeric">${escapeHtml(row.tree)}</td>
              <td class="numeric">${escapeHtml(row.dim)}</td>
              <td>${escapeHtml(row.features || "")}</td>
              <td class="numeric">${escapeHtml(formatGain(row.gain))}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    `;
    for (const row of target.querySelectorAll("[data-gbm-tree-row]")) {
      row.addEventListener("click", () => selectTree(Number(row.dataset.gbmTreeRow)));
    }
    selectFallbackRow(selectedTree);
  }

  async function selectTree(treeIndex, options = {}) {
    if (!currentModelId || !Number.isFinite(Number(treeIndex))) return;
    if (Number(selectedTree) !== Number(treeIndex)) highlightedNodeId = "";
    selectedTree = Number(treeIndex);
    selectSummaryRow(selectedTree);
    updateTreeDetailSummary(selectedSummaryRow(selectedTree));
    const token = ++renderToken;
    try {
      const detail = await api(
        `/api/gbm/models/${encodeURIComponent(currentModelId)}/trees/${encodeURIComponent(selectedTree)}`,
        { method: "GET" },
      );
      if (token !== renderToken) return;
      selectedDetail = detail;
      if (!options.preserveNotice) setGbmNotice("");
      await drawTree(detail);
    } catch (error) {
      if (token !== renderToken) return;
      selectedDetail = null;
      updateTreeDetailSummary(selectedSummaryRow(selectedTree));
      renderEmpty("Unable to load the selected tree.");
      setGbmNotice(error.message);
    }
  }

  async function drawTree(detail) {
    const target = chartTarget();
    if (!target) return;
    if (!detail?.root) {
      renderEmpty("No tree structure is available for this tree.");
      return;
    }
    const d3 = await loadD3();
    target.innerHTML = "";
    resizeObserver?.disconnect();
    resizeObserver = null;
    cancelResizeFrame();
    treeBounds = null;

    const { width, height } = svgSize(target);
    const svg = d3.select(target)
      .append("svg")
      .attr("class", "gbm-tree-svg")
      .attr("role", "img")
      .attr("aria-label", `GBM tree ${detail.tree}`)
      .attr("width", "100%")
      .attr("height", "100%")
      .attr("viewBox", `0 0 ${width} ${height}`);

    const defs = svg.append("defs");
    defs.append("marker")
      .attr("id", "gbmTreeArrow")
      .attr("viewBox", "0 -5 10 10")
      .attr("refX", 9)
      .attr("refY", 0)
      .attr("markerWidth", 7)
      .attr("markerHeight", 7)
      .attr("orient", "auto")
      .append("path")
      .attr("d", "M0,-5L10,0L0,5")
      .attr("class", "gbm-tree-arrow");

    const viewport = svg.append("g").attr("class", "gbm-tree-viewport");
    const linkLayer = viewport.append("g").attr("class", "gbm-tree-links");
    const labelLayer = viewport.append("g").attr("class", "gbm-tree-link-labels");
    const nodeLayer = viewport.append("g").attr("class", "gbm-tree-nodes");

    const hierarchy = d3.hierarchy(detail.root);
    hierarchy.each((node) => {
      const lines = normaliseLabelLines(node.data.label);
      node.data._labelLines = lines;
      node.data._boxWidth = measureNodeWidth(lines);
      node.data._boxHeight = lines.length * NODE_LINE_HEIGHT + NODE_VERTICAL_PADDING * 2;
    });

    const layout = d3.tree().nodeSize([110, 260]).separation((left, right) => (left.parent === right.parent ? 1.15 : 1.5));
    layout(hierarchy);
    const nodes = hierarchy.descendants();
    const links = hierarchy.links();
    const xMin = Math.min(...nodes.map((node) => node.x - node.data._boxHeight / 2));
    const xMax = Math.max(...nodes.map((node) => node.x + node.data._boxHeight / 2));
    const yMin = Math.min(...nodes.map((node) => node.y - node.data._boxWidth / 2));
    const yMax = Math.max(...nodes.map((node) => node.y + node.data._boxWidth / 2));
    treeBounds = { xMin, xMax, yMin, yMax };

    zoomBehavior = d3.zoom()
      .scaleExtent([0.03, 24])
      .extent(() => {
        const size = svgSize(target);
        return [[0, 0], [size.width, size.height]];
      })
      .translateExtent([[-100000, -100000], [100000, 100000]])
      .wheelDelta((event) => {
        const unit = event.deltaMode === 1 ? 0.055 : event.deltaMode === 2 ? 0.7 : 0.0022;
        return -event.deltaY * unit;
      })
      .on("zoom", (event) => viewport.attr("transform", event.transform));
    zoomSvg = svg;
    svg.call(zoomBehavior).on("dblclick.zoom", null);
    fitTree(false);

    linkLayer.selectAll("path")
      .data(links)
      .join("path")
      .attr("class", (link) => `gbm-tree-link${link.target.data.default_branch ? " gbm-tree-link-default" : ""}`)
      .attr("marker-end", "url(#gbmTreeArrow)")
      .attr("d", (link) => elbowPath(link.source, link.target));

    labelLayer.selectAll("text")
      .data(links)
      .join("text")
      .attr("class", "gbm-tree-edge-label")
      .attr("x", (link) => edgeLabelPlacement(link).x)
      .attr("y", (link) => edgeLabelPlacement(link).y)
      .attr("text-anchor", "middle")
      .attr("dominant-baseline", "middle")
      .attr("data-tooltip", (link) => link.target.data.edge_tooltip || link.target.data.edge_label || "")
      .each(function renderEdgeLabel(link) {
        const placement = edgeLabelPlacement(link);
        const lines = edgeLabelLines(link.target.data.edge_label || "");
        d3.select(this).selectAll("tspan")
          .data(lines.map((line, index) => ({ line, index, total: lines.length, x: placement.x })))
          .join("tspan")
          .attr("x", (line) => line.x)
          .attr("dy", (line) => line.index === 0 ? `${-(line.total - 1) * 0.62}em` : "1.24em")
          .text((line) => line.line);
      });

    const node = nodeLayer.selectAll("g")
      .data(nodes)
      .join("g")
      .attr("class", (item) => `gbm-tree-node gbm-tree-node-${item.data.type || "split"}`)
      .attr("transform", (item) => `translate(${item.y},${item.x})`)
      .on("click", (event, item) => {
        event.stopPropagation();
        highlightedNodeId = highlightedNodeId === item.data.id ? "" : item.data.id;
        updateTreeHighlight(highlightedNodeId);
      });

    node.each(function appendShape(item) {
      const selection = d3.select(this);
      const fill = nodeFill(item.data, detail.values || [], palette);
      item.data._fill = fill;
      item.data._textFill = readableTextColor(fill);
      if (item.data.type === "leaf") {
        selection.append("ellipse")
          .attr("class", "gbm-tree-leaf-node")
          .attr("rx", item.data._boxWidth / 2)
          .attr("ry", item.data._boxHeight / 2)
          .attr("fill", fill);
      } else {
        selection.append("rect")
          .attr("class", "gbm-tree-split-node")
          .attr("x", -item.data._boxWidth / 2)
          .attr("y", -item.data._boxHeight / 2)
          .attr("width", item.data._boxWidth)
          .attr("height", item.data._boxHeight)
          .attr("rx", 3)
          .attr("fill", fill);
      }
    });

    node.append("text")
      .attr("class", "gbm-tree-node-label")
      .attr("fill", (item) => item.data._textFill || "#111827")
      .attr("text-anchor", "middle")
      .attr("dominant-baseline", "middle")
      .selectAll("tspan")
      .data((item) => item.data._labelLines.map((line, index) => ({
        line,
        index,
        total: item.data._labelLines.length,
        emphasis: isEmphasisLabelLine(item.data, index),
      })))
      .join("tspan")
      .attr("x", 0)
      .attr("dy", (line) => line.index === 0 ? `${-(line.total - 1) * 0.62}em` : "1.24em")
      .attr("font-weight", (line) => line.emphasis ? 700 : 400)
      .text((line) => line.line);

    updateTreeHighlight(highlightedNodeId);

    if (window.ResizeObserver) {
      resizeObserver = new ResizeObserver(scheduleSvgResize);
      resizeObserver.observe(target);
    }
  }

  function bindPaletteControls() {
    for (const button of document.querySelectorAll("[data-gbm-tree-palette]")) {
      button.classList.toggle("active", button.dataset.gbmTreePalette === palette);
      button.setAttribute("aria-pressed", String(button.dataset.gbmTreePalette === palette));
      button.onclick = () => {
        palette = button.dataset.gbmTreePalette || "plain";
        bindPaletteControls();
        updateTreePalette();
      };
    }
  }

  function updateTreePalette() {
    if (!zoomSvg || !selectedDetail || !window.d3) return;
    const values = selectedDetail.values || [];
    zoomSvg.selectAll(".gbm-tree-node").each(function updateNodePalette(item) {
      const fill = nodeFill(item.data, values, palette);
      const textFill = readableTextColor(fill);
      item.data._fill = fill;
      item.data._textFill = textFill;
      const selection = window.d3.select(this);
      selection.select("rect.gbm-tree-split-node, ellipse.gbm-tree-leaf-node").attr("fill", fill);
      selection.select("text.gbm-tree-node-label").attr("fill", textFill);
    });
  }

  function updateTreeHighlight(nodeId) {
    if (!zoomSvg || !window.d3) return;
    let targetNode = null;
    zoomSvg.selectAll(".gbm-tree-node").each((item) => {
      if (item.data.id === nodeId) targetNode = item;
    });
    const pathIds = targetNode ? nodePathIds(targetNode) : new Set();
    if (!targetNode) highlightedNodeId = "";

    zoomSvg.selectAll(".gbm-tree-node")
      .classed("gbm-tree-node-highlighted", (item) => pathIds.has(item.data.id))
      .classed("gbm-tree-node-selected", (item) => item.data.id === nodeId);

    zoomSvg.selectAll(".gbm-tree-link")
      .classed("gbm-tree-link-highlighted", (link) => pathIds.has(link.source.data.id) && pathIds.has(link.target.data.id));

    zoomSvg.selectAll(".gbm-tree-edge-label")
      .classed("gbm-tree-edge-label-highlighted", (link) => pathIds.has(link.source.data.id) && pathIds.has(link.target.data.id));
  }

  function bindZoomControls() {
    for (const button of document.querySelectorAll("[data-gbm-tree-zoom]")) {
      button.onclick = () => applyZoom(button.dataset.gbmTreeZoom);
    }
  }

  function bindTreeResizer(root) {
    const resizer = document.getElementById("gbmTreeResizer");
    if (!resizer || resizer.dataset.bound === "true") return;
    resizer.dataset.bound = "true";
    resizer.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      resizePointerId = event.pointerId;
      resizer.classList.add("dragging");
      root.classList.add("resizing");
      resizer.setPointerCapture(event.pointerId);
    });
    resizer.addEventListener("pointermove", (event) => {
      if (resizePointerId !== event.pointerId) return;
      const bounds = root.getBoundingClientRect();
      setSummaryWidth(root, event.clientX - bounds.left);
    });
    const finishDrag = (event) => {
      if (resizePointerId !== event.pointerId) return;
      resizePointerId = null;
      resizer.classList.remove("dragging");
      root.classList.remove("resizing");
      if (resizer.hasPointerCapture(event.pointerId)) {
        resizer.releasePointerCapture(event.pointerId);
      }
      finishSummaryResize();
    };
    resizer.addEventListener("pointerup", finishDrag);
    resizer.addEventListener("pointercancel", finishDrag);
  }

  function setSummaryWidth(root, rawWidth) {
    summaryWidth = clampSummaryWidth(root, rawWidth);
    applySummaryWidth(root, summaryWidth);
  }

  function finishSummaryResize() {
    summaryTable?.redraw(true);
    updateSvgSize();
  }

  function applySummaryWidth(root, width) {
    const nextWidth = clampSummaryWidth(root, width || DEFAULT_SUMMARY_WIDTH);
    root.style.setProperty("--gbm-tree-summary-width", `${Math.round(nextWidth)}px`);
  }

  function clampSummaryWidth(root, rawWidth) {
    const bounds = root.getBoundingClientRect();
    const maxWidth = Math.max(MIN_SUMMARY_WIDTH, bounds.width - MIN_DIAGRAM_WIDTH - 32);
    return Math.min(maxWidth, Math.max(MIN_SUMMARY_WIDTH, Number(rawWidth) || DEFAULT_SUMMARY_WIDTH));
  }

  function applyZoom(action) {
    if (!zoomSvg || !zoomBehavior || !window.d3) return;
    if (action === "in") {
      scaleTree(1.45);
    } else if (action === "out") {
      scaleTree(1 / 1.45);
    } else {
      fitTree(true);
    }
  }

  function scaleTree(scale) {
    const { width, height } = svgSize(chartTarget());
    zoomSvg
      .interrupt()
      .transition()
      .duration(150)
      .ease(window.d3.easeCubicOut)
      .call(zoomBehavior.scaleBy, scale, [width / 2, height / 2]);
  }

  function fitTree(animated = true) {
    if (!zoomSvg || !zoomBehavior || !window.d3) return;
    const transform = fittedTreeTransform();
    if (!transform) return;
    resetTransform = transform;
    zoomSvg.interrupt();
    if (animated) {
      zoomSvg
        .transition()
        .duration(180)
        .ease(window.d3.easeCubicOut)
        .call(zoomBehavior.transform, resetTransform);
    } else {
      zoomSvg.call(zoomBehavior.transform, resetTransform);
    }
  }

  function fittedTreeTransform() {
    if (!treeBounds || !window.d3) return null;
    const { width, height } = svgSize(chartTarget());
    const pad = 44;
    const graphWidth = Math.max(1, treeBounds.yMax - treeBounds.yMin);
    const graphHeight = Math.max(1, treeBounds.xMax - treeBounds.xMin);
    const fitScale = Math.min(
      1.2,
      Math.max(0.03, Math.min((width - pad * 2) / graphWidth, (height - pad * 2) / graphHeight)),
    );
    const tx = pad + (width - pad * 2 - graphWidth * fitScale) / 2 - treeBounds.yMin * fitScale;
    const ty = pad + (height - pad * 2 - graphHeight * fitScale) / 2 - treeBounds.xMin * fitScale;
    return window.d3.zoomIdentity.translate(tx, ty).scale(fitScale);
  }

  function scheduleSvgResize() {
    if (resizeFrame) return;
    resizeFrame = window.requestAnimationFrame(() => {
      resizeFrame = null;
      updateSvgSize();
    });
  }

  function updateSvgSize() {
    if (!zoomSvg) return;
    const { width, height } = svgSize(chartTarget());
    zoomSvg.attr("viewBox", `0 0 ${width} ${height}`);
    resetTransform = fittedTreeTransform();
  }

  function svgSize(target) {
    const bounds = target?.getBoundingClientRect?.();
    return {
      width: Math.max(640, Math.round(bounds?.width || target?.clientWidth || 640)),
      height: Math.max(420, Math.round(bounds?.height || target?.clientHeight || 420)),
    };
  }

  function cancelResizeFrame() {
    if (!resizeFrame) return;
    window.cancelAnimationFrame(resizeFrame);
    resizeFrame = null;
  }

  function bindSearch() {
    const input = document.getElementById("gbmTreeSearch");
    if (!input) return;
    input.oninput = filterSummaryTable;
  }

  function filterSummaryTable() {
    const input = document.getElementById("gbmTreeSearch");
    const query = String(input?.value || "").trim().toLowerCase();
    if (summaryTable) {
      summaryTable.clearFilter(true);
      if (query) {
        summaryTable.setFilter((row) => {
          const values = [row.tree, row.dim, row.features, formatGain(row.gain)];
          return values.some((value) => String(value || "").toLowerCase().includes(query));
        });
      }
      return;
    }
    for (const row of document.querySelectorAll("[data-gbm-tree-row]")) {
      row.hidden = query ? !row.textContent.toLowerCase().includes(query) : false;
    }
  }

  function selectSummaryRow(treeIndex) {
    if (summaryTable && Number.isFinite(Number(treeIndex))) {
      summaryTable.deselectRow();
      summaryTable.selectRow(Number(treeIndex));
      return;
    }
    selectFallbackRow(treeIndex);
  }

  function selectedSummaryRow(treeIndex) {
    return summaryRows.find((row) => Number(row.tree) === Number(treeIndex)) || null;
  }

  function updateTreeDetailSummary(row) {
    const target = document.getElementById("gbmTreeDetailSummary");
    if (!target) return;
    if (!row) {
      target.innerHTML = '<h3 class="gbm-section-title">Tree viewer</h3>';
      return;
    }
    target.innerHTML = `
      <div class="gbm-tree-detail-title">Tree ${escapeHtml(row.tree)}</div>
      <div class="gbm-tree-detail-line"><span>Dimensionality:</span> ${escapeHtml(row.dim)}</div>
      <div class="gbm-tree-detail-line gbm-tree-detail-features"><span>Tree features:</span> ${escapeHtml(row.features || "")}</div>
      <div class="gbm-tree-detail-line"><span>Tree gain:</span> ${escapeHtml(formatGain(row.gain))}</div>
    `;
  }

  function selectFallbackRow(treeIndex) {
    for (const row of document.querySelectorAll("[data-gbm-tree-row]")) {
      row.classList.toggle("active", Number(row.dataset.gbmTreeRow) === Number(treeIndex));
    }
  }

  function clearSummaryTable() {
    if (summaryTable) {
      summaryTable.destroy();
      summaryTable = null;
    }
    document.getElementById("gbmTreeSummaryGrid")?.replaceChildren();
    document.getElementById("gbmTreeSummaryFallback")?.replaceChildren();
  }

  function renderEmpty(message) {
    const target = chartTarget();
    if (!target) return;
    resizeObserver?.disconnect();
    resizeObserver = null;
    cancelResizeFrame();
    zoomBehavior = null;
    zoomSvg = null;
    resetTransform = null;
    treeBounds = null;
    target.innerHTML = `<div class="gbm-tree-empty">${escapeHtml(message)}</div>`;
  }

  function chartTarget() {
    return document.getElementById("gbmTreeSvgMount") || document.getElementById("gbmTreeChart");
  }

  return {
    dispose,
    render,
    refreshTheme,
  };
}

function isEmphasisLabelLine(node, index) {
  if (index === 0) return true;
  return node?.type !== "leaf" && index === 1 && /^Tree \d+$/u.test(String(node?._labelLines?.[0] || ""));
}

function nodePathIds(node) {
  return new Set((node?.ancestors?.() || []).map((item) => item.data.id));
}

function waitForTableBuilt(table) {
  return new Promise((resolve) => {
    let settled = false;
    const finish = () => {
      if (settled) return;
      settled = true;
      resolve();
    };
    table.on("tableBuilt", finish);
    window.setTimeout(finish, 50);
  });
}

function loadD3() {
  if (window.d3) return Promise.resolve(window.d3);
  if (d3Promise) return d3Promise;
  d3Promise = new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = D3_SRC;
    script.onload = () => window.d3 ? resolve(window.d3) : reject(new Error("D3 did not load"));
    script.onerror = () => reject(new Error("D3 did not load"));
    document.head.append(script);
  });
  return d3Promise;
}

function treeFeaturesFormatter(cell) {
  const value = String(cell.getValue() || "");
  const element = document.createElement("span");
  element.className = "gbm-tree-features-cell";
  element.textContent = value;
  element.title = value;
  return element;
}

function treeGainFormatter(cell) {
  return formatGain(cell.getValue());
}

function normaliseLabelLines(label) {
  if (Array.isArray(label)) return label.map((line) => String(line || ""));
  return String(label || "").split("\n").filter(Boolean);
}

function measureNodeWidth(lines) {
  const maxChars = Math.max(6, ...lines.map((line) => String(line || "").length));
  return Math.max(NODE_MIN_WIDTH, Math.min(NODE_MAX_WIDTH, maxChars * 7 + NODE_HORIZONTAL_PADDING * 2));
}

function edgeLabelLines(label) {
  const text = String(label || "").replace(/\s+/g, " ").trim();
  if (!text) return [];
  if (isCategoricalEdgeLabel(text)) return wrapDelimitedLabel(text, " / ", CATEGORICAL_EDGE_LABEL_WRAP_CHARS);
  return wrapWordLabel(text, EDGE_LABEL_WRAP_CHARS);
}

function edgeLabelPlacement(link) {
  const label = String(link.target.data.edge_label || "");
  const lines = edgeLabelLines(label);
  const categorical = isCategoricalEdgeLabel(label);
  const position = categorical ? CATEGORICAL_EDGE_LABEL_POSITION : 0.5;
  const x = link.source.y + (link.target.y - link.source.y) * position + (categorical ? CATEGORICAL_EDGE_LABEL_X_OFFSET : 0);
  const y = link.source.x + (link.target.x - link.source.x) * 0.5 - (lines.length > 1 ? 14 : 6);
  return { x, y };
}

function isCategoricalEdgeLabel(label) {
  return String(label || "").includes(" / ");
}

function wrapDelimitedLabel(text, delimiter, maxChars) {
  const parts = String(text || "").split(delimiter).map((part) => part.trim()).filter(Boolean);
  const lines = [];
  let current = "";
  for (const part of parts) {
    const candidate = current ? `${current}${delimiter}${part}` : part;
    if (candidate.length <= maxChars || !current) {
      current = candidate;
    } else {
      lines.push(current);
      current = part;
    }
  }
  if (current) lines.push(current);
  return lines.flatMap((line) => line.length > maxChars ? wrapWordLabel(line, maxChars) : [line]);
}

function wrapWordLabel(text, maxChars) {
  const words = String(text || "").split(/\s+/u).filter(Boolean);
  const lines = [];
  let current = "";
  for (const word of words) {
    const candidate = current ? `${current} ${word}` : word;
    if (candidate.length <= maxChars || !current) {
      current = candidate;
    } else {
      lines.push(current);
      current = word;
    }
  }
  if (current) lines.push(current);
  return lines;
}

function elbowPath(source, target) {
  const midY = source.y + (target.y - source.y) * 0.55;
  return `M${source.y},${source.x}C${midY},${source.x} ${midY},${target.x} ${target.y},${target.x}`;
}

function nodeFill(node, values, selectedPalette) {
  if (selectedPalette === "plain") {
    return node.type === "leaf"
      ? cssVar("--gbm-tree-leaf-fill", "#fef3c7")
      : cssVar("--gbm-tree-split-fill", "#dbeafe");
  }
  const scale = quantileColor(values, TREE_PALETTES[selectedPalette] || TREE_PALETTES.viridis);
  return scale(node.value);
}

function readableTextColor(fill) {
  const rgb = hexToRgb(fill);
  if (!rgb) return "#111827";
  const darkContrast = contrastRatio(rgb, { r: 17, g: 24, b: 39 });
  const lightContrast = contrastRatio(rgb, { r: 255, g: 255, b: 255 });
  return darkContrast >= lightContrast ? "#111827" : "#ffffff";
}

function contrastRatio(left, right) {
  const leftLum = relativeLuminance(left);
  const rightLum = relativeLuminance(right);
  const lighter = Math.max(leftLum, rightLum);
  const darker = Math.min(leftLum, rightLum);
  return (lighter + 0.05) / (darker + 0.05);
}

function relativeLuminance({ r, g, b }) {
  const [red, green, blue] = [r, g, b].map((value) => {
    const channel = value / 255;
    return channel <= 0.03928 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * red + 0.7152 * green + 0.0722 * blue;
}

function hexToRgb(color) {
  const match = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(String(color || "").trim());
  if (!match) return null;
  return {
    r: Number.parseInt(match[1], 16),
    g: Number.parseInt(match[2], 16),
    b: Number.parseInt(match[3], 16),
  };
}

function quantileColor(values, colors) {
  const numbers = values.map(Number).filter(Number.isFinite).sort((left, right) => left - right);
  if (!numbers.length) return () => colors[Math.floor(colors.length / 2)];
  const thresholds = [];
  for (let index = 1; index < colors.length; index += 1) {
    thresholds.push(numbers[Math.min(numbers.length - 1, Math.max(0, Math.floor((index / colors.length) * numbers.length)))]);
  }
  return (value) => {
    const number = Number(value);
    if (!Number.isFinite(number)) return colors[Math.floor(colors.length / 2)];
    let bucket = 0;
    while (bucket < thresholds.length && number > thresholds[bucket]) bucket += 1;
    return colors[Math.min(colors.length - 1, bucket)];
  };
}

function formatGain(value) {
  const number = Number(value || 0);
  if (!Number.isFinite(number)) return "0";
  return Math.round(number).toLocaleString();
}

function cssVar(name, fallback) {
  return getComputedStyle(document.body).getPropertyValue(name).trim() || fallback;
}
