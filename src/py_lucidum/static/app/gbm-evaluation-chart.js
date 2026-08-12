import { gbmEvaluationChartOption } from "./gbm-evaluation-chart-options.js";
import { observeResize } from "./shared/model-ui.js";

export function createGbmEvaluationChart({ escapeHtml, formatEvaluationValue, showClipboardToast = () => {} }) {
  let chart = null;
  let resizeObserver = null;
  let viewMode = "all";

  function render(source = null) {
    const target = document.getElementById("gbmEvaluationChart");
    if (!target || !window.echarts) return;
    const option = gbmEvaluationChartOption(source || {}, {
      viewMode,
      escapeHtml,
      formatValue: formatEvaluationValue,
      colors: evaluationChartColors(),
    });
    if (!option) return;
    if (!chart) {
      chart = window.echarts.init(target);
      bindResize(target);
    }
    chart.setOption(option, true);
    requestAnimationFrame(() => chart?.resize());
  }

  function setViewMode(mode) {
    viewMode = mode === "tail" ? "tail" : "all";
  }

  function getViewMode() {
    return viewMode;
  }

  async function copyToClipboard() {
    if (!chart || !navigator.clipboard?.write || typeof window.ClipboardItem !== "function") {
      showClipboardToast("Could not copy Evaluation Log chart image", true);
      return;
    }
    try {
      const dataUrl = chart.getDataURL({
        type: "png",
        pixelRatio: 2,
        backgroundColor: cssVar("--panel", "#ffffff"),
      });
      const blob = await fetch(dataUrl).then((response) => response.blob());
      await navigator.clipboard.write([new window.ClipboardItem({ "image/png": blob })]);
      showClipboardToast("Evaluation Log chart image copied");
    } catch (_) {
      showClipboardToast("Could not copy Evaluation Log chart image", true);
    }
  }

  function dispose() {
    resizeObserver?.disconnect();
    resizeObserver = null;
    if (chart) {
      chart.dispose();
      chart = null;
    }
  }

  function bindResize(target) {
    resizeObserver = observeResize([target, target.parentElement], () => chart?.resize());
  }

  return {
    copyToClipboard,
    dispose,
    getViewMode,
    render,
    setViewMode,
  };
}

function evaluationChartColors() {
  return {
    text: cssVar("--text", "#3f3f46"),
    muted: cssVar("--muted", "#4b5563"),
    line: cssVar("--line", "#e5e7eb"),
    panel: cssVar("--panel", "#ffffff"),
    actual: cssVar("--actual-line", "#050505"),
  };
}

function cssVar(name, fallback) {
  return getComputedStyle(document.body).getPropertyValue(name).trim() || fallback;
}
