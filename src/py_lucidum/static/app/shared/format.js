export function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" })[char]);
}

export function createFormatters({ getActiveKpiFormat }) {
  function formatNumber(value) {
    if (value === null || value === undefined || Number.isNaN(value)) return "";
    const number = Number(value);
    if (!Number.isFinite(number)) return "";
    const abs = Math.abs(number);
    let maximumFractionDigits = 0;
    if (abs !== 0 && abs < 0.01) maximumFractionDigits = 6;
    else if (abs < 1) maximumFractionDigits = 4;
    else if (abs < 10) maximumFractionDigits = 3;
    else if (abs < 1000) maximumFractionDigits = 2;
    else maximumFractionDigits = 1;
    return number.toLocaleString(undefined, {
      minimumFractionDigits: 0,
      maximumFractionDigits,
    });
  }

  function formatChartLabel(params) {
    const value = Array.isArray(params.value) ? params.value[1] : params.value;
    return formatNumber(value);
  }

  function formatLineLabel(params) {
    const value = Array.isArray(params.value) ? params.value[1] : params.value;
    return formatLineValue(value);
  }

  function formatLineValueForFormat(value, activeKpiFormat = null) {
    if (value === null || value === undefined || Number.isNaN(value)) return "";
    const number = Number(value);
    if (!Number.isFinite(number)) return "";
    if (activeKpiFormat) {
      const decimals = Number(activeKpiFormat.decimals);
      const fractionDigits = Number.isInteger(decimals) ? Math.max(0, Math.min(12, decimals)) : 2;
      const displayNumber = activeKpiFormat.format === "percent" ? number * 100 : number;
      const formatted = Math.abs(displayNumber).toLocaleString(undefined, {
        minimumFractionDigits: fractionDigits,
        maximumFractionDigits: fractionDigits,
      });
      const sign = displayNumber < 0 ? "-" : "";
      if (activeKpiFormat.format === "currency") return `${sign}£${formatted}`;
      const signed = `${sign}${formatted}`;
      if (activeKpiFormat.format === "percent") return `${signed}%`;
      return signed;
    }
    const abs = Math.abs(number);
    let fractionDigits = 2;
    if (abs !== 0 && abs < 0.01) fractionDigits = 6;
    else if (abs < 1) fractionDigits = 4;
    return number.toLocaleString(undefined, {
      minimumFractionDigits: fractionDigits,
      maximumFractionDigits: fractionDigits,
    });
  }

  function formatLineValue(value) {
    return formatLineValueForFormat(value, getActiveKpiFormat?.());
  }

  function formatWeightValue(value) {
    if (value === null || value === undefined || Number.isNaN(value)) return "";
    const number = Number(value);
    if (!Number.isFinite(number)) return "";
    const abs = Math.abs(number);
    if (abs >= 10 || Number.isInteger(number)) {
      return number.toLocaleString(undefined, {
        minimumFractionDigits: 0,
        maximumFractionDigits: 0,
      });
    }
    return formatNumber(number);
  }

  function formatFileSize(value) {
    const bytes = Number(value);
    if (!Number.isFinite(bytes) || bytes < 0) return "";
    let divisor = 1024;
    let suffix = "Kb";
    if (bytes >= 1024 ** 3) {
      divisor = 1024 ** 3;
      suffix = "Gb";
    } else if (bytes >= 1024 ** 2) {
      divisor = 1024 ** 2;
      suffix = "Mb";
    }
    const size = bytes / divisor;
    return `${(bytes > 0 ? Math.max(0.1, size) : 0).toFixed(1)}${suffix}`;
  }

  function formatXLabel(value, kind) {
    if (kind === "numeric") return formatNumericXLabel(value);
    if (kind !== "integer") return String(value);
    const number = Number(value);
    if (!Number.isFinite(number) || !Number.isInteger(number)) return String(value);
    return number.toLocaleString(undefined, { maximumFractionDigits: 0 });
  }

  function formatNumericXLabel(value) {
    const text = String(value);
    const number = Number(text);
    if (!Number.isFinite(number)) return text;
    return number.toLocaleString(undefined, { maximumFractionDigits: 12 });
  }

  function formatRowMeta(rowCount, filteredRowCount = rowCount) {
    const total = Number(rowCount);
    if (!Number.isFinite(total)) return "";
    const filtered = Number(filteredRowCount ?? total);
    const shown = Number.isFinite(filtered) ? filtered : total;
    return shown === total
      ? `${total.toLocaleString()} rows`
      : `${shown.toLocaleString()} / ${total.toLocaleString()} rows`;
  }

  return {
    formatNumber,
    formatChartLabel,
    formatLineLabel,
    formatLineValue,
    formatLineValueForFormat,
    formatWeightValue,
    formatFileSize,
    formatXLabel,
    formatNumericXLabel,
    formatRowMeta,
  };
}
