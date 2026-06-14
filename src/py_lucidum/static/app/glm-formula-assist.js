const FORMULA_NAME_RE = /^[A-Za-z_][A-Za-z0-9_]*$/;

export const GLM_FORMULA_FUNCTIONS = [
  { caption: "ifelse", value: "ifelse()", meta: "function" },
  { caption: "pmin", value: "pmin()", meta: "function" },
  { caption: "pmax", value: "pmax()", meta: "function" },
  { caption: "np.isin", value: "np.isin()", meta: "function" },
  { caption: "C", value: "C()", meta: "factor" },
  { caption: "bs", value: "bs()", meta: "spline" },
  { caption: "ns", value: "ns()", meta: "spline" },
  { caption: "cs", value: "cs()", meta: "spline" },
  { caption: "poly", value: "poly()", meta: "transform" },
  { caption: "log", value: "log()", meta: "transform" },
  { caption: "log1p", value: "log1p()", meta: "transform" },
  { caption: "exp", value: "exp()", meta: "transform" },
  { caption: "sqrt", value: "sqrt()", meta: "transform" },
  { caption: "abs", value: "abs()", meta: "transform" },
  { caption: "offset", value: "offset()", meta: "offset" },
];

export const GLM_FORMULA_SNIPPETS = [
  { id: "identity", label: "Identity", group: "Basic" },
  { id: "log", label: "log(feature)", group: "Transforms" },
  { id: "log1p", label: "log1p(feature)", group: "Transforms" },
  { id: "sqrt", label: "sqrt(feature)", group: "Transforms" },
  { id: "abs", label: "abs(feature)", group: "Transforms" },
  { id: "poly2", label: "poly degree 2", group: "Transforms" },
  { id: "bs4", label: "bs df 4", group: "Splines" },
  { id: "ns4", label: "ns df 4", group: "Splines" },
  { id: "cs4", label: "cs df 4", group: "Splines" },
  { id: "lower_hinge", label: "pmax(0, feature)", group: "Caps" },
  { id: "upper_cap", label: "pmin(upper, feature)", group: "Caps" },
  { id: "clamp", label: "pmax(lower, pmin(upper, feature))", group: "Caps" },
  { id: "positive_offset", label: "offset(log(pmax(weight, 1)))", group: "Offsets" },
];

export function quoteFormulaName(name) {
  const text = String(name || "").trim();
  if (!text) return "";
  if (FORMULA_NAME_RE.test(text)) return text;
  return `\`${text.replace(/\\/g, "\\\\").replace(/`/g, "\\`")}\``;
}

export function unquoteFormulaName(name) {
  const text = String(name || "").trim();
  if (text.length >= 2 && text[0] === "`" && text[text.length - 1] === "`") {
    return text.slice(1, -1).replace(/\\`/g, "`").replace(/\\\\/g, "\\");
  }
  return text;
}

export function formulaStringLiteral(value) {
  return `"${String(value ?? "")
    .replace(/\\/g, "\\\\")
    .replace(/"/g, "\\\"")
    .replace(/\n/g, "\\n")
    .replace(/\r/g, "\\r")
    .replace(/\t/g, "\\t")}"`;
}

export function formulaLiteral(value) {
  if (typeof value === "boolean") return value ? "True" : "False";
  if (typeof value === "number" && Number.isFinite(value)) return String(value);
  return formulaStringLiteral(value);
}

export function buildGroupedLevelsFormula(feature, levels = []) {
  const name = quoteFormulaName(feature);
  const values = levels.map((level) => formulaLiteral(level)).join(", ");
  return `ifelse(np.isin(${name}, [${values}]), 1, 0)`;
}

export function buildIndividualLevelsFormula(feature, levels = []) {
  const name = quoteFormulaName(feature);
  return levels
    .map((level) => `+ ifelse(${name} == ${formulaLiteral(level)}, 1, 0)`)
    .join("\n");
}

export function parseBreakpoints(text) {
  const tokens = String(text || "").split(/[,\s]+/).map((token) => token.trim()).filter(Boolean);
  if (!tokens.length) return { values: [], error: "Enter at least one breakpoint" };
  const values = tokens.map((token) => Number(token));
  if (values.some((value) => !Number.isFinite(value))) return { values: [], error: "Breakpoints must be numeric" };
  for (let index = 1; index < values.length; index += 1) {
    if (values[index] <= values[index - 1]) return { values: [], error: "Breakpoints must increase" };
  }
  return { values, error: "" };
}

export function buildPiecewiseFormula(feature, breakpoints = []) {
  const name = quoteFormulaName(feature);
  const breaks = breakpoints.map((value) => Number(value)).filter((value) => Number.isFinite(value));
  if (!name || !breaks.length) return "";
  const terms = [`pmin(${formatFormulaNumber(breaks[0])}, ${name})`];
  for (let index = 1; index < breaks.length; index += 1) {
    terms.push(`pmax(${formatFormulaNumber(breaks[index - 1])}, pmin(${formatFormulaNumber(breaks[index])}, ${name}))`);
  }
  terms.push(`pmax(${formatFormulaNumber(breaks[breaks.length - 1])}, ${name})`);
  return terms.map((term) => `+ ${term}`).join("\n");
}

export function withFormulaHeader(text, feature, includeHeader = false) {
  const value = String(text || "").trimEnd();
  if (!includeHeader || !value) return value;
  return `# ${String(feature || "").trim()}\n${value}`;
}

export function formatDrawerInsertion(text, beforeText = "", options = {}) {
  const value = String(text || "").trimEnd();
  if (!value) return "";
  if (options.replaceSelection) return value;
  const before = String(beforeText || "");
  const startsWithPlus = formulaBodyStartsWithPlus(value);
  const needsPlus = before.trim().length > 0 && !startsWithPlus && !/[+~(,]\s*$/.test(before);
  return `${needsPlus ? prefixFirstFormulaLine(value) : value}\n`;
}

function formulaBodyStartsWithPlus(text) {
  const line = firstFormulaLine(text);
  return /^\s*\+/.test(line || "");
}

function prefixFirstFormulaLine(text) {
  const lines = String(text || "").split("\n");
  const index = lines.findIndex((line) => {
    const trimmed = line.trim();
    return trimmed && !trimmed.startsWith("#");
  });
  if (index < 0) return `+ ${String(text || "")}`;
  lines[index] = `+ ${lines[index]}`;
  return lines.join("\n");
}

function firstFormulaLine(text) {
  return String(text || "").split("\n").find((line) => {
    const trimmed = line.trim();
    return trimmed && !trimmed.startsWith("#");
  }) || "";
}

export function buildSnippetFormula(snippetId, feature, options = {}) {
  const name = quoteFormulaName(feature);
  const second = quoteFormulaName(options.secondaryFeature || "");
  const denominator = quoteFormulaName(options.denominator || feature);
  if (!name && snippetId !== "positive_offset") return "";
  switch (snippetId) {
    case "identity":
      return name;
    case "factor":
      return `C(${name})`;
    case "interaction":
      return second ? `${name}:${second}` : `${name}:`;
    case "log":
      return `log(${name})`;
    case "log1p":
      return `log1p(${name})`;
    case "sqrt":
      return `sqrt(${name})`;
    case "abs":
      return `abs(${name})`;
    case "poly2":
      return `poly(${name}, degree=2)`;
    case "bs4":
      return `bs(${name}, df=4)`;
    case "ns4":
      return `ns(${name}, df=4, constraints="center")`;
    case "cs4":
      return `cs(${name}, df=4)`;
    case "lower_hinge":
      return `pmax(0, ${name})`;
    case "upper_cap":
      return `pmin(upper_bound, ${name})`;
    case "clamp":
      return `pmax(lower_bound, pmin(upper_bound, ${name}))`;
    case "positive_offset":
      return denominator ? `offset(log(pmax(${denominator}, 1)))` : "";
    default:
      return "";
  }
}

export function rankFormulaSuggestions(items = [], prefix = "") {
  const needle = String(prefix || "").toLowerCase();
  return items
    .map((item, index) => {
      const caption = String(item.caption || item.name || item.value || "");
      const lower = caption.toLowerCase();
      let rank = 0;
      if (needle) {
        if (lower === needle) rank = 100;
        else if (lower.startsWith(needle)) rank = 80;
        else if (lower.includes(needle)) rank = 40;
        else rank = -1;
      }
      return { ...item, rank, originalIndex: index };
    })
    .filter((item) => item.rank >= 0)
    .sort((a, b) => (b.rank - a.rank) || String(a.caption || "").localeCompare(String(b.caption || "")) || (a.originalIndex - b.originalIndex));
}

export function formulaCompletionContext(text, row, column) {
  const lines = String(text || "").split("\n");
  const line = lines[row] || "";
  const beforeLine = line.slice(0, column);
  if (isCommentOrStringContext(beforeLine)) return { type: "none", prefix: "", replaceStartColumn: column };
  const before = [...lines.slice(0, row), beforeLine].join("\n");
  const level = levelCompletionContext(before, beforeLine, column);
  if (level) return level;
  const prefixMatch = beforeLine.match(/[A-Za-z_][A-Za-z0-9_]*$/);
  const prefix = prefixMatch ? prefixMatch[0] : "";
  return { type: "formula", prefix, replaceStartColumn: column - prefix.length };
}

export function formulaColumnSuggestions(columns = []) {
  return columns.map((column) => ({
    type: "column",
    caption: String(column.name || ""),
    value: quoteFormulaName(column.name),
    meta: String(column.kind || "column"),
  }));
}

export function formulaFunctionSuggestions() {
  return GLM_FORMULA_FUNCTIONS.map((item) => ({ ...item, type: "function" }));
}

function levelCompletionContext(before, beforeLine, column) {
  const match = before.match(/np\.isin\s*\(\s*(`(?:\\`|[^`])+`|[A-Za-z_][A-Za-z0-9_]*)\s*,\s*\[([^\]\)]*)$/s);
  if (!match) return null;
  const token = levelToken(beforeLine);
  return {
    type: "levels",
    feature: unquoteFormulaName(match[1]),
    prefix: token.prefix,
    replaceStartColumn: token.startColumn,
  };
}

function levelToken(beforeLine) {
  const lastDelimiter = Math.max(beforeLine.lastIndexOf("["), beforeLine.lastIndexOf(","));
  const rawStart = lastDelimiter >= 0 ? lastDelimiter + 1 : beforeLine.length;
  const raw = beforeLine.slice(rawStart);
  const leading = raw.match(/^\s*/)?.[0] || "";
  const tokenStart = rawStart + leading.length;
  const token = beforeLine.slice(tokenStart);
  const quoted = token.match(/^["']([^"']*)$/);
  if (quoted) return { prefix: quoted[1], startColumn: tokenStart };
  const bare = token.match(/([^\s,\[]*)$/)?.[1] || "";
  return { prefix: bare, startColumn: beforeLine.length - bare.length };
}

function isCommentOrStringContext(beforeLine) {
  let quote = "";
  let escaped = false;
  for (const char of String(beforeLine || "")) {
    if (escaped) {
      escaped = false;
      continue;
    }
    if (quote && char === "\\") {
      escaped = true;
      continue;
    }
    if (char === "\"" || char === "'" || char === "`") {
      if (quote === char) quote = "";
      else if (!quote) quote = char;
      continue;
    }
    if (char === "#" && !quote) return true;
  }
  return Boolean(quote);
}

function formatFormulaNumber(value) {
  const number = Number(value);
  return Number.isInteger(number) ? String(number) : String(number);
}
