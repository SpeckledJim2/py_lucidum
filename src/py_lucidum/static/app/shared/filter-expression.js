function normalisedRows(rows) {
  return (Array.isArray(rows) ? rows : [])
    .map((row) => ({
      theme: String(row?.theme || "General"),
      expression: String(row?.expression || "").trim(),
    }))
    .filter((row) => row.expression);
}

function groupedExpressions(rows) {
  const groups = [];
  const byTheme = new Map();
  normalisedRows(rows).forEach((row) => {
    if (!byTheme.has(row.theme)) {
      const expressions = [];
      byTheme.set(row.theme, expressions);
      groups.push(expressions);
    }
    byTheme.get(row.theme).push(row.expression);
  });
  return groups;
}

function identifierCharacter(character) {
  return Boolean(character && /[A-Za-z0-9_$]/.test(character));
}

function dollarQuoteTag(expression, index) {
  const match = expression.slice(index).match(/^\$(?:[A-Za-z_][A-Za-z0-9_]*)?\$/);
  return match?.[0] || "";
}

export function filterExpressionHasTopLevelOr(expression) {
  const text = String(expression || "");
  let depth = 0;
  let quote = "";
  let quoteBackslashEscapes = false;
  let dollarTag = "";
  let lineComment = false;
  let blockComment = false;

  for (let index = 0; index < text.length; index += 1) {
    const character = text[index];
    const next = text[index + 1] || "";

    if (lineComment) {
      if (character === "\n" || character === "\r") lineComment = false;
      continue;
    }
    if (blockComment) {
      if (character === "*" && next === "/") {
        blockComment = false;
        index += 1;
      }
      continue;
    }
    if (dollarTag) {
      if (text.startsWith(dollarTag, index)) {
        index += dollarTag.length - 1;
        dollarTag = "";
      }
      continue;
    }
    if (quote) {
      if (quoteBackslashEscapes && character === "\\" && next) {
        index += 1;
        continue;
      }
      if (character !== quote) continue;
      if (next === quote) {
        index += 1;
        continue;
      }
      quote = "";
      quoteBackslashEscapes = false;
      continue;
    }

    if (character === "'" || character === '"') {
      quote = character;
      quoteBackslashEscapes = character === "'"
        && (text[index - 1] === "E" || text[index - 1] === "e")
        && !identifierCharacter(text[index - 2]);
      continue;
    }
    if (character === "$") {
      const tag = dollarQuoteTag(text, index);
      if (tag) {
        dollarTag = tag;
        index += tag.length - 1;
        continue;
      }
    }
    if (character === "-" && next === "-") {
      lineComment = true;
      index += 1;
      continue;
    }
    if (character === "/" && next === "*") {
      blockComment = true;
      index += 1;
      continue;
    }
    if (character === "(") {
      depth += 1;
      continue;
    }
    if (character === ")") {
      depth = Math.max(0, depth - 1);
      continue;
    }
    if (
      depth === 0
      && text.slice(index, index + 2).toUpperCase() === "OR"
      && !identifierCharacter(text[index - 1])
      && !identifierCharacter(text[index + 2])
    ) {
      return true;
    }
  }
  return false;
}

export function combineGroupedFilterRows(rows) {
  const groups = groupedExpressions(rows);
  if (!groups.length) return "";
  if (groups.length === 1) return groups[0].join(" OR ");
  return groups
    .map((expressions) => {
      if (expressions.length > 1) return `(${expressions.join(" OR ")})`;
      const expression = expressions[0];
      return filterExpressionHasTopLevelOr(expression) ? `(${expression})` : expression;
    })
    .join(" AND ");
}

export function combineLegacyGroupedFilterRows(rows) {
  const groups = groupedExpressions(rows);
  if (!groups.length) return "";
  if (groups.length === 1 && groups[0].length === 1) return groups[0][0];
  return groups
    .map((expressions) => {
      const grouped = expressions.map((expression) => `(${expression})`).join(" OR ");
      return expressions.length > 1 ? `(${grouped})` : grouped;
    })
    .join(" AND ");
}

function combineFlatFilterRows(rows, operator) {
  const expressions = normalisedRows(rows).map((row) => row.expression);
  if (!expressions.length) return "";
  const sqlOperator = operator === "or" || operator === "nor" ? "OR" : "AND";
  const grouped = expressions.length > 1
    ? expressions.map((expression) => `(${expression})`)
    : expressions;
  const combined = grouped.join(` ${sqlOperator} `);
  return operator === "nand" || operator === "nor" ? `NOT (${combined})` : combined;
}

export function combineSavedFilterRows(rows, { mode = "grouped", operator = "and" } = {}) {
  return mode === "grouped"
    ? combineGroupedFilterRows(rows)
    : combineFlatFilterRows(rows, operator);
}

export function normaliseLegacyGroupedFilter(expression, rows, { allRowsRestored = true } = {}) {
  const current = String(expression || "").trim();
  if (!allRowsRestored || current !== combineLegacyGroupedFilterRows(rows)) return current;
  return combineGroupedFilterRows(rows);
}
