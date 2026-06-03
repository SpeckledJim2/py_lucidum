const TABULATOR_CSS_HREF = "/static/vendor/tabulator/tabulator.min.css";
const TABULATOR_SCRIPT_SRC = "/static/vendor/tabulator/tabulator.min.js";

let tabulatorPromise = null;

function ensureTabulatorStylesheet() {
  if (document.querySelector(`link[href="${TABULATOR_CSS_HREF}"]`)) return;
  if ([...document.styleSheets].some((sheet) => sheet.href?.endsWith(TABULATOR_CSS_HREF))) return;
  const link = document.createElement("link");
  link.rel = "stylesheet";
  link.href = TABULATOR_CSS_HREF;
  document.head.append(link);
}

export function loadTabulator() {
  if (window.Tabulator) return Promise.resolve(window.Tabulator);
  if (tabulatorPromise) return tabulatorPromise;
  tabulatorPromise = new Promise((resolve, reject) => {
    ensureTabulatorStylesheet();
    const resolveLoaded = () => window.Tabulator ? resolve(window.Tabulator) : reject(new Error("Tabulator did not load"));
    const failLoad = () => reject(new Error("Tabulator did not load"));
    const existing = document.querySelector(`script[src="${TABULATOR_SCRIPT_SRC}"]`);
    if (existing) {
      existing.addEventListener("load", resolveLoaded, { once: true });
      existing.addEventListener("error", failLoad, { once: true });
      if (existing.dataset.loaded === "true") resolveLoaded();
      return;
    }
    const script = document.createElement("script");
    script.src = TABULATOR_SCRIPT_SRC;
    script.async = true;
    script.dataset.loaded = "false";
    script.addEventListener("load", () => {
      script.dataset.loaded = "true";
      resolveLoaded();
    }, { once: true });
    script.addEventListener("error", failLoad, { once: true });
    document.head.append(script);
  });
  return tabulatorPromise;
}
