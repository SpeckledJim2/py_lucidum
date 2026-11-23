const packageLabels = {
  duckdb: "DuckDB",
  fastapi: "FastAPI",
  uvicorn: "Uvicorn",
  lightgbm: "LightGBM",
  numpy: "NumPy",
  polars: "Polars",
  pyarrow: "PyArrow",
};

export function bindSidebarVersions(button, panel) {
  let pinned = false;
  let closeTimer;
  const isOpen = () => panel.matches(":popover-open");

  function position() {
    if (!isOpen()) return;
    const anchor = button.getBoundingClientRect();
    const gap = 6;
    const fitsBeside = anchor.right + gap + panel.offsetWidth <= window.innerWidth - gap;
    const bottom = fitsBeside ? anchor.bottom : anchor.top - gap;
    panel.style.maxHeight = `${Math.max(0, Math.min(bottom, window.innerHeight - gap) - gap)}px`;
    const left = fitsBeside ? anchor.right + gap : anchor.left + (anchor.width - panel.offsetWidth) / 2;
    panel.style.left = `${Math.max(gap, Math.min(left, window.innerWidth - panel.offsetWidth - gap))}px`;
    panel.style.top = `${Math.max(gap, Math.min(bottom, window.innerHeight - gap) - panel.offsetHeight)}px`;
  }

  function open() {
    clearTimeout(closeTimer);
    if (isOpen()) return;
    panel.showPopover();
    button.setAttribute("aria-expanded", "true");
    position();
  }

  function close() {
    clearTimeout(closeTimer);
    pinned = false;
    panel.hidePopover();
    button.setAttribute("aria-expanded", "false");
  }

  button.addEventListener("pointerenter", (event) => {
    if (event.pointerType === "mouse") open();
  });
  panel.addEventListener("pointerenter", () => clearTimeout(closeTimer));
  for (const element of [button, panel]) {
    element.addEventListener("pointerleave", (event) => {
      if (event.pointerType === "mouse" && !pinned) closeTimer = setTimeout(close, 250);
    });
  }
  button.addEventListener("click", () => {
    if (pinned) return close();
    pinned = true;
    open();
  });
  function dismissOutside(event) {
    if (isOpen() && !button.contains(event.target) && !panel.contains(event.target)) close();
  }
  document.addEventListener("pointerdown", dismissOutside);
  document.addEventListener("focusin", dismissOutside);
  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape" || !isOpen()) return;
    const restoreFocus = pinned || panel.contains(document.activeElement);
    close();
    if (restoreFocus) button.focus();
  });
  window.addEventListener("resize", position);

  function render(packages) {
    const list = panel.querySelector("dl");
    list.replaceChildren();
    for (const [name, version] of Object.entries(packages || {})) {
      const term = document.createElement("dt");
      term.textContent = packageLabels[name] || name;
      const value = document.createElement("dd");
      value.textContent = version;
      list.append(term, value);
    }
    position();
  }

  return { close, render };
}
