const ICON_PATHS = {
  features: `
    <path d="M4 7h10"></path><path d="M18 7h2"></path><circle cx="16" cy="7" r="2"></circle>
    <path d="M4 17h2"></path><path d="M10 17h10"></path><circle cx="8" cy="17" r="2"></circle>`,
  models: `
    <path d="M4 7.5 12 4l8 3.5-8 3.5-8-3.5Z"></path>
    <path d="m4 12.5 8 3.5 8-3.5"></path><path d="m4 17 8 3 8-3"></path>`,
  shap: `
    <path d="m12 3 1.35 4.15L17.5 8.5l-4.15 1.35L12 14l-1.35-4.15L6.5 8.5l4.15-1.35L12 3Z"></path>
    <path d="m18.5 14 .65 1.85L21 16.5l-1.85.65L18.5 19l-.65-1.85L16 16.5l1.85-.65.65-1.85Z"></path>`,
  layers: `
    <path d="m12 3 9 5-9 5-9-5 9-5Z"></path><path d="m3 12 9 5 9-5"></path><path d="m3 16 9 5 9-5"></path>`,
  tree: `
    <path d="M12 5v5"></path><path d="M6 14v-2a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2v2"></path>
    <rect x="9" y="2" width="6" height="4" rx="1"></rect><rect x="3" y="14" width="6" height="5" rx="1"></rect><rect x="15" y="14" width="6" height="5" rx="1"></rect>`,
  formula: `
    <path d="M5 19c3 0 3-14 7-14 1.4 0 2.2.8 2.2 2"></path><path d="M7 11h7"></path>
    <path d="m16 13 4 6"></path><path d="m20 13-4 6"></path>`,
  table: `
    <rect x="3" y="4" width="18" height="16" rx="2"></rect><path d="M3 9h18M3 14h18M9 4v16M15 4v16"></path>`,
  kpi: `
    <path d="M5 18a8 8 0 1 1 14 0"></path><path d="m12 14 4-4"></path><circle cx="12" cy="14" r="1.5"></circle>`,
  filter: `
    <path d="M3 5h18l-7 8v5l-4 2v-7L3 5Z"></path>`,
};

function escapeAttribute(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll('"', "&quot;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

export function toolScreenNavButtonHtml({
  active = false,
  buttonId,
  controlsId,
  icon,
  label,
  targetId,
  toolDataAttribute,
}) {
  const safeLabel = escapeAttribute(label);
  const safeTargetId = escapeAttribute(targetId);
  const iconPaths = ICON_PATHS[icon] || ICON_PATHS.table;
  return `
    <button id="${escapeAttribute(buttonId)}" class="tool-screen-nav-item${active ? " active" : ""}" type="button"
      role="tab" aria-selected="${active}" aria-controls="${escapeAttribute(controlsId)}"
      aria-label="${safeLabel}" title="${safeLabel}" tabindex="${active ? "0" : "-1"}"
      data-screen-nav-id="${safeTargetId}" data-${toolDataAttribute}="${safeTargetId}">
      <svg class="tool-screen-nav-icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false">${iconPaths}</svg>
      <span class="tool-screen-nav-label">${safeLabel}</span>
    </button>`;
}

export function bindToolScreenNavigation(nav, onSelect) {
  if (!nav) return;
  const root = nav.getRootNode();
  const buttons = () => [...nav.querySelectorAll("[data-screen-nav-id]")];
  for (const button of buttons()) {
    button.addEventListener("click", () => onSelect(button.dataset.screenNavId));
    button.addEventListener("keydown", (event) => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
      const items = buttons();
      if (!items.length) return;
      const currentIndex = Math.max(0, items.indexOf(button));
      let nextIndex = currentIndex;
      if (event.key === "Home") nextIndex = 0;
      if (event.key === "End") nextIndex = items.length - 1;
      if (event.key === "ArrowLeft") nextIndex = (currentIndex - 1 + items.length) % items.length;
      if (event.key === "ArrowRight") nextIndex = (currentIndex + 1) % items.length;
      event.preventDefault();
      const nextId = items[nextIndex].dataset.screenNavId;
      const nextButtonId = items[nextIndex].id;
      onSelect(nextId);
      queueMicrotask(() => {
        const nextButton = root.getElementById?.(nextButtonId)
          || [...root.querySelectorAll("[data-screen-nav-id]")].find((item) => item.id === nextButtonId);
        nextButton?.focus();
      });
    });
  }
}

export function syncToolScreenNavigation(nav, activeId) {
  if (!nav) return;
  for (const button of nav.querySelectorAll("[data-screen-nav-id]")) {
    const active = button.dataset.screenNavId === activeId;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
    button.tabIndex = active ? 0 : -1;
  }
}
