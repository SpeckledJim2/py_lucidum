const GBM_TABS = [
  { id: "features", label: "Features and parameters" },
  { id: "models", label: "Model navigator" },
  { id: "shap", label: "SHAP" },
  { id: "stacked-shap", label: "Stacked SHAP" },
  { id: "trees", label: "Tree viewer" },
];

export function gbmTabsHtml(activeTab) {
  return GBM_TABS.map((tab) => `<button class="tab ${tab.id === activeTab ? "active" : ""}" type="button" data-gbm-tab="${tab.id}">${tab.label}</button>`).join("");
}

export function gbmPanelClass(activeTab, panel) {
  return `gbm-tab-panel ${activeTab === panel ? "" : "hidden"}`;
}

export function bindGbmTabs(mount, onSelect) {
  for (const button of mount.querySelectorAll("[data-gbm-tab]")) {
    button.addEventListener("click", () => onSelect(button.dataset.gbmTab));
  }
}

export function syncGbmRenderedTab(mount, nextTab) {
  for (const button of mount.querySelectorAll("[data-gbm-tab]")) {
    button.classList.toggle("active", button.dataset.gbmTab === nextTab);
  }
  for (const panel of mount.querySelectorAll("[data-gbm-panel]")) {
    panel.classList.toggle("hidden", panel.dataset.gbmPanel !== nextTab);
  }
}
