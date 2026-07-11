import {
  bindToolScreenNavigation,
  syncToolScreenNavigation,
  toolScreenNavButtonHtml,
} from "./shared/tool-screen-nav.js";

const GBM_TABS = [
  { id: "features", label: "Features and parameters", icon: "features" },
  { id: "models", label: "Model navigator", icon: "models" },
  { id: "shap", label: "SHAP", icon: "shap" },
  { id: "stacked-shap", label: "Stacked SHAP", icon: "layers" },
  { id: "trees", label: "Tree viewer", icon: "tree" },
];

export function gbmTabsHtml(activeTab) {
  return GBM_TABS.map((tab) => toolScreenNavButtonHtml({
    active: tab.id === activeTab,
    buttonId: `gbm-screen-tab-${tab.id}`,
    controlsId: `gbm-screen-panel-${tab.id}`,
    icon: tab.icon,
    label: tab.label,
    targetId: tab.id,
    toolDataAttribute: "gbm-tab",
  })).join("");
}

export function gbmPanelClass(activeTab, panel) {
  return `gbm-tab-panel ${activeTab === panel ? "" : "hidden"}`;
}

export function bindGbmTabs(mount, onSelect) {
  bindToolScreenNavigation(mount.querySelector(".gbm-tabs"), onSelect);
}

export function syncGbmRenderedTab(mount, nextTab) {
  syncToolScreenNavigation(mount.querySelector(".gbm-tabs"), nextTab);
  for (const panel of mount.querySelectorAll("[data-gbm-panel]")) {
    panel.classList.toggle("hidden", panel.dataset.gbmPanel !== nextTab);
  }
}
