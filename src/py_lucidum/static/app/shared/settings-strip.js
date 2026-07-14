const boundSettingsStrips = new WeakSet();

export function bindSettingsStripOverflowCue(toolbar) {
  if (!toolbar || boundSettingsStrips.has(toolbar)) return;
  boundSettingsStrips.add(toolbar);

  let syncScheduled = false;
  const syncOverflow = () => {
    syncScheduled = false;
    const maxScrollLeft = Math.max(0, toolbar.scrollWidth - toolbar.clientWidth);
    toolbar.classList.toggle("app-settings-overflow-left", toolbar.scrollLeft > 1);
    toolbar.classList.toggle(
      "app-settings-overflow-right",
      maxScrollLeft > 1 && toolbar.scrollLeft < maxScrollLeft - 1,
    );
  };
  const scheduleSync = () => {
    if (syncScheduled) return;
    syncScheduled = true;
    requestAnimationFrame(syncOverflow);
  };

  toolbar.addEventListener("scroll", scheduleSync, { passive: true });
  if (typeof ResizeObserver === "function") {
    const resizeObserver = new ResizeObserver(scheduleSync);
    resizeObserver.observe(toolbar);
  }
  if (typeof MutationObserver === "function") {
    const mutationObserver = new MutationObserver(scheduleSync);
    mutationObserver.observe(toolbar, {
      attributes: true,
      attributeFilter: ["class"],
      characterData: true,
      childList: true,
      subtree: true,
    });
  }
  scheduleSync();
}
