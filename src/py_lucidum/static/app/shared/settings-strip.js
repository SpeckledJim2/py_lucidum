const boundSettingsStrips = new WeakSet();

export function bindSettingsStripOverflowCue(toolbar) {
  if (!toolbar || boundSettingsStrips.has(toolbar)) return () => {};
  boundSettingsStrips.add(toolbar);

  let syncScheduled = false;
  let disposed = false;
  const syncOverflow = () => {
    if (disposed) return;
    syncScheduled = false;
    const maxScrollLeft = Math.max(0, toolbar.scrollWidth - toolbar.clientWidth);
    toolbar.classList.toggle("app-settings-overflow-left", toolbar.scrollLeft > 1);
    toolbar.classList.toggle(
      "app-settings-overflow-right",
      maxScrollLeft > 1 && toolbar.scrollLeft < maxScrollLeft - 1,
    );
  };
  const scheduleSync = () => {
    if (disposed || syncScheduled) return;
    syncScheduled = true;
    requestAnimationFrame(syncOverflow);
  };

  toolbar.addEventListener("scroll", scheduleSync, { passive: true });
  let resizeObserver = null;
  if (typeof ResizeObserver === "function") {
    resizeObserver = new ResizeObserver(scheduleSync);
    resizeObserver.observe(toolbar);
  }
  let mutationObserver = null;
  if (typeof MutationObserver === "function") {
    mutationObserver = new MutationObserver(scheduleSync);
    mutationObserver.observe(toolbar, {
      attributes: true,
      attributeFilter: ["class"],
      characterData: true,
      childList: true,
      subtree: true,
    });
  }
  scheduleSync();
  return () => {
    if (disposed) return;
    disposed = true;
    toolbar.removeEventListener("scroll", scheduleSync);
    resizeObserver?.disconnect();
    mutationObserver?.disconnect();
    boundSettingsStrips.delete(toolbar);
  };
}
