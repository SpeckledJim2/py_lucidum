const boundSettingsStrips = new WeakSet();
const SETTINGS_STRIP_EDITABLE_SELECTOR = "input, textarea, select, [contenteditable]";

function settingsStripWheelDelta(event, toolbar) {
  if (event.deltaMode === 1) return event.deltaY * 16;
  if (event.deltaMode === 2) return event.deltaY * toolbar.clientWidth;
  return event.deltaY;
}

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
  const handleWheel = (event) => {
    if (
      event.ctrlKey
      || event.target?.closest?.(SETTINGS_STRIP_EDITABLE_SELECTOR)
      || Math.abs(event.deltaX) >= Math.abs(event.deltaY)
    ) {
      return;
    }
    const maxScrollLeft = Math.max(0, toolbar.scrollWidth - toolbar.clientWidth);
    if (maxScrollLeft <= 1) return;
    const previousScrollLeft = toolbar.scrollLeft;
    const nextScrollLeft = Math.min(
      maxScrollLeft,
      Math.max(0, previousScrollLeft + settingsStripWheelDelta(event, toolbar)),
    );
    if (Math.abs(nextScrollLeft - previousScrollLeft) <= 0.5) return;
    toolbar.scrollLeft = nextScrollLeft;
    event.preventDefault();
    scheduleSync();
  };

  toolbar.addEventListener("scroll", scheduleSync, { passive: true });
  toolbar.addEventListener("wheel", handleWheel, { passive: false });
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
    toolbar.removeEventListener("wheel", handleWheel);
    resizeObserver?.disconnect();
    mutationObserver?.disconnect();
    boundSettingsStrips.delete(toolbar);
  };
}
