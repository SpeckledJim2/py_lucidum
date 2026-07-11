function defaultItemKey(item) {
  return item?.dataset?.navigationKey || "";
}

function documentLostFocus(item) {
  const documentNode = item?.ownerDocument;
  const focused = documentNode?.activeElement;
  return !focused || focused === documentNode.body || focused === documentNode.documentElement;
}

export function bindVerticalListNavigation({
  list,
  itemSelector,
  getItemKey = defaultItemKey,
  onActivate,
}) {
  if (!list || !itemSelector || typeof onActivate !== "function") {
    throw new Error("Vertical list navigation requires a list, item selector, and activation callback.");
  }

  let navigationSequence = 0;

  function items() {
    return [...list.querySelectorAll(itemSelector)]
      .filter((item) => !item.disabled && item.offsetParent !== null);
  }

  function itemByKey(key) {
    if (!key) return null;
    return items().find((item) => getItemKey(item) === key) || null;
  }

  function focusedItemKey() {
    const focused = list.ownerDocument.activeElement;
    const item = focused?.closest?.(itemSelector);
    return item && list.contains(item) ? getItemKey(item) : "";
  }

  function focusItem(key) {
    const item = itemByKey(key);
    if (!item) return false;
    item.focus({ preventScroll: true });
    item.scrollIntoView({ block: "nearest" });
    return true;
  }

  async function activateItem(key) {
    if (!key) return;
    const sequence = ++navigationSequence;
    focusItem(key);
    await onActivate(key);
    if (sequence !== navigationSequence) return;
    if (focusedItemKey() === key) return;
    if (documentLostFocus(list)) focusItem(key);
  }

  function handleClick(event) {
    const item = event.target.closest?.(itemSelector);
    if (!item || !list.contains(item)) return;
    list.classList.remove("list-keyboard-navigation");
    return activateItem(getItemKey(item));
  }

  function handleKeydown(event) {
    if (!["ArrowUp", "ArrowDown"].includes(event.key)) return;
    const currentItem = event.target.closest?.(itemSelector);
    if (!currentItem || !list.contains(currentItem)) return;
    const availableItems = items();
    const currentIndex = availableItems.indexOf(currentItem);
    if (currentIndex < 0) return;
    const direction = event.key === "ArrowDown" ? 1 : -1;
    const nextIndex = Math.min(Math.max(currentIndex + direction, 0), availableItems.length - 1);
    event.preventDefault();
    if (nextIndex === currentIndex) return;

    const targetKey = getItemKey(availableItems[nextIndex]);
    if (!targetKey) return;
    list.classList.add("list-keyboard-navigation");
    return activateItem(targetKey);
  }

  function handleFocusout(event) {
    if (event.relatedTarget && !list.contains(event.relatedTarget)) navigationSequence += 1;
  }

  function handlePointermove() {
    list.classList.remove("list-keyboard-navigation");
  }

  list.addEventListener("click", handleClick);
  list.addEventListener("keydown", handleKeydown);
  list.addEventListener("focusout", handleFocusout);
  list.addEventListener("pointermove", handlePointermove);

  return {
    destroy() {
      navigationSequence += 1;
      list.removeEventListener("click", handleClick);
      list.removeEventListener("keydown", handleKeydown);
      list.removeEventListener("focusout", handleFocusout);
      list.removeEventListener("pointermove", handlePointermove);
      list.classList.remove("list-keyboard-navigation");
    },
    focusItem,
    focusedItemKey,
  };
}
