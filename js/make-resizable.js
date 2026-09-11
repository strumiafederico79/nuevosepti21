// js/make-resizable.js — Helper de drag compartido (mouse+touch+keyboard)
export function makeResizable(handleEl, options = {}) {
  const { axis, getSize, setSize, min = 100, max = 800, invert = false, onStart, onEnd } = options;
  if (!handleEl || !axis || !getSize || !setSize) return () => {};
  let dragging = false, startPos = 0, startSize = 0;
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const getMin = typeof min === 'function' ? min : () => min;
  const getMax = typeof max === 'function' ? max : () => max;
  function toDelta(clientPos) {
    const delta = invert ? startPos - clientPos : clientPos - startPos;
    return clamp(startSize + delta, getMin(), getMax());
  }
  function onMove(e) {
    if (!dragging) return;
    e.preventDefault();
    const clientPos = axis === 'x' ? (e.clientX ?? e.touches?.[0]?.clientX) : (e.clientY ?? e.touches?.[0]?.clientY);
    if (clientPos == null) return;
    setSize(toDelta(clientPos));
  }
  function onUp() {
    if (!dragging) return;
    dragging = false;
    handleEl.classList.remove('dragging');
    document.removeEventListener('mousemove', onMove, { passive: false });
    document.removeEventListener('mouseup', onUp);
    document.removeEventListener('touchmove', onMove, { passive: false });
    document.removeEventListener('touchend', onUp);
    onEnd?.();
  }
  function onDown(e) {
    if (e.button && e.button !== 0) return;
    dragging = true;
    handleEl.classList.add('dragging');
    startPos = axis === 'x' ? e.clientX : e.clientY;
    startSize = getSize();
    onStart?.();
    document.addEventListener('mousemove', onMove, { passive: false });
    document.addEventListener('mouseup', onUp);
    document.addEventListener('touchmove', onMove, { passive: false });
    document.addEventListener('touchend', onUp);
  }
  function onKey(e) {
    if (!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','Home','End'].includes(e.key)) return;
    e.preventDefault();
    const step = e.shiftKey ? 40 : 12;
    let cur = getSize();
    if (e.key === 'Home') cur = getMin();
    else if (e.key === 'End') cur = getMax();
    else {
      const dir = invert ? -1 : 1;
      const axisDir = (axis === 'x' && (e.key === 'ArrowRight')) || (axis === 'y' && (e.key === 'ArrowDown')) ? 1 : -1;
      cur = clamp(cur + dir * axisDir * step, getMin(), getMax());
    }
    setSize(cur);
    onEnd?.();
  }
  handleEl.addEventListener('mousedown', onDown);
  handleEl.addEventListener('touchstart', onDown, { passive: false });
  handleEl.addEventListener('keydown', onKey);
  return function destroy() {
    handleEl.removeEventListener('mousedown', onDown);
    handleEl.removeEventListener('touchstart', onDown);
    handleEl.removeEventListener('keydown', onKey);
    document.removeEventListener('mousemove', onMove, { passive: false });
    document.removeEventListener('mouseup', onUp);
    document.removeEventListener('touchmove', onMove, { passive: false });
    document.removeEventListener('touchend', onUp);
  };
}
