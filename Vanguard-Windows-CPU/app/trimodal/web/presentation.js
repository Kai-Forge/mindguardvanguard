/* Read-only presentation bindings: no requests, model calls, input mutation or timers.
 * Existing app.js remains the sole owner of data and interaction behavior.
 */
(() => {
  const watch = (element, callback, attributes = []) => {
    callback();
    new MutationObserver(callback).observe(element, {
      childList: true, characterData: true, subtree: true,
      ...(attributes.length ? {attributes: true, attributeFilter: attributes} : {})
    });
  };
  const byId = id => document.getElementById(id);
  for (const button of document.querySelectorAll('.nav,.mode-switch button')) {
    watch(button, () => button.setAttribute('aria-pressed',
      String(button.classList.contains('active') || button.classList.contains('selected'))), ['class']);
  }
  const tones = {
    connection: text => text.startsWith('3/3') ? 'success' : text.includes('未连接') ? 'error' : /^\d\/3/.test(text) ? 'warning' : 'neutral',
    fusionState: text => text === '三路已融合' ? 'success' : 'neutral',
    recordState: text => text.includes('正在保存') ? 'success' : 'neutral',
    cameraState: text => text === 'LIVE' ? 'success' : 'neutral',
    micState: text => text === 'LIVE' ? 'success' : 'neutral',
    imageBadge: text => text === '输入异常' ? 'error' : 'neutral',
    speechBadge: text => text === '输入异常' ? 'error' : 'neutral',
    eggBadge: text => text === '输入异常' ? 'error' : 'neutral'
  };
  for (const [id, tone] of Object.entries(tones)) {
    const element = byId(id);
    watch(element, () => { element.dataset.tone = tone(element.textContent); });
  }
  // Numeric text stays exactly as rendered by app.js. No synthetic percentage
  // or aria-valuenow is assigned when a model has not produced a result.
  for (const key of ['image', 'speech', 'egg', 'fusion']) {
    const element = byId(key + 'Bars');
    element.setAttribute('role', 'group');
    element.setAttribute('aria-label', '负向、中性、正向概率分布');
    for (const row of element.children) row.querySelector('.track').setAttribute('aria-hidden', 'true');
    watch(element, () => {
      element.dataset.empty = String([...element.querySelectorAll('.bar-number')]
        .every(number => number.textContent === '—'));
    });
  }
})();
