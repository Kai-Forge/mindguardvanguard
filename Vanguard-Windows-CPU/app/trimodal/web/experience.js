/* UI-only interactions. app.js owns devices, requests, inference and recording. */
(() => {
  const get = id => document.getElementById(id);
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const animate = (element, frames, duration = 420) => {
    if (reduced.matches || !element.animate) return;
    element.getAnimations().forEach(animation => animation.cancel());
    element.animate(frames, {duration, easing: 'ease-out'});
  };
  get('save').addEventListener('change', () => {
    if (get('save').checked) get('captureSettings').open = true;
  });
  // Expose the relevant input if existing validation reports a missing participant.
  new MutationObserver(() => {
    if (get('notice').textContent.includes('参与者编号')) get('captureSettings').open = true;
  }).observe(get('notice'), {childList: true, subtree: true, characterData: true});

  for (const key of ['image', 'speech', 'egg', 'fusion']) {
    const bars = get(key + 'Bars');
    const prediction = get(key + 'Prediction');
    let lastSignature = '';
    const update = () => {
      const numbers = [...bars.querySelectorAll('.bar-number')].map(el => el.textContent);
      const empty = numbers.every(value => value === '—');
      const values = [...bars.querySelectorAll('.fill')].map(el => parseFloat(el.style.width) || 0);
      const signature = numbers.join('|') + prediction.textContent;
      prediction.dataset.result = ({'负向':'negative', '中性':'neutral', '正向':'positive'})[prediction.textContent] || 'empty';
      if (key === 'fusion') {
        const orbit = get('fusionOrbit');
        orbit.dataset.empty = String(empty);
        // Use exact widths written by the existing renderer, not rounded labels.
        // This is another rendering of the same probabilities, not a new score.
        const total = values.reduce((a,b) => a+b, 0);
        if (!empty && total > 0) {
          const a = values[0] / total * 100;
          const b = (values[0] + values[1]) / total * 100;
          orbit.style.background = `conic-gradient(var(--negative) 0% ${a}%,var(--neutral) ${a}% ${b}%,var(--positive) ${b}% 100%)`;
        } else orbit.style.removeProperty('background');
      }
      if (signature !== lastSignature && !empty) {
        animate(prediction, [{opacity:.5,transform:'translateY(4px)'},{opacity:1,transform:'translateY(0)'}]);
        const panel = bars.closest('.card,.fusion-panel');
        animate(panel, [{boxShadow:'0 0 0 1px #91e1ce88,0 0 28px #70cdb414'},{boxShadow:'0 0 0 0px #91e1ce00,0 0 0px #70cdb400'}], 650);
      }
      lastSignature = signature;
    };
    new MutationObserver(update).observe(bars, {childList:true,characterData:true,subtree:true});
    update();
  }
  document.querySelectorAll('.nav').forEach(button => button.addEventListener('click', () => {
    animate(get('cards'), [{opacity:.5,transform:'translateY(6px)'},{opacity:1,transform:'none'}], 260);
  }));
})();
