// Robust KaTeX auto-render integration for MkDocs Material (instant navigation)
// Causes of "first load not rendered":
// 1. KaTeX auto-render script loads after our script executes => window.renderMathInElement undefined initially.
// 2. Content area replaced by Material's instant navigation after initial DOMContentLoaded.
// 3. Race between script loading order and hydration of .md-content.

const MATH_RENDER_CONFIG = {
  delimiters: [
    { left: "$$", right: "$$", display: true },
    { left: "$", right: "$", display: false },
    { left: "\\(", right: "\\)", display: false },
    { left: "\\[", right: "\\]", display: true }
  ],
  ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"],
  throwOnError: false
};

let lastRenderSignature = null;

function renderAllMath(root) {
  if (!window.renderMathInElement) return false;
  const container = root || document.querySelector('.md-content') || document.body;
  // Create a signature to avoid redundant full re-renders (optional micro-optimization)
  const sig = container.innerHTML.length;
  if (lastRenderSignature === sig) return true; // assume already rendered
  window.renderMathInElement(container, MATH_RENDER_CONFIG);
  renderTOCMath();
  lastRenderSignature = sig;
  return true;
}

function attemptRenderWithRetry(max=40, interval=50){
  let tries = 0;
  (function loop(){
    if (renderAllMath()) return;
    if (tries++ < max) setTimeout(loop, interval);
  })();
}

// Primary initial hook
document.addEventListener('DOMContentLoaded', () => {
  attemptRenderWithRetry();
});

// MkDocs Material provides a global observable "document$" (if features enabled)
if (window.document$) {
  window.document$.subscribe(() => {
    // Defer slightly to allow DOM swap to settle
    setTimeout(() => attemptRenderWithRetry(10, 40), 10);
    setTimeout(renderTOCMath, 25);
  });
}

// Backward compatibility: listen for legacy event name if present
document.addEventListener('DOMContentSwitch', () => {
  setTimeout(() => attemptRenderWithRetry(10, 40), 10);
});

// MutationObserver fallback (in case navigation events missed)
let observerStarted = false;
function ensureObserver(){
  if (observerStarted) return;
  const target = document.querySelector('.md-content');
  if (!target) return setTimeout(ensureObserver, 100);
  observerStarted = true;
  const obs = new MutationObserver((mutations) => {
    for (const m of mutations) {
      if (m.type === 'childList') {
        attemptRenderWithRetry(5, 30);
        renderTOCMath();
        break;
      }
    }
  });
  obs.observe(target, { childList: true, subtree: true });

  // Also observe the secondary ToC; it changes when scrolling or headings update
  const toc = document.querySelector('.md-nav--secondary');
  if (toc) {
    const tocObs = new MutationObserver(() => { renderTOCMath(); });
    tocObs.observe(toc, { childList: true, subtree: true });
  }
}
ensureObserver();

// Expose manual trigger for console debugging if needed
window.__forceMathRender = () => attemptRenderWithRetry(1, 0);

// ---------- TOC Math Rendering (anchors) ----------
function renderTOCMath(){
  try {
    if(!window.katex || !window.renderMathInElement) return;
    const tocNav = document.querySelector('#toc nav, .md-nav--secondary nav');
    if(!tocNav) return;
    // Avoid reprocessing if already rendered
    if(tocNav.dataset.tocMathProcessed === '1') return;
    const anchors = tocNav.querySelectorAll('a');
    if(!anchors.length) return;
    let needs = false;
    anchors.forEach(a => { if(/\\\(|\\\[|\$[^$].*\$/.test(a.textContent) || /\\\(|\\\[|\$[^$].*\$/.test(a.innerHTML)) needs = true; });
    if(!needs){ tocNav.dataset.tocMathProcessed = '1'; return; }
    // Build sandbox with cloned anchor labels only (no links) to render inline math safely
    const sandbox = document.createElement('div');
    sandbox.style.position='absolute'; sandbox.style.left='-9999px'; sandbox.style.top='0'; sandbox.style.width='10px'; sandbox.style.overflow='hidden';
    sandbox.innerHTML = Array.from(anchors).map(a => `<div class="toc-math-frag">${a.innerHTML}</div>`).join('');
    document.body.appendChild(sandbox);
    try {
      window.renderMathInElement(sandbox, MATH_RENDER_CONFIG);
      const frags = sandbox.querySelectorAll('.toc-math-frag');
      frags.forEach((frag, i) => {
        // Normalize block math to inline
        frag.querySelectorAll('.katex-display').forEach(el => { el.classList.remove('katex-display'); el.style.display='inline-block'; });
        anchors[i].innerHTML = frag.innerHTML;
      });
      tocNav.dataset.tocMathProcessed = '1';
    } catch(e){ /* ignore */ }
    sandbox.remove();
  } catch(err){ /* silent */ }
}
