// Progress tracking for curriculum checklist
(function(){
  // Note completion & curriculum progress script
  const STORAGE_KEY = 'note-completion-v1';

  function getNoteId(){
    // Derive a stable note id from URL path for ANY page under /notes/
    // Examples:
    //  - /notes/note6/               -> note6
    //  - /notes/module4-clustering/  -> module4-clustering
    //  - /notes/module3-iris-classification/ -> module3-iris-classification
    const path = window.location.pathname.replace(/\/index\.html$/, '');
    const parts = path.split('/').filter(Boolean);
    const notesIdx = parts.indexOf('notes');
    if (notesIdx >= 0 && parts.length > notesIdx + 1) {
      let slug = parts[notesIdx + 1];
      slug = slug.replace(/\.html$/, '').replace(/\/index$/, '');
      return slug || null;
    }
    // Fallback to legacy pattern
    const legacy = parts.find(p => /^note\d+/.test(p));
    return legacy || null;
  }

  function loadState(){
    try { return JSON.parse(localStorage.getItem(STORAGE_KEY)) || {}; } catch(e){ return {}; }
  }
  function saveState(state){
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  }

  function markCompleteButton(noteId, state){
    if(!noteId) return;
    // MkDocs Material uses article.md-content__inner.md-typeset (classes on same element)
    let container = document.querySelector('main .md-content__inner.md-typeset');
    if(!container) {
      // fallback to previous (older structure) or nested variant
      container = document.querySelector('main .md-content__inner .md-typeset');
    }
    if(!container) return;
    // Avoid duplicate insertion of section
    if(container.querySelector('.note-completion-section')) return;
    const isDone = !!state[noteId];

    const section = document.createElement('section');
      // Compute relative path back to site root so curriculum link works at any depth
      const pathParts = window.location.pathname.split('/').filter(Boolean);
      const relToRoot = pathParts.map(()=>'..').join('/') || '.'; // e.g. ../../ for /notes/note1/
    section.className = 'note-completion-section';
    section.innerHTML = `
      <hr class="note-sep" />
      <div class="note-completion-inner" role="complementary" aria-label="Note completion controls">
        <div class="note-completion-status" data-status="${isDone ? 'complete' : 'incomplete'}">
          <strong>Status:</strong> <span class="status-text">${isDone ? 'Completed ✅' : 'Incomplete'}</span>
        </div>
        <div class="note-completion-actions">
          <button type="button" class="note-complete-toggle" data-complete="${isDone}"> ${isDone ? '✓ Note Completed' : 'Mark as Complete'} </button>
          <button type="button" class="note-reset-toggle" aria-label="Reset completion for this note" title="Reset completion">Reset</button>
            <a class="note-curriculum-link" href="${relToRoot}/MLUP-Notes/curriculum/" title="View overall curriculum progress">View Curriculum Progress →</a>
        </div>
        <p class="note-completion-hint">Progress is stored locally in your browser (no server sync). Clearing site data resets it.</p>
      </div>`;

    // Append near end
    container.appendChild(section);

    const btn = section.querySelector('.note-complete-toggle');
    const resetBtn = section.querySelector('.note-reset-toggle');
    const statusText = section.querySelector('.status-text');
    const statusWrap = section.querySelector('.note-completion-status');

    function updateUI(done){
      btn.dataset.complete = done;
      btn.innerHTML = done ? '✓ Note Completed' : 'Mark as Complete';
      statusText.textContent = done ? 'Completed ✅' : 'Incomplete';
      statusWrap.dataset.status = done ? 'complete' : 'incomplete';
    }

    btn.addEventListener('click', () => {
      const current = loadState();
      current[noteId] = !current[noteId];
      saveState(current);
      updateUI(!!current[noteId]);
      updateOverallProgress();
    });

    resetBtn.addEventListener('click', () => {
      const current = loadState();
      if(current[noteId]){
        delete current[noteId];
        saveState(current);
        updateUI(false);
        updateOverallProgress();
      }
    });
  }

  function extractAllNoteIds(){
    // Collect all unique slugs under /notes/ from the navigation
    const links = Array.from(document.querySelectorAll('nav a, .md-nav a')).map(a=>a.getAttribute('href'));
    const ids = new Set();
    links.forEach(href => {
      if(!href) return;
      const m = href.match(/(?:^|\/)notes\/(.+?)(?:\/?|\.html|\.md)(?:#|$)/);
      if(m){
        let slug = m[1].replace(/\.md$/, '').replace(/\/index$/, '');
        if(slug) ids.add(slug);
      }
    });
    return Array.from(ids).sort();
  }

  function updateOverallProgress(){
    const fill = document.getElementById('curriculum-progress-fill');
    const pct = document.getElementById('curriculum-progress-percent');
  const state = loadState();
  const notes = extractAllNoteIds();
    if(notes.length){
      const done = notes.filter(id => state[id]).length;
      const percent = Math.round(done/notes.length*100);
      if(fill) fill.style.width = percent + '%';
      if(pct) pct.textContent = percent + '%';
      updateMiniWidget(done, notes.length, percent);
    }
  }

  // Mini progress widget injection (homepage & any page)
  function ensureMiniWidget(){
    if(document.querySelector('.mini-progress-widget')) return; // already present
    const mountTarget = document.querySelector('main .md-content__inner.md-typeset, main .md-content__inner .md-typeset');
    if(!mountTarget) return;
    const widget = document.createElement('div');
    widget.className = 'mini-progress-widget';
    widget.innerHTML = `
      <div class="mpw-top">
        <div class="mpw-circle-wrap" aria-label="Overall notes completion">
          <svg class="mpw-circle" viewBox="0 0 40 40" width="72" height="72" role="img">
            <circle class="mpw-circle-bg" cx="20" cy="20" r="18" />
            <circle class="mpw-circle-fg" cx="20" cy="20" r="18" stroke-dasharray="113.097" stroke-dashoffset="113.097" />
            <text class="mpw-circle-text" x="50%" y="50%" dominant-baseline="middle" text-anchor="middle">0%</text>
          </svg>
        </div>
        <div class="mpw-meta">
          <div class="mpw-count">0/0</div>
          <div class="mpw-percent">0%</div>
          <button type="button" class="mpw-toggle-remaining" aria-expanded="false" aria-controls="mpw-remaining-list">Remaining ▼</button>
        </div>
      </div>
      <div class="mpw-remaining hidden" id="mpw-remaining-list" aria-live="polite">
        <ul class="mpw-remaining-list"></ul>
      </div>
    `;
    // Prefer top insertion on homepage; else append near end
    mountTarget.appendChild(widget);

    // toggle remaining list
    const toggleBtn = widget.querySelector('.mpw-toggle-remaining');
    const remainingBox = widget.querySelector('.mpw-remaining');
    if(toggleBtn && remainingBox){
      toggleBtn.addEventListener('click', () => {
        const expanded = toggleBtn.getAttribute('aria-expanded') === 'true';
        toggleBtn.setAttribute('aria-expanded', !expanded);
        remainingBox.classList.toggle('hidden', expanded);
        toggleBtn.textContent = expanded ? 'Remaining ▼' : 'Hide ▲';
      });
    }
  }

  function updateMiniWidget(done, total, percent){
    const widget = document.querySelector('.mini-progress-widget');
    if(!widget) return;
    const count = widget.querySelector('.mpw-count');
    const perc = widget.querySelector('.mpw-percent');
    if(count) count.textContent = `${done}/${total}`;
    if(perc) perc.textContent = percent + '%';

    // SVG circle
    const circle = widget.querySelector('.mpw-circle-fg');
    const text = widget.querySelector('.mpw-circle-text');
    if(circle){
      const circumference = 2 * Math.PI * 18; // r=18
      const offset = circumference * (1 - percent/100);
      circle.style.strokeDashoffset = offset.toFixed(3);
    }
    if(text) text.textContent = percent + '%';

    // Remaining list
    const listEl = widget.querySelector('.mpw-remaining-list');
    if(listEl){
      const state = loadState();
      const notes = extractAllNoteIds().filter(id => /note\d+/.test(id));
      const remaining = notes.filter(id => !state[id]);
      if(!remaining.length){
        listEl.innerHTML = '<li class="mpw-empty">All notes completed 🎉</li>';
      } else {
        listEl.innerHTML = remaining.map(id => `<li><a href="../${id}/" data-note-link="${id}">${id.replace('note','Note ')}</a></li>`).join('');
      }
    }
  }

  function init(){
    const state = loadState();
    markCompleteButton(getNoteId(), state);
    ensureMiniWidget();
    updateOverallProgress();
  }

  if(window.document$){
    window.document$.subscribe(() => setTimeout(init, 40));
  } else {
    document.addEventListener('DOMContentLoaded', init);
  }
})();
