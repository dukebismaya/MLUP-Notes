# Dev Bismaya – AI & ML Notes (MkDocs)

This repository has been migrated from a custom single-page app (Primer CSS + Marked + DOMPurify) to a **MkDocs Material** documentation site with built‑in search, navigation, and KaTeX math rendering.

## Current Stack
| Layer | Tool | Purpose |
|-------|------|---------|
| Static Site Generator | MkDocs + Material theme | Structure, navigation, search |
| Markdown Extensions | `pymdownx.*` | Admonitions, details, code fences, math hooks |
| Math | KaTeX (`pymdownx.arithmatex`) | `$..$`, `$$..$$`, `\( .. \)`, `\[ .. \]` |
| Styling | Material + `docs/assets/css/extra.css` | Badge + minor layout tweaks |
| JS Enhancements | `docs/assets/js/math.js` | Re-render math after instant navigation |

## Key Features
- Instant navigation (Material `navigation.instant`)
- Dark / light palette switching
- First‑class search (lunr) with no custom indexing code
- KaTeX math rendering for inline & block formulas
- Collapsible sections using `<details>` or `pymdownx.details`
- Syntax highlighting & copy buttons
- Front matter–like tags retained (can later integrate a tags index)

## Project Structure (Post-Migration)
```
mkdocs.yml                     # MkDocs configuration (theme, nav, extensions)
docs/
	index.md                     # Landing page
	notes/                       # All note markdown files
		note1.md ... note15.md
	assets/
		css/extra.css              # Custom overrides
		js/math.js                 # Math auto-render helper
```

Legacy files removed: `index.html`, `assets/js/app.js`, `sw.js` (service worker), and the old SPA-specific CSS/JS logic. If you still need any styling snippets from `assets/css/custom.css`, migrate relevant rules into `docs/assets/css/extra.css`.

## Local Development
Activate your Python environment (optional if using system Python), then:

```powershell
pip install mkdocs mkdocs-material pymdown-extensions
mkdocs serve
```

Open: http://127.0.0.1:8000/

The site auto-reloads on file changes under `docs/`.

## Building For Deployment
```powershell
mkdocs build
```
Outputs static site to `site/` (add to `.gitignore` if not already).

## Deploying to GitHub Pages
Simplest: built-in deploy command (creates/updates `gh-pages` branch):
```powershell
mkdocs gh-deploy --clean
```
Make sure repository Settings → Pages points to `gh-pages` branch (root).

## Adding New Notes
1. Create a new markdown file: `docs/notes/note16.md` (or a semantic name).  
2. Add it to the `nav:` section in `mkdocs.yml` under the Notes group.  
3. (Optional) Include tags in a front matter style block at top for future tag indexing.

Example front matter style (retained as plain YAML at top):
```markdown
---
title: "Advanced Dimensionality Reduction"
description: "Manifold and probabilistic approaches"
tags: [dimensionality-reduction, manifolds]
---
```

## Math Usage
Inline: `$L = -\sum_i y_i \log p_i$`  
Block:
```markdown
$$
VR(k)=\frac{\sum_{i=1}^k \lambda_i}{\sum_{i=1}^n \lambda_i}
$$
```

## Extension Highlights in `mkdocs.yml`
- `pymdownx.arithmatex` – math bridging to KaTeX
- `pymdownx.details` – collapsible sections
- `pymdownx.superfences` – nested code fences & tabs
- `pymdownx.highlight` – enhanced code highlighting + line anchors

## Migrating Additional Styling
If you had custom layout or typography from `custom.css`, copy only the needed rules into `docs/assets/css/extra.css` to avoid theme conflicts.

## Future Enhancements
- Add a tags index page (`material` + custom plugin or a simple manual page).
- Introduce versioning with `mike` if you want historical snapshots.
- Add `mkdocs-minify-plugin` for smaller assets.
- Add `mkdocs-pwa-plugin` if offline capability is desired again.

## Troubleshooting
| Issue | Likely Cause | Fix |
|-------|--------------|-----|
| Math not rendering | KaTeX JS blocked / not loaded | Check console/network; ensure CDN reachable |
| New note not in nav | Missing nav entry in `mkdocs.yml` | Add under `nav:` and re-serve |
| Search misses new note | Build cache during dev | Refresh page; ensure file saved |

## License
MIT (add a LICENSE file if distributing publicly).

---
_Migrated on: 2025-09-24_
