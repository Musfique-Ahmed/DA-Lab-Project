# DA Lab Project

Companion site for the **UIU Data Analytics Laboratory** project proposal (Summer 2026 Trimester) — *Forecasting loan default at the point of application* on the **Home Credit Default Risk** dataset.

## What's here

- `index.html` — a single, self-contained webpage that mirrors the proposal narrative with interactive charts, a click-through **presentation mode** (PPT-style), and a working mock of the planned analyst dashboard. No build step — just open in a browser.

## Open the site

```bash
# any modern browser, no server required
start index.html        # Windows
open index.html         # macOS
xdg-open index.html     # Linux
```

Or visit the published page: **[musfique-ahmed.github.io/DA-Lab-Project](https://musfique-ahmed.github.io/DA-Lab-Project/)**

## Presentation mode

Click the **Present** button (top-right) or press **`P`** to enter deck mode. Then:

| Action | Key / Input |
|---|---|
| Next slide | **Click** anywhere · `→` · `Space` · `PageDown` |
| Previous slide | `←` · `PageUp` |
| First / Last | `Home` · `End` |
| Exit | `Esc` · `P` |

## Tech

- Vanilla HTML, CSS, and JavaScript
- [Chart.js](https://www.chartjs.org/) (loaded from CDN) for the donut, volume, feature-importance, and segment charts
- [Google Fonts](https://fonts.google.com/) — Fraunces (display) + Inter (body)
- Single file, ~80 KB, opens from `file://`

## License

MIT — see [LICENSE](./LICENSE).

---

© 2026 Musfique Ahmed
