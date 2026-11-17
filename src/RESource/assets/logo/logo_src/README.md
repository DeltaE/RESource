# RESource logo sources

Author: EL

| File (in `src/assets`) | Use |
|---|---|
| `logo_dash_animated.svg` | Intro animation, about 7 s, plays once: grey grid, tilt, land use, exclusions, solar/wind, logo |
| `logo_dash.svg` | Full static logo with wordmark (About page, reports) |
| `logo_dash_min.svg` | Minimal logo with wordmark, flat colours |
| `logo_dash_mark.svg` | Minimal mark, no wordmark (header, favicon) |
| `logo_dash_mark_light.svg` | Minimal logo with white wordmark, for dark backgrounds |
| `logo_loader.svg` | Looping loader mark, 1.4 s cycle |
| `loader_demo.html` | Blurred overlay loader: CSS, markup and `RDLoader.show()/hide()` API to copy into the dash |

Regenerate the logos: `python gen_logo.py` (needs `fonttools`). The wordmark is converted to outlines from `Outfit.ttf` (SIL Open Font License), so no font loads at runtime.

All SVGs are plain CSS animation with no JavaScript, so they embed as data URIs in the single-file build. Each SVG respects `prefers-reduced-motion`.
