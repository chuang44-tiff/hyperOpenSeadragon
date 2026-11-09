# Changelog

All notable changes to hyperOpenSeadragon. Versions are tagged on GitHub.

## v3.7.2 — 2026-11-09

- README: PICASSO is no longer labelled beta.
- JOSS paper: "N up to 8" replaces a symbol the JOSS PDF font cannot render. No code changes.

## v3.7.1 — 2026-11-09

- `CITATION.cff`, `.zenodo.json`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `THIRD_PARTY_NOTICES.md`, this changelog, and the JOSS paper (`paper/`).
- README: the demo dataset is a mouse knee from the high-throughput hyperspectral imaging paper; the PICASSO section no longer describes the removed K spinner. No code changes.

## v3.7 — 2026-10-19

- H&E and Masson Trichrome run entirely on the GPU, with their own controls under Post-Processing: k, the stain gains, unmix (H&E) and the hue, saturation and value of each stain, with a live colour swatch and a Reset button. H&E and Trichrome keep separate settings. The defaults render exactly as before.
- A dataset can set its own stain style with `defaultStainHE` / `defaultStainTrichrome` in the DATASET CONFIGURATION block.
- The old CPU stain filter code is removed, so the filter script is smaller and stains switch on faster.
- Without WebGL, the H&E and Trichrome toggles are disabled with an explanation instead of doing nothing.
- Opening the viewer from a plain web server or from disk no longer logs a 404 from the optional PICASSO server check.

## v3.6 — 2026-10-19

- Synthetic 16-channel, two-z-level demo dataset (`demo_data/`) and its generator (`make_demo_data.py`); `zstackHyper.html` opens it as shipped.
- Test suite (`tests/`) covering the viewer pages, the tile pipeline, the demo dataset and a headless-browser smoke test, run on every push and pull request by GitHub Actions.
- Removed an unused NIST page stylesheet and the NIST logo images.

## v3.5.2 — 2026-10-09

- Annotation survives a narrow window: shrinking the browser until the side panel covered the viewer no longer freezes the image and the drawn shapes.

## v3.5.1 — 2026-10-09

- Faster first view and less GPU memory: minified OpenSeadragon and filter scripts (with fallbacks to the full files), and neighbouring z-levels are fetched only after the visible view has loaded. On a 5 Mbit/s link the first tile appears about 40% sooner; peak GPU memory drops by 28–53%. Trade-off: on a slow link, a z-level switch within about 20 s of page load can take a few seconds longer.
- Fixes a stale image on the first Linear Unmixing Apply with more than 4 outputs, and when re-enabling Denoise or H&E.

## v3.5 — 2026-10-03

- Every packed channel survives zoom-out. Packed RGBA datasets made by older versions should be regenerated.
- Simplified viewer, renderer and tile-generator code, with no intended change in behaviour.

## v3.4.4 — 2026-08-28

- Loading a mask without its `classes.json` traces it into editable polygons automatically; traced polygons can be edited with the brush and eraser.

## v3.4.3 — 2026-08-28

- Save Mask / Load Mask replace Export Mask; saves honour the Downscale setting (default 4096 px).

## v3.4.2 — 2026-08-27

- Overlapping annotation classes, per-class mask files, and vertex editing.

## v3.4.1 — 2026-08-25

- Polygons are protected from the eraser; edge-insert handles; hardened path handling in the tile-generator app.

## v3.4 — 2026-08-21

- Polygon annotation tool, separate Polygon and Brush modes, and keyboard shortcuts.

## v3.3 — 2026-07-05

- Multi-class paintbrush annotation (OSDAnnotator).

## v3.2.1 — 2026-06-26

- On-the-fly, display-only GPU guided-filter denoise with a strength slider.

## v3.2 — 2026-06-15

- In-browser PICASSO matrix generation in a Web Worker (no server needed) and GPU PICASSO apply for up to 8 channels.
- Fixes grayscale colour corruption when packing channels, and a startup error with default configuration variables.

## v3.1 — 2026-06-13

- Performance pass; fixes the fourth Linear Unmixing output in RGB mode and the DZI pyramid.

## v3.0 — 2026-06-07

- PICASSO non-linear unmixing.

## v2.1 — 2026-04-08

- Spectrum Inspector: right-click a pixel to see its spectrum and add it to the unmixing matrix.

## v2.0 — 2026-03-26

- Interactive linear spectral unmixing.

## v1.0 — 2026-03-22

- Initial release: multi-channel blending viewer and tile generator.
