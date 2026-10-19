# Demo dataset

A small synthetic 16-channel fluorescence dataset: 4 emission spectra with
overlapping (crosstalking) Gaussian profiles, smooth blobs, 2 z-levels (0 um and
3 um). It is generated, not measured, and contains no noise.

- `deepzoom_RGBA/{0.0um,3.0um}/{1..4}/stitch.xml` - Deep Zoom tiles, 4 RGBA tile
  sources per z-level = 16 channels.
- `unmix_matrix.txt` - a 4 x 16 Linear Unmixing matrix for this dataset.

Open `zstackHyper.html` from the repository root (serve the folder over http,
for example `python -m http.server`, then browse to `/zstackHyper.html`).

## Try Linear Unmixing

1. Tick **Linear Unmixing** in the Post-Processing section.
2. Press **Load** and choose `demo_data/unmix_matrix.txt`.
3. Press **Apply**. The four blended colours separate into four components.

## Regenerate

    python make_demo_data.py            # writes ./demo_data (repository root)
    python make_demo_data.py --out DIR --size 1024

Requires numpy and pyvips. The output is deterministic.
