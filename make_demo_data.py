#!/usr/bin/env python3
"""
make_demo_data.py - build the small synthetic demo dataset shipped with hyperOSD.

Writes a 16-channel, 2-z-level RGBA Deep Zoom dataset plus a matching Linear
Unmixing matrix, so the viewer shows something useful straight out of the box:

    demo_data/
      deepzoom_RGBA/{0.0um,3.0um}/{1..4}/stitch.xml  (+ stitch_files/)
      unmix_matrix.txt                                (4 x 16, tab-separated)
      README.md

Content: four fluorophore-like components, each with a Gaussian emission
spectrum over the 16 channels. The spectra are wide enough that neighbouring
components bleed into each other (real crosstalk), so a raw 16-channel blend
shows mixed colours and Linear Unmixing visibly separates them. Spatially each
component is a few smooth blobs; the second z-level moves, rescales and
re-weights them. The data is noise-free and fully deterministic.

Usage:
    python make_demo_data.py [--out DIR] [--size PX]

Dependencies: numpy and pyvips (see generate_dzi.py). The tiles are written
with generate_dzi.py's own pack_channels / generate_tile_source, which sits
next to this script, so they get the standard 1024 px tile size, the
stitch.xml descriptor name and multiband tagging.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import generate_dzi  # noqa: E402  (needs HERE on sys.path)

DEFAULT_SIZE = 1536
N_CHANNELS = 16
Z_NAMES = ("0.0um", "3.0um")

# Emission peaks (0-based channel index) and spectral width (sigma, channels).
# Peaks 4 channels apart with sigma 2.5 give ~28% bleed into the next component.
PEAKS = (2.0, 6.0, 10.0, 14.0)
SPECTRAL_SIGMA = 2.5

# Spatial blobs per component and z-level, in unit-square coordinates:
# (cx, cy, sx, sy, weight). Defined per z so z=1 visibly differs from z=0.
BLOBS = {
    0: (  # z = 0.0um
        ((0.28, 0.30, 0.16, 0.10, 1.00), (0.40, 0.46, 0.07, 0.07, 0.70)),
        ((0.70, 0.28, 0.12, 0.15, 0.90), (0.52, 0.40, 0.08, 0.06, 0.60)),
        ((0.30, 0.72, 0.14, 0.12, 0.95), (0.47, 0.58, 0.07, 0.09, 0.65)),
        ((0.72, 0.70, 0.11, 0.14, 1.00), (0.56, 0.54, 0.06, 0.06, 0.75)),
    ),
    1: (  # z = 3.0um: shifted, rescaled, re-weighted
        ((0.34, 0.38, 0.10, 0.15, 0.80), (0.20, 0.22, 0.06, 0.06, 0.90)),
        ((0.62, 0.34, 0.16, 0.09, 1.00), (0.78, 0.50, 0.07, 0.07, 0.55)),
        ((0.26, 0.64, 0.10, 0.10, 0.70), (0.44, 0.72, 0.12, 0.07, 1.00)),
        ((0.66, 0.62, 0.15, 0.12, 0.85), (0.80, 0.78, 0.06, 0.08, 0.80)),
    ),
}
GAIN = 0.85  # overall brightness before 8-bit quantisation (values are clipped to 1)


def mixing_matrix():
    """(4, 16) float64: row k = emission spectrum of component k (peak 1.0)."""
    ch = np.arange(N_CHANNELS, dtype=np.float64)
    return np.stack([np.exp(-0.5 * ((ch - p) / SPECTRAL_SIGMA) ** 2) for p in PEAKS])


def unmix_matrix():
    """(4, 16) float64: the unmixing weights the viewer derives on Apply.

    The viewer's matrix file holds the MIXING matrix (one emission spectrum per
    row); on Apply it computes pinv(A^T) = (A A^T)^-1 A from it. Applied to the 16
    raw channel intensities this returns the four component abundances (to within
    8-bit quantisation). Used by the tests; not written to disk.
    """
    return np.linalg.pinv(mixing_matrix().T)


def abundance_maps(z_index, size):
    """(4, size, size) float32 component abundances for one z-level, in 0..1+."""
    coords = (np.arange(size, dtype=np.float64) + 0.5) / size
    maps = np.zeros((len(PEAKS), size, size), dtype=np.float32)
    for k, blobs in enumerate(BLOBS[z_index]):
        for cx, cy, sx, sy, w in blobs:
            gx = np.exp(-0.5 * ((coords - cx) / sx) ** 2)
            gy = np.exp(-0.5 * ((coords - cy) / sy) ** 2)
            maps[k] += (w * np.outer(gy, gx)).astype(np.float32)
    return maps


def render_channels(z_index, size):
    """(16, size, size) uint8: the 16 raw channel images of one z-level."""
    maps = abundance_maps(z_index, size)
    mix = mixing_matrix().astype(np.float32)
    out = np.empty((N_CHANNELS, size, size), dtype=np.uint8)
    for c in range(N_CHANNELS):
        plane = np.tensordot(mix[:, c], maps, axes=1) * GAIN
        out[c] = np.rint(np.clip(plane, 0.0, 1.0) * 255.0).astype(np.uint8)
    return out


def _to_vips(plane):
    pv = generate_dzi._require_pyvips()
    h, w = plane.shape
    return pv.Image.new_from_memory(np.ascontiguousarray(plane).tobytes(), w, h, 1, "uchar")


def write_dataset(out_dir, size=DEFAULT_SIZE):
    """Generate the whole dataset under out_dir. Returns out_dir."""
    generate_dzi._require_pyvips()
    out_dir = os.path.abspath(out_dir)
    dz_root = os.path.join(out_dir, "deepzoom_RGBA")
    for z_index, z_name in enumerate(Z_NAMES):
        z_dir = os.path.join(dz_root, z_name)
        os.makedirs(z_dir, exist_ok=True)
        planes = render_channels(z_index, size)
        for src in range(4):
            chans = [_to_vips(planes[src * 4 + i]) for i in range(4)]
            packed = generate_dzi.pack_channels(chans, "rgba")
            generate_dzi.generate_tile_source(packed, z_dir, src + 1, ".png")
    # vips can leave a vips-properties.xml next to the tiles; it carries no
    # information the viewer needs, so drop it.
    for root, _dirs, files in os.walk(dz_root):
        for name in files:
            if name == "vips-properties.xml":
                os.remove(os.path.join(root, name))

    with open(os.path.join(out_dir, "unmix_matrix.txt"), "w") as f:
        for row in mixing_matrix():
            f.write("\t".join("%.6f" % v for v in row) + "\n")
    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write(README_TEXT)
    return out_dir


README_TEXT = """# Demo dataset

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
"""


def default_out_dir():
    """<repo root>/demo_data. The script lives in dziGeneration/ (dev) or at the
    repo root (public release); both layouts are handled."""
    root = os.path.dirname(HERE) if os.path.basename(HERE) == "dziGeneration" else HERE
    return os.path.join(root, "demo_data")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate the synthetic hyperOSD demo dataset.")
    ap.add_argument("--out", default=None, help="output directory (default: <repo root>/demo_data)")
    ap.add_argument("--size", type=int, default=DEFAULT_SIZE,
                    help="image width/height in pixels (default: %d)" % DEFAULT_SIZE)
    args = ap.parse_args(argv)
    if args.size < 64:
        ap.error("--size must be at least 64")
    out = write_dataset(args.out or default_out_dir(), args.size)
    print("Demo dataset written to", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
