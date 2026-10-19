"""Pins the synthetic demo dataset generator (make_demo_data.py) and the committed
demo_data/ it produced.

Tile PNG bytes are never compared (libvips / zlib differ across platforms); the
finest pyramid level is decoded (every tile) and compared pixel-for-pixel with a fresh numpy
render from the script. A fresh regeneration on the same machine must match EXACTLY. The
committed tiles were written on another platform (macOS arm64), where numpy's exp() and
rounding may differ from Linux x86 by one least-significant bit, so the committed-vs-render
comparison allows |diff| <= 1 on at most 0.1 % of the pixels of each tile source and fails
on anything larger.

Run: python -m unittest tests.test_demo_data          (public layout)
     python -m unittest tests.public.test_demo_data   (dev layout)
Needs numpy and pyvips; set HYPEROSD_REQUIRE_ALL=1 to fail instead of skip without them.
"""
import math
import re
import tempfile
import unittest
from pathlib import Path

from . import _layout

try:
    import numpy as np
    import pyvips
    pyvips.Image.black(1, 1)  # import can succeed with a broken libvips
    import make_demo_data as mdd
    _SKIP = None
except Exception as exc:  # pyvips / numpy missing
    _SKIP = "demo-data tests need numpy and pyvips (%s)" % exc

COMMITTED = _layout.DEMO_DATA
SMALL = 1100  # regeneration size for the test: 2x2 finest tiles, still fast


def _decode(path):
    img = pyvips.Image.new_from_file(str(path))
    arr = np.ndarray(buffer=img.write_to_memory(), dtype=np.uint8,
                     shape=(img.height, img.width, img.bands))
    return arr


def _level_image(source_dir, level, size):
    """Stitch every tile of one pyramid level back into a full image (tiles carry
    the descriptor's Overlap pixels on each interior edge)."""
    sdir = Path(source_dir)
    ov = int(re.search(r'Overlap="(\d+)"', (sdir / "stitch.xml").read_text()).group(1))
    side = -(-size // 2 ** (math.ceil(math.log2(size)) - level))  # ceil
    n = -(-side // 1024)
    out = np.zeros((side, side, 4), dtype=np.uint8)
    for y in range(n):
        for x in range(n):
            t = _decode(sdir / "stitch_files" / str(level) / ("%d_%d.png" % (x, y)))
            t = t[ov if y else 0:, ov if x else 0:]
            h, w = min(1024, side - y * 1024), min(1024, side - x * 1024)
            out[y * 1024:y * 1024 + h, x * 1024:x * 1024 + w] = t[:h, :w]
    return out


def _check_against_render(testcase, root, size, tolerant=False):
    """Finest level (every tile) of every source vs the numpy render; level finest-2 vs
    a 4x4 box mean of the render; a coarse level must keep every non-empty channel
    non-empty (packed channels must not go black on zoom-out).

    tolerant=False: exact. tolerant=True: |diff| <= 1 everywhere and at most 0.1 % of
    the pixels of a source differ at all (cross-platform float rounding)."""
    finest = math.ceil(math.log2(size))
    coarse = max(finest - 3, 0)
    for z_index, z_name in enumerate(mdd.Z_NAMES):
        planes = mdd.render_channels(z_index, size)
        for src in range(4):
            sdir = Path(root) / "deepzoom_RGBA" / z_name / str(src + 1)
            img = _level_image(sdir, finest, size)
            testcase.assertEqual(img.shape, (size, size, 4))
            want = np.stack([planes[src * 4 + i] for i in range(4)], axis=-1)
            label = "z=%s src=%d" % (z_name, src + 1)
            if tolerant:
                diff = np.abs(img.astype(np.int16) - want.astype(np.int16))
                testcase.assertLessEqual(int(diff.max()), 1, "pixel off by more than 1 LSB, " + label)
                frac = float((diff > 0).sum()) / diff.size
                testcase.assertLessEqual(frac, 0.001, "%.4f%% of pixels differ, %s" % (frac * 100, label))
            else:
                testcase.assertTrue(np.array_equal(img, want), "pixel mismatch " + label)
            # One coarser level, pixel for pixel: libvips halves each level with a rounded
            # 2x2 mean, so level finest-2 is a 4x4 box mean of the render within ~1 LSB
            # (measured <= 1.0). A tile shifted by one pixel or the wrong z plane differs by
            # 4 or more. The last partial row/column depends on edge handling and is skipped.
            m = size // 4
            mid = _level_image(sdir, finest - 2, size)[:m, :m].astype(np.float64)
            box = want[:m * 4, :m * 4].astype(np.float64).reshape(m, 4, m, 4, 4).mean(axis=(1, 3))
            testcase.assertLessEqual(float(np.abs(mid - box).max()), 2.0,
                                     "level %d differs from a 4x4 mean of the render, %s" % (finest - 2, label))
            small = _level_image(sdir, coarse, size)
            for i in range(4):
                if want[..., i].max() > 0:
                    testcase.assertGreater(int(small[..., i].max()), 0,
                        "coarse level %d channel %d black z=%s src=%d"
                        % (coarse, i, z_name, src + 1))


def _check_all_tiles_present(testcase, xml_path, size):
    """Every tile of every pyramid level exists and is a non-empty file (a tile missing
    from a level the pixel checks do not read would otherwise go unnoticed)."""
    sdir = Path(xml_path).parent / "stitch_files"
    finest = math.ceil(math.log2(size))
    for level in range(finest + 1):
        side = -(-size // 2 ** (finest - level))
        n = -(-side // 1024)
        for y in range(n):
            for x in range(n):
                tile = sdir / str(level) / ("%d_%d.png" % (x, y))
                testcase.assertTrue(tile.is_file() and tile.stat().st_size > 0, "missing tile %s" % tile)


def _check_descriptor(testcase, xml_path, size):
    text = Path(xml_path).read_text()
    testcase.assertIn('TileSize="1024"', text)
    testcase.assertIn('Format="png"', text)
    testcase.assertIn('Width="%d"' % size, text)
    testcase.assertIn('Height="%d"' % size, text)
    levels = sorted(int(p.name) for p in Path(xml_path).parent.joinpath("stitch_files").iterdir())
    testcase.assertEqual(levels, list(range(math.ceil(math.log2(size)) + 1)))
    _check_all_tiles_present(testcase, xml_path, size)


class TestGeneratorRegeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _layout.require_deps(_SKIP)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name) / "demo"
        mdd.write_dataset(cls.out, SMALL)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_pixels_match_numpy_render(self):
        _check_against_render(self, self.out, SMALL)

    def test_z_levels_differ(self):
        a = mdd.render_channels(0, 64)
        b = mdd.render_channels(1, 64)
        self.assertGreater(int(np.abs(a.astype(int) - b.astype(int)).max()), 40)

    def test_descriptors(self):
        for z in mdd.Z_NAMES:
            for src in "1234":
                _check_descriptor(self, self.out / "deepzoom_RGBA" / z / src / "stitch.xml", SMALL)

    def test_no_dzi_or_vips_properties_left(self):
        names = {p.name for p in self.out.rglob("*") if p.is_file()}
        self.assertNotIn("vips-properties.xml", names)
        self.assertFalse(any(n.endswith(".dzi") for n in names))

    def test_matrix_file(self):
        _check_matrix(self, self.out / "unmix_matrix.txt")

    def test_matrix_unmixes(self):
        # Mirror the viewer's Apply: it reads the file as A (4 x 16 spectra) and
        # unmixes with pinv(A^T) = (A A^T)^-1 A. That must recover the abundances.
        A = np.array([[float(v) for v in r.split("\t")] for r in
                      (self.out / "unmix_matrix.txt").read_text().strip().split("\n")])
        W = np.linalg.inv(A @ A.T) @ A
        self.assertTrue(np.allclose(W, mdd.unmix_matrix(), atol=1e-5))
        maps = mdd.abundance_maps(0, 64).reshape(4, -1)
        chans = mdd.mixing_matrix().T @ maps
        self.assertTrue(np.allclose(W @ chans, maps, atol=1e-3))


def _check_matrix(testcase, path):
    rows = [r for r in Path(path).read_text().strip().split("\n") if r.strip()]
    testcase.assertEqual(len(rows), 4)
    for r in rows:
        vals = [float(v) for v in re.split(r"[\t,\s]+", r.strip())]
        testcase.assertEqual(len(vals), 16)
        testcase.assertTrue(all(math.isfinite(v) for v in vals))
        testcase.assertIn("\t", r)


class TestCommittedDemoData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _layout.require_deps(_SKIP)

    def setUp(self):
        if not COMMITTED.is_dir():
            self.fail("demo_data/ is missing; regenerate it with make_demo_data.py")
        self.size = int(re.search(r'Width="(\d+)"', (
            COMMITTED / "deepzoom_RGBA" / "0.0um" / "1" / "stitch.xml").read_text()).group(1))

    def test_descriptors(self):
        for z in mdd.Z_NAMES:
            for src in "1234":
                _check_descriptor(self, COMMITTED / "deepzoom_RGBA" / z / src / "stitch.xml", self.size)

    def test_pixels_match_numpy_render(self):
        _check_against_render(self, COMMITTED, self.size, tolerant=True)

    def test_matrix_file(self):
        _check_matrix(self, COMMITTED / "unmix_matrix.txt")

    def test_matrix_matches_generator(self):
        # The viewer expects the MIXING matrix (spectra) and inverts it on Apply.
        want = mdd.mixing_matrix()
        got = np.array([[float(v) for v in r.split("\t")] for r in
                        (COMMITTED / "unmix_matrix.txt").read_text().strip().split("\n")])
        self.assertTrue(np.allclose(got, want, atol=1e-6))

    def test_size_budget(self):
        total = sum(p.stat().st_size for p in COMMITTED.rglob("*") if p.is_file())
        self.assertLess(total, 10.5 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
