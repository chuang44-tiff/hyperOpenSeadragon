"""DZI generation pipeline and the Flask app helpers (public test subset).

Covers generate_dzi (8-bit conversion ladder, channel packing, descriptor layout,
coarse pyramid levels) and dzi_app (matrix validation, generated-viewer config
block, path containment). Runs the real load -> pack -> dzsave path on tiny
images and reads the written PNG tiles back.

Run: python -m unittest tests.test_dzi_pipeline          (public layout)
     python -m unittest tests.public.test_dzi_pipeline   (dev layout)
Needs pyvips + numpy (pipeline) and flask (app helpers). HYPEROSD_REQUIRE_ALL=1
turns a missing dependency into a failure instead of a skip.
"""
import contextlib
import io
import json
import os
import re
import shutil
import tempfile
import unittest
from pathlib import Path

from . import _layout

try:
    import numpy as np
    import pyvips
    pyvips.Image.black(1, 1)  # import can succeed with a broken libvips
    import generate_dzi
    _VIPS_SKIP = None
except Exception as exc:
    _VIPS_SKIP = "pipeline tests need pyvips (working libvips) and numpy (%s)" % exc

try:
    import dzi_app
    _APP_SKIP = None
except Exception as exc:
    _APP_SKIP = "app tests need flask (%s)" % exc

W, H = 64, 48            # dz pyramid: full-res level 6, coarse levels 5..0
FULL, COARSE = 6, 5


def _quiet():
    return contextlib.redirect_stdout(io.StringIO())


def _uchar_pixels(image):
    assert image.format == "uchar"
    mem = image.write_to_memory()
    if isinstance(mem, (bytes, bytearray)):
        return list(mem)
    return list(b"".join(mem))


class VipsCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _layout.require_deps(_VIPS_SKIP)


class To8BitTests(VipsCase):
    def test_uchar_passes_through(self):
        img = pyvips.Image.new_from_array([[0, 7], [128, 255]]).cast("uchar")
        self.assertEqual(_uchar_pixels(generate_dzi.to_8bit(img)), [0, 7, 128, 255])

    def test_ushort_divides_by_256_and_truncates(self):
        # 384 -> 1 (not 2), 510 -> 1, 255 -> 0, 1000 -> 3: separates /256 from /255 and
        # truncation from rounding.
        img = pyvips.Image.new_from_array([[384, 510], [255, 1000]]).cast("ushort")
        self.assertEqual(_uchar_pixels(generate_dzi.to_8bit(img)), [1, 1, 0, 3])

    def test_float_is_scaled_by_255_and_clamped(self):
        img = pyvips.Image.new_from_array([[0.5, 1.0], [0.25, 0.0]]).cast("float")
        self.assertEqual(_uchar_pixels(generate_dzi.to_8bit(img)), [127, 255, 63, 0])
        over = pyvips.Image.new_from_array([[2.0, -1.0]]).cast("float")
        self.assertEqual(_uchar_pixels(generate_dzi.to_8bit(over)), [255, 0])

    def test_signed_short_takes_the_direct_cast_not_the_ushort_divide(self):
        # Direct cast clamps (512 -> 255, -512 -> 0); the /256 branch would give [2, 0, 1, 0].
        img = pyvips.Image.new_from_array([[512, -512], [300, 7]]).cast("short")
        self.assertEqual(_uchar_pixels(generate_dzi.to_8bit(img)), [255, 0, 255, 7])


class PackedPyramidTests(VipsCase):
    """pack_channels + process_z_level + dzsave, read back from the written PNG tiles."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    @staticmethod
    def _const(value):
        return pyvips.Image.black(W, H).cast("uchar") + value

    def _write_channels(self, images):
        paths = []
        for i, img in enumerate(images):
            p = os.path.join(self.tmp, "c%02d.tif" % i)
            img.cast("uchar").write_to_file(p)
            paths.append(p)
        return paths

    def _generate(self, images):
        out = os.path.join(self.tmp, "out")
        with _quiet():
            generate_dzi.process_z_level("0.0um", self._write_channels(images), out, "rgba")
        return os.path.join(out, "0.0um")

    @staticmethod
    def _tile(z_dir, source, level):
        return pyvips.Image.new_from_file(
            os.path.join(z_dir, str(source), "stitch_files", str(level), "0_0.png"))

    @staticmethod
    def _means(tile, left=None, top=0, width=None, height=None):
        if left is not None:
            tile = tile.crop(left, top, width, height)
        return [round(tile.extract_band(b).avg()) for b in range(tile.bands)]

    def test_pack_channels_tags_multiband_and_zero_fills(self):
        chans = [self._const(v) for v in (10, 20)]
        packed = generate_dzi.pack_channels(chans, "rgba")
        self.assertEqual(packed.bands, 4)
        self.assertEqual(packed.interpretation, "multiband")
        self.assertEqual(self._means(packed), [10, 20, 0, 0])

    def test_pack_channels_rejects_too_many_and_mismatched_sizes(self):
        with self.assertRaises(ValueError):
            generate_dzi.pack_channels([self._const(1)] * 5, "rgba")
        small = pyvips.Image.black(W // 2, H).cast("uchar")
        with self.assertRaises(ValueError):
            generate_dzi.pack_channels([self._const(1), small], "rgba")

    def test_four_channels_make_one_source_in_channel_order(self):
        z_dir = self._generate([self._const(v) for v in (10, 20, 30, 40)])
        self.assertEqual(sorted(os.listdir(z_dir)), ["1"])
        tile = self._tile(z_dir, 1, FULL)
        self.assertEqual(tile.bands, 4)
        self.assertEqual(self._means(tile), [10, 20, 30, 40])

    def test_five_channels_make_two_sources_with_zero_filled_slots(self):
        z_dir = self._generate([self._const(v) for v in (10, 20, 30, 40, 99)])
        self.assertEqual(sorted(os.listdir(z_dir)), ["1", "2"])
        self.assertEqual(self._means(self._tile(z_dir, 1, FULL)), [10, 20, 30, 40])
        self.assertEqual(self._means(self._tile(z_dir, 2, FULL)), [99, 0, 0, 0])

    def test_six_channels_make_two_sources_and_coarse_level_keeps_the_partial_one(self):
        # Source 2 holds (50, 60, 0, 0). Read as colour + alpha, every coarse level was black.
        z_dir = self._generate([self._const(v) for v in (10, 20, 30, 40, 50, 60)])
        self.assertEqual(sorted(os.listdir(z_dir)), ["1", "2"])
        self.assertEqual(self._means(self._tile(z_dir, 2, FULL)), [50, 60, 0, 0])
        self.assertEqual(self._means(self._tile(z_dir, 2, COARSE)), [50, 60, 0, 0])

    def test_sparse_fourth_channel_does_not_black_out_coarse_levels(self):
        # Regression guard: ch4 is 0 on the left half. If the packed image were tagged with
        # an alpha band, the coarse level of ch1-3 would go black wherever ch4 is 0.
        ch4 = self._const(200).insert(pyvips.Image.black(W // 2, H).cast("uchar"), 0, 0)
        z_dir = self._generate([self._const(90), self._const(150), self._const(210), ch4])
        coarse = self._tile(z_dir, 1, COARSE)          # 32 x 24
        self.assertEqual(self._means(coarse, 0, 0, 12, 24), [90, 150, 210, 0])
        self.assertEqual(self._means(coarse, 20, 0, 12, 24), [90, 150, 210, 200])

    def test_coarse_levels_are_a_plain_mean_not_weighted_by_band_4(self):
        # ch1 alternates 200/0 while ch4 alternates 255/1: the true 2x2 mean of ch1 is 100;
        # an alpha-weighted shrink gives ~199.
        xy = pyvips.Image.xyz(W, H)
        checker = (xy[0] + xy[1]) % 2
        ch1 = (checker == 0).ifthenelse(200, 0)
        ch4 = (checker == 0).ifthenelse(255, 1)
        z_dir = self._generate([ch1, self._const(0), self._const(0), ch4])
        self.assertAlmostEqual(self._means(self._tile(z_dir, 1, COARSE))[0], 100, delta=2)

    def test_descriptor_layout(self):
        z_dir = self._generate([self._const(5)])
        names = {p.name for p in Path(z_dir).rglob("*") if p.is_file()}
        self.assertIn("stitch.xml", names)
        self.assertFalse(any(n.endswith(".dzi") for n in names))
        # (libvips may leave a vips-properties.xml beside the tiles; that is harmless here.)
        xml = (Path(z_dir) / "1" / "stitch.xml").read_text()
        self.assertIn('TileSize="1024"', xml)
        self.assertIn('Format="png"', xml)
        self.assertIn('Width="%d"' % W, xml)
        self.assertIn('Height="%d"' % H, xml)


class ValidateUnmixMatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _layout.require_deps(_APP_SKIP)

    def test_valid_matrix_returns_none(self):
        text = "\n".join("\t".join(["0.5"] * 16) for _ in range(4))
        self.assertIsNone(dzi_app.validate_unmix_matrix(text, 16))

    def test_comma_and_space_separators_are_accepted(self):
        self.assertIsNone(dzi_app.validate_unmix_matrix("1,2,3\n4 5 6", 3))

    def test_nan_and_inf_are_rejected(self):
        for bad in ("1\t2\nnan\t4", "1\tinf\n3\t4", "1\t2\n-inf\t4"):
            err = dzi_app.validate_unmix_matrix(bad, 2)
            self.assertIsInstance(err, str, bad)
            self.assertIn("NaN or infinite", err)

    def test_ragged_rows_are_rejected(self):
        err = dzi_app.validate_unmix_matrix("1\t2\t3\n4\t5", 3)
        self.assertIn("inconsistent column counts", err)

    def test_wrong_width_empty_and_non_numeric(self):
        self.assertIn("2 columns but dataset has 3", dzi_app.validate_unmix_matrix("1\t2", 3))
        self.assertIn("empty", dzi_app.validate_unmix_matrix("  \n\n", 3))
        self.assertIn("non-numeric", dzi_app.validate_unmix_matrix("1\tx", 2))

    def test_row_count_limits(self):
        nine = "\n".join("1\t2" for _ in range(9))
        self.assertIn("1 to 8", dzi_app.validate_unmix_matrix(nine, 2))
        eight = "\n".join("1\t2" for _ in range(8))
        self.assertIsNone(dzi_app.validate_unmix_matrix(eight, 2))


def _config_block(html):
    m = re.search(r"// ===== DATASET CONFIGURATION =====(.*?)// ===== END DATASET CONFIGURATION =====",
                  html, re.S)
    assert m, "no DATASET CONFIGURATION block"
    return m.group(1)


def _image_sources(block):
    start = block.index("var imageSources = [")
    body_start = block.index("\n", start)         # skip the trailing comment on that line
    end = block.index("\n\t\t];", body_start)
    return json.loads("[" + block[body_start:end].strip().rstrip(",") + "]")


class GenerateViewerHtmlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _layout.require_deps(_APP_SKIP)

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="hyperosd-viewer-test-"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _generate(self, n_z, per_z, is_rgb, channels=None):
        zs = [{"folder": "%.1fum" % (z * 3), "name": "%.1fum" % (z * 3)} for z in range(n_z)]
        sources = ["deepzoom_RGBA/%s/%d/stitch.xml" % (z["name"], i + 1) for z in zs for i in range(per_z)]
        path = dzi_app._generate_viewer_html(
            str(self.tmp), "My Set", zs, sources, images_per_z=per_z,
            is_grayscale=False, is_rgb=is_rgb, actual_channel_count=channels, um_per_pixel=0.5)
        return Path(path).read_text(), sources

    def test_rgba_16_channels(self):
        html, sources = self._generate(2, 4, is_rgb=False)
        block = _config_block(html)
        self.assertIn("var channelsPerImage = 4;", block)
        self.assertIn("var zLevelCount = 2;", block)
        self.assertIn("var defaultChannels = %s;" % json.dumps(list(range(1, 17))), block)
        self.assertEqual(_image_sources(block), sources)
        self.assertEqual(len(_image_sources(block)), 8)
        self.assertIn("var matrixFileName = 'unmix_matrix.txt';", block)

    def test_rgb_12_channels(self):
        html, sources = self._generate(1, 4, is_rgb=True)
        block = _config_block(html)
        self.assertIn("var channelsPerImage = 3;", block)
        self.assertIn("var defaultChannels = %s;" % json.dumps(list(range(1, 13))), block)
        self.assertEqual(len(_image_sources(block)), 4)
        self.assertIn("var matrixFileName = 'unmix_matrix.txt';", block)

    def test_generated_viewer_keeps_the_rest_of_the_template(self):
        html, _ = self._generate(1, 1, is_rgb=False)
        template = _layout.TEMPLATE.read_text()
        self.assertEqual(html.count("// ===== DATASET CONFIGURATION ====="), 1)
        # Everything outside the config block (and the title / reinhard tweaks) is the template.
        self.assertIn("openseadragon/hyperblend-webgl.js", html)
        self.assertEqual(html.count("ANNOTATION-WIRING-START"), template.count("ANNOTATION-WIRING-START"))

    def test_hostile_source_cannot_break_out_of_the_js_string(self):
        zs = [{"folder": "0.0um", "name": "0.0um"}]
        evil = 'a"];alert(1);//</script>'
        path = dzi_app._generate_viewer_html(
            str(self.tmp), "x", zs, [evil], images_per_z=1,
            is_grayscale=False, is_rgb=False, actual_channel_count=None, um_per_pixel=None)
        block = _config_block(Path(path).read_text())
        self.assertEqual(_image_sources(block), [evil])
        self.assertNotIn("</script>", block)


class IsContainedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _layout.require_deps(_APP_SKIP)

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="hyperosd-contain-test-")).resolve()
        self.root = self.tmp / "root"
        (self.root / "sub").mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_inside_and_root_itself(self):
        self.assertTrue(dzi_app._is_contained(str(self.root / "sub" / "f.txt"), str(self.root)))
        self.assertTrue(dzi_app._is_contained(str(self.root), str(self.root)))

    def test_dotdot_absolute_and_sibling_prefix_escape(self):
        self.assertFalse(dzi_app._is_contained(os.path.join(str(self.root), "..", "x"), str(self.root)))
        self.assertFalse(dzi_app._is_contained("/etc/passwd", str(self.root)))
        sibling = str(self.root) + "EVIL"
        os.makedirs(sibling)
        self.assertFalse(dzi_app._is_contained(os.path.join(sibling, "f"), str(self.root)))

    def test_escaping_symlink_is_not_contained(self):
        outside = self.tmp / "outside.txt"
        outside.write_text("secret")
        link = self.root / "link"
        try:
            os.symlink(str(outside), str(link))
        except (OSError, NotImplementedError):
            self.skipTest("symlinks unavailable")
        self.assertFalse(dzi_app._is_contained(str(link), str(self.root)))


class FlaskRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _layout.require_deps(_APP_SKIP)

    def setUp(self):
        dzi_app.app.config["TESTING"] = True
        self.client = dzi_app.app.test_client()
        self.tmp = Path(tempfile.mkdtemp(prefix="hyperosd-route-test-")).resolve()
        self.root = self.tmp / "dataset"
        self.root.mkdir()
        (self.root / "ok.txt").write_text("public content")
        (self.tmp / "secret.txt").write_text("TOP-SECRET")
        with dzi_app._state_lock:
            self._saved = (dzi_app._dataset_root, dzi_app._osd_serve_dir)
            dzi_app._dataset_root = str(self.root)
            dzi_app._osd_serve_dir = None

    def tearDown(self):
        with dzi_app._state_lock:
            dzi_app._dataset_root, dzi_app._osd_serve_dir = self._saved
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_scan_rejects_empty_and_missing_paths(self):
        for path in ("", str(self.tmp / "does-not-exist")):
            resp = self.client.post("/scan", json={"path": path})
            self.assertEqual(resp.status_code, 400, path)
            self.assertIn("error", resp.get_json())

    def test_dataset_serves_a_file_inside_the_root(self):
        resp = self.client.get("/dataset/ok.txt")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"public content", resp.data)
        resp.close()

    def test_dataset_traversal_is_refused(self):
        for url in ("/dataset/../secret.txt", "/dataset/%2e%2e/secret.txt",
                    "/dataset/..%2fsecret.txt", "/dataset/sub/../../secret.txt"):
            resp = self.client.get(url)
            self.assertIn(resp.status_code, (400, 403, 404), url)
            self.assertNotIn(b"TOP-SECRET", resp.data, url)
            resp.close()

    def test_dataset_escaping_symlink_is_refused(self):
        try:
            os.symlink(str(self.tmp / "secret.txt"), str(self.root / "shadow"))
        except (OSError, NotImplementedError):
            self.skipTest("symlinks unavailable")
        resp = self.client.get("/dataset/shadow")
        self.assertEqual(resp.status_code, 403)
        self.assertNotIn(b"TOP-SECRET", resp.data)

    def test_view_missing_viewer_is_404_and_html_escaped(self):
        resp = self.client.get("/view/%3Cscript%3Ex%3C%2Fscript%3E.html")
        self.assertEqual(resp.status_code, 404)
        self.assertNotIn(b"<script>x", resp.data)


if __name__ == "__main__":
    unittest.main()
