"""Browser smoke test of the viewer on the committed demo dataset.

Opens ``zstackHyper.html`` in headless Chromium (software GL, so it behaves the same on a
laptop and on a GPU-less CI runner) and checks that the WebGL HyperBlend drawer starts,
renders the demo data, switches z-level, and that Linear Unmixing separates the four
blobs of the demo dataset.

Run: python -m unittest tests.test_viewer_smoke          (public layout)
     python -m unittest tests.public.test_viewer_smoke   (dev layout)
Needs playwright and its Chromium (``python -m playwright install chromium``).
HYPEROSD_REQUIRE_ALL=1 turns a missing dependency into a failure instead of a skip.
"""
import time
import unittest

from . import _layout

try:
    from playwright.sync_api import sync_playwright
    _SKIP = None
except Exception as exc:
    _SKIP = "viewer smoke test needs playwright (%s)" % exc

# Software GL, unconditionally: the same renderer on a Mac and on Linux CI.
CHROMIUM_ARGS = [
    "--use-angle=swiftshader",
    "--use-gl=angle",
    "--enable-unsafe-swiftshader",
    "--ignore-gpu-blocklist",
]
TIMEOUT_MS = 60000

# Fractions of the canvas at which the four demo blobs sit (home view, 1400x900 viewport).
BLOB_CENTRES = [[0.3, 0.25], [0.68, 0.25], [0.3, 0.7], [0.68, 0.7]]

ALL_LOADED_JS = """(z) => {
    const row = window.allLayers && window.allLayers[z];
    return !!row && row.length === 4 && row.every(it => it && it.getFullyLoaded());
}"""

# Mean colour of an 8x8 downsample of the current frame (a cheap image signature).
SIGNATURE_JS = """() => {
    const src = viewer.drawer.captureCanvas();
    const c = document.createElement('canvas'); c.width = 8; c.height = 8;
    const x = c.getContext('2d'); x.drawImage(src, 0, 0, 8, 8);
    return Array.from(x.getImageData(0, 0, 8, 8).data);
}"""

SAMPLE_JS = """(pts) => {
    const c = viewer.drawer.captureCanvas(); const x = c.getContext('2d');
    return pts.map(([fx, fy]) => {
        const d = x.getImageData(Math.round(fx * c.width), Math.round(fy * c.height), 1, 1).data;
        return [d[0], d[1], d[2]];
    });
}"""


def _poll(fn, timeout_s=60.0, interval_s=0.25):
    """Call fn until it returns a truthy value (returned) or the timeout expires (returns last value)."""
    deadline = time.monotonic() + timeout_s
    value = fn()
    while not value and time.monotonic() < deadline:
        time.sleep(interval_s)
        value = fn()
    return value


class ViewerSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _layout.require_deps(_SKIP)
        cls._server_cm = _layout.static_server()
        cls.base = cls._server_cm.__enter__()
        cls._pw = sync_playwright().start()
        try:
            cls.browser = cls._pw.chromium.launch(args=CHROMIUM_ARGS)
        except Exception as exc:
            cls._server_cm.__exit__(None, None, None)
            cls._pw.stop()
            _layout.require_deps("Chromium could not be launched (%s); run: python -m playwright install chromium" % exc)
            raise

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls._pw.stop()
        cls._server_cm.__exit__(None, None, None)

    def setUp(self):
        self.page = self.browser.new_page(viewport={"width": 1400, "height": 900})
        self.addCleanup(self.page.close)
        self.page.set_default_timeout(TIMEOUT_MS)
        self.errors = []
        self.page.on("pageerror", lambda e: self.errors.append(str(e)))
        self.page.goto(self.base + "/zstackHyper.html")
        # The drawer type is asserted before any pixel check: a silent fallback to the canvas
        # drawer must fail here, not pass by rendering something.
        self.page.wait_for_function(
            "() => window.viewer && viewer.drawer && typeof viewer.drawer.getType === 'function'")
        self.assertEqual(self.page.evaluate("() => viewer.drawer.getType()"), "hyperblend-webgl")
        self.page.wait_for_function(ALL_LOADED_JS, arg=0)

    def _signature(self):
        return self.page.evaluate(SIGNATURE_JS)

    def _wait_for_nonblack(self):
        def check():
            sig = self._signature()
            return sig if max(sig) > 30 else None
        sig = _poll(check)
        self.assertIsNotNone(sig, "canvas stayed black")
        return sig

    def test_drawer_channels_and_first_frame(self):
        self.assertEqual(
            self.page.evaluate("() => $('#csDropdownMenu .cs-dropdown-item').not('.cs-all-toggle').length"), 16)
        self.assertEqual(self.page.inner_text("#csChannelCount").strip(), "4 / 16 active")
        self._wait_for_nonblack()
        self.assertEqual(self.errors, [])

    def test_z_switch_changes_the_image(self):
        z0 = self._wait_for_nonblack()
        handle = self.page.locator("#custom-zhandle")
        handle.wait_for(state="attached")
        handle.focus()
        self.page.keyboard.press("ArrowRight")
        try:
            self.page.wait_for_function("() => window.currentZIndex === 1", timeout=8000)
        except Exception:
            # Keyboard focus is the preferred path; fall back to the slider's own `stop` callback.
            self.page.evaluate("""() => {
                const s = window.$('#zlevel'); const stop = s.slider('option', 'stop');
                if (stop) stop.call(s, null, { value: 1 });
            }""")
            self.page.wait_for_function("() => window.currentZIndex === 1")
        self.page.wait_for_function(ALL_LOADED_JS, arg=1)

        def changed():
            sig = self._signature()
            diff = max(abs(a - b) for a, b in zip(sig, z0))
            return (sig, diff) if diff > 20 and max(sig) > 30 else None
        res = _poll(changed)
        self.assertIsNotNone(res, "frame at z=3 um never differed from z=0 (or stayed black)")
        self.assertEqual(self.errors, [])

    def test_linear_unmixing_separates_the_four_blobs(self):
        self._wait_for_nonblack()
        page = self.page
        page.click("#unmixToggle")
        page.set_input_files("#unmixLoadFile", str(_layout.DEMO_DATA / "unmix_matrix.txt"))
        page.wait_for_function("() => $('#unmixMatrixTable tr').length >= 4")
        page.click("#unmixApply")
        page.wait_for_function("() => viewer.drawer._unmixEnabled === true")
        self.assertEqual(page.evaluate("() => viewer.drawer._numOutputs"), 4)
        self.assertIn("4×16 matrix applied (pinv)", page.inner_text("#unmixStatus"))
        self.assertTrue(page.is_visible("#unmixPinvTag"))

        def separated():
            pts = page.evaluate(SAMPLE_JS, BLOB_CENTRES)
            return pts if all(max(p) >= 120 and min(p) <= 25 for p in pts) else None
        pts = _poll(separated)
        pts = pts or page.evaluate(SAMPLE_JS, BLOB_CENTRES)
        for p in pts:
            # One output per blob: a pure hue (S = 1) has a near-zero minimum channel; a blend
            # of several outputs, as before unmixing, does not.
            self.assertGreaterEqual(max(p), 120, pts)
            self.assertLessEqual(min(p), 25, pts)
        for i in range(4):
            for j in range(i + 1, 4):
                dist = sum((a - b) ** 2 for a, b in zip(pts[i], pts[j])) ** 0.5
                self.assertGreater(dist, 100, "blobs %d and %d not separated: %s" % (i, j, pts))
        self.assertEqual(self.errors, [])


if __name__ == "__main__":
    unittest.main()
