"""Browser smoke test of the viewer on the committed demo dataset.

Opens ``zstackHyper.html`` in headless Chromium (software GL, so it behaves the same on a
laptop and on a GPU-less CI runner) and checks that the WebGL HyperBlend drawer starts,
renders the demo data, switches z-level, that Linear Unmixing separates the four
blobs of the demo dataset, and that the H&E / Masson Trichrome post-process renders.

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

# Mean colour of an 8x8 downsample of the current frame (a cheap image signature), RGBA.
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


# Mean R, G, B over the whole current frame.
MEAN_RGB_JS = """() => {
    const c = viewer.drawer.captureCanvas();
    const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
    const s = [0, 0, 0];
    for (let i = 0; i < d.length; i += 4) { s[0] += d[i]; s[1] += d[i + 1]; s[2] += d[i + 2]; }
    const n = d.length / 4;
    return s.map(v => v / n);
}"""


def _rgb_max(sig):
    """Brightest colour value of a signature, ignoring alpha (always 255 on this canvas)."""
    return max(v for i, v in enumerate(sig) if i % 4 != 3)


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
        # Class cleanups run (in reverse) even when a later step here raises, so the server
        # and Playwright are always released.
        server = _layout.static_server()
        cls.base = server.__enter__()
        cls.addClassCleanup(server.__exit__, None, None, None)
        pw = sync_playwright().start()
        cls.addClassCleanup(pw.stop)
        try:
            cls.browser = pw.chromium.launch(args=CHROMIUM_ARGS)
        except Exception as exc:
            _layout.require_deps("Chromium could not be launched (%s); run: python -m playwright install chromium" % exc)
            raise
        cls.addClassCleanup(cls.browser.close)

    def setUp(self):
        self.page = self.browser.new_page(viewport={"width": 1400, "height": 900})
        self.addCleanup(self.page.close)
        self.page.set_default_timeout(TIMEOUT_MS)
        self.errors = []
        self.page.on("pageerror", lambda e: self.errors.append(str(e)))
        self.failed = []
        self.page.on("response", lambda r: self.failed.append((r.status, r.url)) if r.status >= 400 else None)
        self.page.on("requestfailed", lambda r: self.failed.append(("failed", r.url)))
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
            return sig if _rgb_max(sig) > 30 else None
        sig = _poll(check)
        self.assertIsNotNone(sig, "canvas stayed black")
        return sig

    def _settled_signature(self):
        """A non-black frame that no longer changes: two samples 0.5 s apart agree, so
        coarse-to-fine refinement has finished."""
        state = {"prev": self._wait_for_nonblack()}

        def check():
            sig = self._signature()
            same = _rgb_max(sig) > 30 and sig == state["prev"]
            state["prev"] = sig
            return sig if same else None
        sig = _poll(check, interval_s=0.5)
        self.assertIsNotNone(sig, "frame never settled")
        return sig

    def test_drawer_channels_and_first_frame(self):
        self.assertEqual(
            self.page.evaluate("() => $('#csDropdownMenu .cs-dropdown-item').not('.cs-all-toggle').length"), 16)
        self.assertEqual(self.page.inner_text("#csChannelCount").strip(), "4 / 16 active")
        self._wait_for_nonblack()
        self.assertEqual(self.errors, [])

    def test_no_failed_requests_on_a_static_host(self):
        # A plain static host has no optimizer server: the page must not probe /health
        # (or request any other missing file), so the console stays free of 404s.
        self._settled_signature()
        self.page.wait_for_function("() => /in-browser optimizer/.test($('#hpDepRows').text())")
        self.assertEqual(self.failed, [])
        self.assertEqual(self.errors, [])

    def test_z_switch_changes_the_image(self):
        z0 = self._settled_signature()
        handle = self.page.locator("#custom-zhandle")
        handle.wait_for(state="attached")
        handle.focus()
        # The real user path, with no fallback: if the slider stops reacting to the
        # keyboard, this fails.
        self.page.keyboard.press("ArrowRight")
        self.page.wait_for_function("() => window.currentZIndex === 1", timeout=8000)
        self.page.wait_for_function(ALL_LOADED_JS, arg=1)
        z1 = self._settled_signature()
        diff = max(abs(a - b) for a, b in zip(z1, z0))
        self.assertGreater(diff, 20, "settled frame at z=3 um does not differ from z=0")
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

    def test_he_post_process_renders(self):
        page = self.page
        self._settled_signature()
        off = page.evaluate(MEAN_RGB_JS)
        page.click("#heToggle")
        page.wait_for_function("() => viewer.drawer._postProcessConfig.active === true")
        self.assertEqual(page.evaluate("() => viewer.drawer.getType()"), "hyperblend-webgl")
        self.assertEqual(page.evaluate("() => viewer.drawer._postProcessConfig.mode"), 0)
        on = page.evaluate(MEAN_RGB_JS)
        # Beer's law turns the dark fluorescence background into a bright, stain-coloured
        # (hematoxylin-blue / eosin-pink) image.
        self.assertGreater(sum(on) / 3 - sum(off) / 3, 50, (off, on))
        self.assertGreater(on[2], on[1], on)
        page.click("#trichromeToggle")
        page.wait_for_function("() => viewer.drawer._postProcessConfig.mode === 1")
        self.assertTrue(page.evaluate("() => viewer.drawer._postProcessConfig.active"))
        self.assertFalse(page.is_checked("#heToggle"))
        self.assertEqual(self.errors, [])


if __name__ == "__main__":
    unittest.main()
