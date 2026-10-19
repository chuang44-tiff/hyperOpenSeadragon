"""Static checks on the two viewer hosts (stdlib only, no browser).

The hosts are ``zstackHyper.html`` and the generated-viewer template. They must
reference only files that exist, load the same scripts in the same order (the
OSD 6.0.2 / NIST bundle coexistence shim is order-sensitive), keep their
annotation blocks byte-identical, and the checked-in configuration must point at
the committed demo dataset. Published files must not mention private paths.

Run: python -m unittest tests.test_static_hosts   (public layout)
     python -m unittest tests.public.test_static_hosts   (dev layout)
"""
import re
import unittest
from html.parser import HTMLParser

from . import _layout

HOST, TEMPLATE = _layout.HOSTS
ROOT = _layout.ROOT


def _read(path):
    # CR-normalised, so CRLF checkouts do not matter.
    return path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")


class _Collector(HTMLParser):
    """Collect external references and the ordered script events of a page."""

    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.refs = []      # (tag, attr, value)
        self.scripts = []   # ("src", path) | ("inline", text), in document order
        self._in_inline = False
        self._buf = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "script":
            if a.get("src"):
                self.refs.append(("script", "src", a["src"]))
                self.scripts.append(("src", a["src"]))
            else:
                self._in_inline = True
                self._buf = []
        elif tag == "link":
            rel = (a.get("rel") or "").lower().split()
            if a.get("href") and ("stylesheet" in rel or "preload" in rel):
                self.refs.append(("link", "href", a["href"]))

    def handle_data(self, data):
        if self._in_inline:
            self._buf.append(data)

    def handle_endtag(self, tag):
        if tag == "script" and self._in_inline:
            self._in_inline = False
            self.scripts.append(("inline", "".join(self._buf)))


def _parse(path):
    c = _Collector()
    c.feed(_read(path))
    return c


class TestReferencesExist(unittest.TestCase):
    def test_every_script_and_stylesheet_and_preload_exists(self):
        for host in _layout.HOSTS:
            c = _parse(host)
            self.assertGreater(len(c.refs), 10, host)
            for tag, attr, value in c.refs:
                if re.match(r"^(?:[a-z]+:|//|data:)", value):
                    continue  # external or data: URL, not a repo file
                target = ROOT / value.split("?")[0].split("#")[0]
                self.assertTrue(target.is_file(),
                                "%s: <%s %s=%r> -> missing %s" % (host.name, tag, attr, value, target))


class TestScriptPhases(unittest.TestCase):
    MIN_AND_FALLBACK = [
        ("openseadragon/openseadragon-bin-6.0.2/openseadragon.min.js",
         "openseadragon/openseadragon-bin-6.0.2/openseadragon.js"),
        ("openseadragon/openseadragon-filtering-nistfuncs-12ch.min.js",
         "openseadragon/openseadragon-filtering-nistfuncs-12ch.js"),
    ]

    def test_script_src_sequence_is_identical_across_hosts(self):
        a = [v for k, v in _parse(HOST).scripts if k == "src"]
        b = [v for k, v in _parse(TEMPLATE).scripts if k == "src"]
        self.assertEqual(a, b)

    def test_phase_order(self):
        srcs = [v for k, v in _parse(HOST).scripts if k == "src"]
        expected = [
            "openseadragon/openseadragon-bin-6.0.2/openseadragon.min.js",            # phase 1
            "openseadragon/openseadragon-filtering-nistfuncs-12ch.min.js",           # phase 3
            "openseadragon/hyperblend-webgl.js",                                     # phase 5
            "openseadragon/openseadragon-scalebar.js",                               # phase 6
        ]
        positions = [srcs.index(s) for s in expected]
        self.assertEqual(positions, sorted(positions), srcs)

    def test_min_builds_are_followed_by_document_write_fallbacks(self):
        for host in _layout.HOSTS:
            events = _parse(host).scripts
            for minified, fallback in self.MIN_AND_FALLBACK:
                i = events.index(("src", minified))
                kind, text = events[i + 1]
                self.assertEqual(kind, "inline", host.name)
                self.assertIn("document.write", text, host.name)
                self.assertIn(fallback, text, host.name)

    def test_phase_2_snapshot_sits_between_osd_and_nist(self):
        # The snapshot/restore shim lives in inline scripts between the OSD tag and
        # the NIST tag (snapshot) and after the NIST tag (restore). Reordering them
        # silently breaks the viewer, so pin the relative order.
        for host in _layout.HOSTS:
            events = _parse(host).scripts
            osd = events.index(("src", self.MIN_AND_FALLBACK[0][0]))
            nist = events.index(("src", self.MIN_AND_FALLBACK[1][0]))
            hyper = events.index(("src", "openseadragon/hyperblend-webgl.js"))
            self.assertLess(osd, nist, host.name)
            self.assertLess(nist, hyper, host.name)
            between = [t for k, t in events[osd + 2:nist] if k == "inline"]
            self.assertTrue(any("getOwnPropertyDescriptor" in t for t in between),
                            "%s: no snapshot script between OSD and NIST" % host.name)


class TestAnnotationBlocksIdentical(unittest.TestCase):
    MARKERS = {
        "SCRIPT": ("<!-- ANNOTATION-SCRIPT-START -->", "<!-- ANNOTATION-SCRIPT-END -->"),
        "MARKUP": ("<!-- ANNOTATION-MARKUP-START -->", "<!-- ANNOTATION-MARKUP-END -->"),
        "WIRING": ("// <!-- ANNOTATION-WIRING-START -->", "// <!-- ANNOTATION-WIRING-END -->"),
    }

    @staticmethod
    def _between(src, start, end):
        i = src.index(start) + len(start)
        return src[i:src.index(end, i)]

    def test_blocks_are_byte_identical_and_non_trivial(self):
        a, b = _read(HOST), _read(TEMPLATE)
        for name, (start, end) in self.MARKERS.items():
            for src in (a, b):
                self.assertEqual(src.count(start), 1, name)
                self.assertEqual(src.count(end), 1, name)
            ba, bb = self._between(a, start, end), self._between(b, start, end)
            self.assertGreater(len(ba.strip()), 20, name)
            self.assertEqual(ba, bb, "%s annotation block differs between the hosts" % name)


class TestDatasetConfiguration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        text = _read(HOST)
        m = re.search(r"// ===== DATASET CONFIGURATION =====(.*?)// ===== END DATASET CONFIGURATION =====",
                      text, re.S)
        assert m, "DATASET CONFIGURATION block not found"
        cls.block = m.group(1)

    def _scalar(self, name):
        m = re.search(r"var %s = ([^;]+);" % name, self.block)
        self.assertIsNotNone(m, name)
        return m.group(1).strip()

    def test_image_sources_exist_and_count_matches(self):
        m = re.search(r"var imageSources = \[(.*?)\];", self.block, re.S)
        self.assertIsNotNone(m)
        body = re.sub(r"//[^\n]*", "", m.group(1))  # drop comment lines
        sources = re.findall(r'"([^"]+)"', body)
        z_count = int(self._scalar("zLevelCount"))
        self.assertEqual(len(sources), z_count * 4)
        for s in sources:
            self.assertTrue((ROOT / s).is_file(), s)

    def test_channels_per_image_and_z_values(self):
        self.assertEqual(int(self._scalar("channelsPerImage")), 4)
        values = [float(v) for v in self._scalar("zLevelValues").strip("[]").split(",")]
        self.assertEqual(len(values), int(self._scalar("zLevelCount")))

    def test_default_channels_are_valid_one_based_indices(self):
        chans = [int(v) for v in self._scalar("defaultChannels").strip("[]").split(",") if v.strip()]
        self.assertTrue(chans)
        self.assertTrue(all(1 <= c <= 16 for c in chans))


class TestNoPrivateReferences(unittest.TestCase):
    # Split literals so this file does not itself match a plain-text scan for these strings.
    FORBIDDEN = ("hyperBlend_" "SampleData", "test_" "synthetic", "doc/" "ticket", "doc/" "specs")
    VENDORED = ("openseadragon-filtering-nistfuncs-12ch", "openseadragon-bin-", "jquery")

    def _published(self):
        files = list(_layout.HOSTS)
        files += [p for p in sorted((ROOT / "openseadragon").glob("*.js"))
                  if not any(v in p.name for v in self.VENDORED)]
        files += sorted(_layout.GEN_DIR.glob("*.py"))
        readme = _layout.DEMO_DATA / "README.md"
        assert readme.is_file(), "demo_data/README.md is missing"
        files.append(readme)
        return files

    def test_published_files_do_not_mention_private_paths(self):
        files = self._published()
        self.assertGreater(len(files), 8)
        for path in files:
            text = path.read_text(encoding="utf-8", errors="replace")
            for needle in self.FORBIDDEN:
                self.assertNotIn(needle, text, "%s mentions %r" % (path.name, needle))


if __name__ == "__main__":
    unittest.main()
