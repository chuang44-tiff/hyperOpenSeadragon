# Tests

A small suite that checks the parts of hyperOSD a reviewer can run without a GPU or a
private dataset: the viewer pages, the DZI tile pipeline, the demo dataset, and a headless
browser smoke test.

## Dependencies

- Python 3.11 with `pyvips` (and a working libvips), `numpy`, `flask`, `pyyaml`
- `playwright` plus its Chromium: `python -m playwright install chromium`

`environment.yaml` provides the first group; `pip install -r tests/requirements-test.txt`
adds `numpy` and `playwright`.

## Running

Run each module in its own process, from the repository root:

    python -m unittest tests.test_static_hosts
    python -m unittest tests.test_dzi_pipeline
    python -m unittest tests.test_demo_data
    python -m unittest tests.test_viewer_smoke

or all at once: `python -m unittest discover -s tests -t .`

| Module | Covers | Needs |
|---|---|---|
| `test_static_hosts` | both viewer HTML pages: every referenced file exists, script load order, annotation blocks identical, dataset configuration, no private paths in published files | stdlib |
| `test_dzi_pipeline` | 8-bit conversion, channel packing (4/5/6 channels), coarse pyramid levels keep sparse channels, descriptor layout, matrix validation, generated viewer configuration, path containment | pyvips, numpy, flask |
| `test_demo_data` | `make_demo_data.py` output and the committed `demo_data/` match a numpy render: the finest level pixel for pixel, a mid level against a 4x4 box mean, every tile present | pyvips, numpy |
| `test_viewer_smoke` | headless Chromium (software WebGL): the HyperBlend drawer starts, 16 channels listed, demo data renders, z-level switch, Linear Unmixing on the demo matrix | playwright |

## Missing dependencies

A module whose dependency is missing is skipped, with the reason printed. To make that a
failure instead (for CI, where a skip would hide a broken setup), set

    HYPEROSD_REQUIRE_ALL=1

The smoke test always renders with SwiftShader (software GL), so results do not depend on
the machine's GPU.
