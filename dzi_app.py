#!/usr/bin/env python3
"""
dzi_app.py — Flask web app for DZI tile generation.

Provides a browser UI to scan image folders, configure z-levels,
and generate DZI tile pyramids for the hyperOSD viewer.

Usage:
    pip install flask pyvips
    python dzi_app.py
"""

import json
import math
import os
import re
import sys
import threading
import time
import webbrowser

from flask import Flask, Response, jsonify, render_template, request, send_from_directory

# Ensure generate_dzi can be imported from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_dzi

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Progress tracking (single-user local app)
# ---------------------------------------------------------------------------
_progress_lock = threading.Lock()
_progress = {
    "step": 0,
    "total": 0,
    "message": "",
    "done": False,
    "error": None,
    "viewer_path": None,
}

IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
EXCLUDE_FOLDERS = {"openseadragon", "raw_data", ".git", "__pycache__"}

# Path to the hyperOSD project's openseadragon/ folder (for copy-to-dataset)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_OSD_DIR = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "openseadragon"))

REQUIRED_OSD_FILES = [
    "openseadragon-bin-6.0.2/openseadragon.js",
    "hyperblend-webgl.js",
    "openseadragon-filtering-nistfuncs-12ch.js",
    "openseadragon-scalebar.js",
]

# Track last dataset root and resolved OSD path for serving viewer files
_state_lock = threading.Lock()
_dataset_root = None
_osd_serve_dir = None  # resolved openseadragon/ path (dataset root or app fallback)
_scan_channel_count = None  # total channels derived from last /scan (for matrix validation)


def _check_osd_dir(osd_dir):
    """Check if osd_dir exists and contains all required JS files.
    Returns (ok, missing_files) tuple."""
    if not os.path.isdir(osd_dir):
        return False, REQUIRED_OSD_FILES[:]
    missing = [f for f in REQUIRED_OSD_FILES
               if not os.path.isfile(os.path.join(osd_dir, f))]
    return len(missing) == 0, missing


def _reset_progress():
    with _progress_lock:
        _progress.update(
            step=0, total=0, message="", done=False, error=None, viewer_path=None
        )


def _set_progress(step=None, total=None, message=None, done=None, error=None,
                  viewer_path=None):
    with _progress_lock:
        if step is not None:
            _progress["step"] = step
        if total is not None:
            _progress["total"] = total
        if message is not None:
            _progress["message"] = message
        if done is not None:
            _progress["done"] = done
        if error is not None:
            _progress["error"] = error
        if viewer_path is not None:
            _progress["viewer_path"] = viewer_path


def _get_progress():
    with _progress_lock:
        return dict(_progress)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/load", methods=["POST"])
def load():
    """Scan a folder for existing viewer HTML files."""
    data = request.get_json(force=True)
    folder = data.get("path", "").strip()

    if not folder or not os.path.isdir(folder):
        return jsonify({"error": f"Not a valid directory: {folder}"}), 400

    folder = os.path.abspath(folder)

    # Check openseadragon availability (same fallback logic as scan)
    dataset_osd = os.path.join(folder, "openseadragon")
    app_osd = os.path.join(_SCRIPT_DIR, "openseadragon")
    dataset_osd_ok, _ = _check_osd_dir(dataset_osd)
    app_osd_ok, _ = _check_osd_dir(app_osd)

    global _dataset_root, _osd_serve_dir, _scan_channel_count
    with _state_lock:
        _dataset_root = folder
        _scan_channel_count = None  # invalidate — new dataset needs fresh /scan
        if dataset_osd_ok:
            _osd_serve_dir = dataset_osd
        elif app_osd_ok:
            _osd_serve_dir = app_osd
        else:
            _osd_serve_dir = None

    # Find viewer HTML files (contain zstackHyper or DATASET CONFIGURATION)
    viewers = []
    for f in sorted(os.listdir(folder)):
        if not f.lower().endswith(".html"):
            continue
        fpath = os.path.join(folder, f)
        if not os.path.isfile(fpath):
            continue
        # Quick check: read first 10KB for viewer signature
        with open(fpath, "r", errors="ignore") as fh:
            head = fh.read(10240)
        if "DATASET CONFIGURATION" in head or "hyperblend-webgl" in head:
            viewers.append(f)

    # Check for DZI tiles
    has_tiles = any(
        d.startswith("deepzoom_") and os.path.isdir(os.path.join(folder, d))
        for d in os.listdir(folder)
    )

    return jsonify({
        "root": folder,
        "viewers": viewers,
        "has_tiles": has_tiles,
    })


@app.route("/scan", methods=["POST"])
def scan():
    """Scan a folder for image subfolders (z-levels)."""
    import pyvips

    data = request.get_json(force=True)
    scan_path = data.get("path", "").strip()

    if not scan_path or not os.path.isdir(scan_path):
        return jsonify({"error": f"Not a valid directory: {scan_path}"}), 400

    scan_path = os.path.abspath(scan_path)
    dataset_name = os.path.basename(scan_path)
    warnings = []

    # Check for openseadragon/ — dataset root first, then fallback to app dir
    dataset_osd = os.path.join(scan_path, "openseadragon")
    app_osd = os.path.join(_SCRIPT_DIR, "openseadragon")

    dataset_osd_ok, dataset_osd_missing = _check_osd_dir(dataset_osd)
    app_osd_ok, _ = _check_osd_dir(app_osd)

    if dataset_osd_ok:
        osd_status = "dataset"
    elif app_osd_ok:
        osd_status = "app_fallback"
    else:
        osd_status = "missing"

    global _dataset_root, _osd_serve_dir
    with _state_lock:
        _dataset_root = scan_path
        if dataset_osd_ok:
            # Best case: openseadragon/ in dataset root with all files
            _osd_serve_dir = dataset_osd
        elif app_osd_ok:
            # Fallback: openseadragon/ next to dzi_app.py
            _osd_serve_dir = app_osd
        else:
            _osd_serve_dir = None

    # Determine image root: use raw_data/ if it exists, else scan_path itself
    raw_data_dir = os.path.join(scan_path, "raw_data")
    if os.path.isdir(raw_data_dir):
        image_root = raw_data_dir
        using_raw_data = True
    else:
        image_root = scan_path
        using_raw_data = False

    # Find immediate subfolders (excluding known non-data folders and deepzoom_*)
    subfolders = []
    for entry in sorted(os.listdir(image_root)):
        full = os.path.join(image_root, entry)
        if not os.path.isdir(full):
            continue
        low = entry.lower()
        if low in EXCLUDE_FOLDERS or low.startswith("deepzoom_"):
            continue
        subfolders.append(entry)

    if not subfolders:
        # No subfolders — check if images are directly in image_root (single z-level)
        root_images = _find_images(image_root)
        if root_images:
            subfolders = ["."]  # treat image_root as a single z-level
        else:
            return jsonify({"error": "No image subfolders found. "
                            "Put images in raw_data/{z-level}/ or directly in z-level subfolders."}), 400

    # Build z-level info
    z_levels = []
    band_mode = None

    for folder in subfolders:
        folder_path = os.path.join(image_root, folder) if folder != "." else image_root
        images = _find_images(folder_path)

        if not images:
            continue

        # Read first image metadata
        first_path = os.path.join(folder_path, images[0])
        try:
            img = pyvips.Image.new_from_file(first_path, access="sequential")
            first_bands = img.bands
            first_width = img.width
            first_height = img.height
            first_format = img.format
            # Check for multi-page TIFF (hyperspectral stack)
            n_pages = img.get("n-pages") if img.get_typeof("n-pages") else 1
        except Exception as e:
            warnings.append(f"Could not read {images[0]} in {folder}: {e}")
            continue

        # Detect TIFF stack: single file (or few files) with multiple pages
        is_tiff_stack = (len(images) == 1 and n_pages > 1)

        # Determine band mode from first z-level
        if band_mode is None:
            if is_tiff_stack:
                band_mode = "tiff_stack"
            elif first_bands == 1:
                band_mode = "grayscale"
            elif first_bands == 3:
                band_mode = "rgb"
            elif first_bands == 4:
                band_mode = "rgba"
            else:
                band_mode = f"{first_bands}-band"

        if is_tiff_stack:
            # Single TIFF stack: each page = one channel
            image_infos = [{
                "filename": images[0],
                "bands": first_bands,
                "width": first_width,
                "height": first_height,
                "format": first_format,
                "pages": n_pages,
            }]

            z_levels.append({
                "folder": folder,
                "name": folder,
                "images": image_infos,
                "tiff_stack": True,
                "n_channels": n_pages,
            })
        else:
            # Check sequential naming
            _check_sequential_naming(images, folder, warnings)

            image_infos = []
            for img_name in images:
                image_infos.append({
                    "filename": img_name,
                    "bands": first_bands,
                    "width": first_width,
                    "height": first_height,
                    "format": first_format,
                })

            z_levels.append({
                "folder": folder,
                "name": folder,
                "images": image_infos,
            })

    if not z_levels:
        return jsonify({"error": "No images found in any subfolder."}), 400

    # Cross-z-level validation: check for consistent band counts and image counts
    if len(z_levels) > 1:
        band_counts = [z["images"][0]["bands"] for z in z_levels if z["images"]]
        image_counts = [len(z["images"]) for z in z_levels]

        if len(set(band_counts)) > 1:
            band_details = ", ".join(
                f"{z['folder']}={bc}-band" for z, bc in zip(z_levels, band_counts)
            )
            warnings.append(
                f"Z-levels have different band counts ({band_details}). "
                "This may cause incorrect channel packing."
            )

        if len(set(image_counts)) > 1:
            count_details = ", ".join(
                f"{z['folder']}={ic} images" for z, ic in zip(z_levels, image_counts)
            )
            warnings.append(
                f"Z-levels have different image counts ({count_details}). "
                "Each z-level should have the same number of images."
            )

    # Cache channel count for matrix validation in /generate
    # grayscale/tiff_stack pack 4 channels per RGBA tile source, so round up
    global _scan_channel_count
    if band_mode == "tiff_stack":
        n = z_levels[0].get("n_channels", 16)
        _scan_channel_count = min(((n + 3) // 4) * 4, 16)
    elif band_mode == "grayscale":
        n = len(z_levels[0]["images"])
        _scan_channel_count = min(((n + 3) // 4) * 4, 16)
    elif band_mode == "rgb":
        _scan_channel_count = min(len(z_levels[0]["images"]) * 3, 12)
    else:  # rgba or N-band
        _scan_channel_count = min(len(z_levels[0]["images"]) * 4, 16)

    return jsonify({
        "root": scan_path,
        "image_root": image_root,
        "dataset_name": dataset_name,
        "z_levels": z_levels,
        "band_mode": band_mode,
        "osd_status": osd_status,  # "dataset", "app_fallback", or "missing"
        "osd_missing_files": dataset_osd_missing if not dataset_osd_ok else [],
        "using_raw_data": using_raw_data,
        "warnings": warnings,
    })


def _find_images(folder_path):
    """Return sorted list of image filenames in a folder."""
    images = []
    for f in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, f)):
            _, ext = os.path.splitext(f)
            if ext.lower() in IMAGE_EXTENSIONS:
                images.append(f)
    images.sort()
    return images


def _check_sequential_naming(images, folder, warnings):
    """Warn if image filenames suggest gaps in sequential numbering."""
    numbers = []
    for img in images:
        m = re.search(r"(\d+)", os.path.splitext(img)[0])
        if m:
            numbers.append(int(m.group(1)))
    if len(numbers) >= 2:
        numbers.sort()
        for i in range(1, len(numbers)):
            if numbers[i] - numbers[i - 1] > 1:
                warnings.append(
                    f"Possible gap in sequential numbering in '{folder}': "
                    f"jump from {numbers[i-1]} to {numbers[i]}."
                )
                break


@app.route("/generate", methods=["POST"])
def generate():
    """Start DZI generation in a background thread."""
    root_path = request.form.get("path", "").strip()
    image_root = request.form.get("image_root", "").strip() or root_path
    um_per_pixel = request.form.get("um_per_pixel") or None

    z_levels_raw = request.form.get("z_levels", "[]")
    try:
        z_levels = json.loads(z_levels_raw)
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid z_levels format"}), 400

    if not root_path or not os.path.isdir(root_path):
        return jsonify({"error": "Invalid path"}), 400
    if not z_levels:
        return jsonify({"error": "No z-levels provided"}), 400

    # Handle optional matrix file upload
    matrix_content = None
    matrix_file = request.files.get("matrix_file")
    if matrix_file and matrix_file.filename:
        matrix_content = matrix_file.read().decode("utf-8", errors="replace")

    # Validate matrix file if provided
    if matrix_content is not None:
        expected_channels = _scan_channel_count
        if expected_channels is None:
            return jsonify({"error": "Cannot determine channel count — run Scan first"}), 400
        err = validate_unmix_matrix(matrix_content, expected_channels)
        if err:
            return jsonify({"error": err}), 400

    _reset_progress()

    thread = threading.Thread(
        target=_generate_worker,
        args=(root_path, image_root, z_levels, um_per_pixel, matrix_content),
        daemon=True,
    )
    thread.start()

    return jsonify({"status": "started"})


@app.route("/progress")
def progress():
    """SSE endpoint for generation progress."""
    def stream():
        while True:
            p = _get_progress()
            yield f"data: {json.dumps(p)}\n\n"
            if p["done"] or p["error"]:
                break
            time.sleep(0.5)
    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/view")
@app.route("/view/<filename>")
def view(filename=None):
    """Serve a viewer HTML through Flask with path rewriting."""
    with _state_lock:
        root = _dataset_root

    if not root or not os.path.isdir(root):
        return "No dataset loaded. Scan or load a folder first.", 404

    if filename:
        # Reader mode: specific file requested
        viewer_path = os.path.join(root, filename)
    else:
        # Generator mode: find the auto-generated viewer
        dataset_name = os.path.basename(root)
        safe_name = re.sub(r"[^\w\-.]", "_", dataset_name)
        viewer_filename = f"{safe_name}_zstackHyper.html"
        viewer_path = os.path.join(root, viewer_filename)

    # Path traversal protection: ensure resolved path is inside dataset root
    resolved = os.path.realpath(viewer_path)
    real_root = os.path.realpath(root)
    if not (resolved == real_root or resolved.startswith(real_root + os.sep)):
        return "Access denied", 403

    if not os.path.isfile(viewer_path):
        return f"Viewer not found: {os.path.basename(viewer_path)}", 404

    with open(viewer_path, "r") as f:
        html = f.read()

    # Rewrite all relative paths to go through /dataset/ route
    # Uses regex to catch deepzoom_RGBA, deepzoom_RGBA_16CH, deepzoom_RGB_12CH, etc.
    html = re.sub(r'(["\'])openseadragon/', r'\1/dataset/openseadragon/', html)
    html = re.sub(r'(["\'])deepzoom_', r'\1/dataset/deepzoom_', html)

    return html


@app.route("/dataset/<path:filepath>")
def serve_dataset(filepath):
    """Serve files from the dataset folder, with openseadragon/ fallback."""
    with _state_lock:
        root = _dataset_root
        osd_dir = _osd_serve_dir

    if not root or not os.path.isdir(root):
        return "No dataset loaded", 404

    # If requesting openseadragon/ files and they're not in dataset root,
    # serve from the fallback location (next to dzi_app.py)
    if filepath.startswith("openseadragon/") and osd_dir:
        osd_rel = filepath[len("openseadragon/"):]
        candidate = os.path.join(root, filepath)
        if not os.path.isfile(candidate):
            return send_from_directory(osd_dir, osd_rel)

    return send_from_directory(root, filepath)


# ---------------------------------------------------------------------------
# Unmixing matrix validation
# ---------------------------------------------------------------------------

def validate_unmix_matrix(file_content, expected_channels):
    """Validate an unmixing matrix file.

    Args:
        file_content: raw text content of the matrix file
        expected_channels: number of input channels the dataset has

    Returns:
        None if valid, or an error message string if invalid.
    """
    lines = [line.strip() for line in file_content.splitlines() if line.strip()]
    if not lines:
        return "Matrix file is empty"

    rows = []
    for i, line in enumerate(lines):
        values = re.split(r"[\t,\s]+", line)
        values = [v for v in values if v]  # drop empty strings from leading/trailing delimiters
        try:
            floats = [float(v) for v in values]
        except ValueError:
            return f"Matrix file contains non-numeric values on line {i + 1}"
        for f in floats:
            if math.isnan(f) or math.isinf(f):
                return "Matrix file contains NaN or infinite values"
        rows.append(floats)

    # Check row count (1–8 output components)
    if len(rows) < 1 or len(rows) > 8:
        return f"Matrix has {len(rows)} rows — must be 1 to 8 output components"

    # Check consistent column count
    col_count = len(rows[0])
    for i, row in enumerate(rows[1:], start=2):
        if len(row) != col_count:
            return (f"Matrix file has inconsistent column counts "
                    f"(line {i} has {len(row)}, expected {col_count})")

    # Check column count matches dataset channels
    if col_count != expected_channels:
        return (f"Matrix has {col_count} columns but dataset has "
                f"{expected_channels} channels")

    return None


# ---------------------------------------------------------------------------
# Background generation worker
# ---------------------------------------------------------------------------

def _generate_worker(root_path, image_root, z_levels, um_per_pixel=None, matrix_content=None):
    """Run DZI generation in background thread.

    root_path: dataset root (output goes here)
    image_root: where z-level subfolders live (root_path or root_path/raw_data)
    um_per_pixel: optional pixel size in micrometers (controls scale bar)
    """
    import pyvips

    try:
        dataset_name = os.path.basename(root_path)
        output_dir = os.path.join(root_path, "deepzoom_RGBA")
        os.makedirs(output_dir, exist_ok=True)

        # Write matrix file if provided
        if matrix_content is not None:
            matrix_path = os.path.join(root_path, "unmix_matrix.txt")
            with open(matrix_path, "w", encoding="utf-8") as f:
                f.write(matrix_content)
        pack_mode = "rgba"

        # Determine band mode from first image in first z-level
        first_z = z_levels[0]
        first_folder = first_z["folder"]
        first_folder_path = (
            image_root if first_folder == "." else
            os.path.join(image_root, first_folder)
        )
        first_images = _find_images(first_folder_path)
        if not first_images:
            _set_progress(error="No images found in first z-level folder.")
            return

        first_img_path = os.path.join(first_folder_path, first_images[0])
        first_img = pyvips.Image.new_from_file(first_img_path, access="sequential")
        n_pages = first_img.get("n-pages") if first_img.get_typeof("n-pages") else 1
        is_tiff_stack = (len(first_images) == 1 and n_pages > 1)
        is_grayscale = (not is_tiff_stack) and first_img.bands == 1
        is_rgb = (not is_tiff_stack) and first_img.bands == 3

        # Validate all z-levels have consistent image mode and band count
        first_bands = first_img.bands
        for z in z_levels[1:]:
            v_folder = z["folder"]
            v_path = (
                image_root if v_folder == "." else os.path.join(image_root, v_folder)
            )
            v_images = _find_images(v_path)
            if not v_images:
                _set_progress(error=f"No images found in z-level '{v_folder}'.")
                return

            v_img_path = os.path.join(v_path, v_images[0])
            v_img = pyvips.Image.new_from_file(v_img_path, access="sequential")
            v_pages = v_img.get("n-pages") if v_img.get_typeof("n-pages") else 1
            v_is_stack = (len(v_images) == 1 and v_pages > 1)

            if v_is_stack != is_tiff_stack or v_img.bands != first_bands:
                first_mode = "tiff_stack" if is_tiff_stack else f"{first_bands}-band"
                this_mode = "tiff_stack" if v_is_stack else f"{v_img.bands}-band"
                _set_progress(
                    error=f"Mode mismatch: z-level '{first_z['folder']}' is {first_mode} "
                          f"but z-level '{v_folder}' is {this_mode}. "
                          f"All z-levels must use the same image format."
                )
                return

        # Cap total channel count at 16 (viewer shader limit: 4 layers × 4 channels)
        max_channels = 16
        max_tile_sources = 4
        if is_tiff_stack:
            total_channels = n_pages
            if total_channels > max_channels:
                _set_progress(
                    message=f"Warning: {total_channels} channels detected, viewer supports max "
                            f"{max_channels}. Only the first {max_channels} will be used."
                )
                n_pages = max_channels
        elif is_grayscale:
            total_channels = len(first_images)
            if total_channels > max_channels:
                _set_progress(
                    message=f"Warning: {total_channels} channels detected, viewer supports max "
                            f"{max_channels}. Only the first {max_channels} will be used."
                )
                first_images = first_images[:max_channels]
        else:
            # Pre-composed RGB/RGBA: 1 image = 1 tile source, max 4
            if len(first_images) > max_tile_sources:
                _set_progress(
                    message=f"Warning: {len(first_images)} pre-composed images detected, "
                            f"viewer supports max {max_tile_sources} tile sources. "
                            f"Only the first {max_tile_sources} will be used."
                )
                first_images = first_images[:max_tile_sources]

        # Count total steps (one per tile source per z-level)
        total_steps = 0
        for z in z_levels:
            folder = z["folder"]
            folder_path = (
                image_root if folder == "." else os.path.join(image_root, folder)
            )
            images = _find_images(folder_path)
            if is_tiff_stack:
                total_steps += math.ceil(n_pages / 4)
            elif is_grayscale:
                capped = min(len(images), max_channels)
                total_steps += math.ceil(capped / 4)
            else:
                capped = min(len(images), max_tile_sources)
                total_steps += capped
        total_steps += 1  # viewer HTML generation step

        _set_progress(step=0, total=total_steps, message="Starting generation...")

        step = 0
        all_image_sources = []  # for viewer HTML
        images_per_z = None

        for z in z_levels:
            z_name = z.get("name", z["folder"])
            folder = z["folder"]
            folder_path = (
                image_root if folder == "." else os.path.join(image_root, folder)
            )
            images = sorted(_find_images(folder_path))
            image_paths = [os.path.join(folder_path, img) for img in images]

            # Cap per-z-level image list to respect global channel limit
            if is_grayscale and len(image_paths) > max_channels:
                image_paths = image_paths[:max_channels]
            elif not is_tiff_stack and not is_grayscale and len(image_paths) > max_tile_sources:
                image_paths = image_paths[:max_tile_sources]

            z_output_dir = os.path.join(output_dir, z_name)
            os.makedirs(z_output_dir, exist_ok=True)

            if is_tiff_stack:
                # TIFF stack: single file, each page = one grayscale channel
                stack_path = image_paths[0]
                stack_img = pyvips.Image.new_from_file(stack_path, access="sequential")
                stack_pages = stack_img.get("n-pages") if stack_img.get_typeof("n-pages") else 1
                # Apply the global cap (n_pages was already capped above)
                stack_pages = min(stack_pages, n_pages)
                cpt = 4  # pack 4 pages per RGBA tile source
                n_tile_sources = math.ceil(stack_pages / cpt)
                if images_per_z is None:
                    images_per_z = n_tile_sources

                for tile_idx in range(n_tile_sources):
                    start = tile_idx * cpt
                    end = min(start + cpt, stack_pages)

                    step += 1
                    _set_progress(
                        step=step,
                        message=f"Z={z_name}: extracting pages {start+1}-{end} "
                                f"into tile source {tile_idx + 1}..."
                    )

                    channels = []
                    for page_num in range(start, end):
                        page_img = pyvips.Image.new_from_file(
                            stack_path, page=page_num, access="sequential"
                        )
                        # Extract first band if page is multi-band
                        if page_img.bands > 1:
                            page_img = page_img[0]
                        # Convert to 8-bit if needed
                        if page_img.format == "ushort":
                            page_img = (page_img / 256).cast("uchar")
                        elif page_img.format not in ("uchar",):
                            if page_img.format in ("float", "double"):
                                page_img = (page_img * 255).cast("uchar")
                            else:
                                page_img = page_img.cast("uchar")
                        channels.append(page_img)

                    packed = generate_dzi.pack_channels(channels, pack_mode)
                    mode = generate_dzi.PACK_MODES[pack_mode]
                    generate_dzi.generate_tile_source(
                        packed, z_output_dir, tile_idx + 1, mode["suffix"]
                    )
                    all_image_sources.append(
                        f"deepzoom_RGBA/{z_name}/{tile_idx + 1}/stitch.xml"
                    )

            elif is_grayscale:
                # Pack 4 channels per tile source
                cpt = 4
                n_tile_sources = math.ceil(len(image_paths) / cpt)
                if images_per_z is None:
                    images_per_z = n_tile_sources

                for tile_idx in range(n_tile_sources):
                    start = tile_idx * cpt
                    end = min(start + cpt, len(image_paths))
                    group = image_paths[start:end]

                    step += 1
                    _set_progress(
                        step=step,
                        message=f"Z={z_name}: packing channels {start}-{end-1} "
                                f"into tile source {tile_idx + 1}..."
                    )

                    channels = [
                        generate_dzi.load_channel(p, force_8bit=True) for p in group
                    ]
                    packed = generate_dzi.pack_channels(channels, pack_mode)
                    mode = generate_dzi.PACK_MODES[pack_mode]
                    generate_dzi.generate_tile_source(
                        packed, z_output_dir, tile_idx + 1, mode["suffix"]
                    )
                    all_image_sources.append(
                        f"deepzoom_RGBA/{z_name}/{tile_idx + 1}/stitch.xml"
                    )
            else:
                # RGB/RGBA pass-through
                if images_per_z is None:
                    images_per_z = len(image_paths)

                for tile_idx, path in enumerate(image_paths):
                    step += 1
                    _set_progress(
                        step=step,
                        message=f"Z={z_name}: processing tile source {tile_idx + 1} "
                                f"({os.path.basename(path)})..."
                    )

                    img = generate_dzi.load_rgb_tile_source(
                        path, pack_mode, force_8bit=True
                    )
                    mode = generate_dzi.PACK_MODES[pack_mode]
                    generate_dzi.generate_tile_source(
                        img, z_output_dir, tile_idx + 1, mode["suffix"]
                    )
                    all_image_sources.append(
                        f"deepzoom_RGBA/{z_name}/{tile_idx + 1}/stitch.xml"
                    )

        # Generate viewer HTML
        step += 1
        _set_progress(step=step, message="Generating viewer HTML...")

        # For TIFF stacks, pass actual page count so defaultChannels is correct
        actual_channels = n_pages if is_tiff_stack else None
        viewer_path = _generate_viewer_html(
            root_path, dataset_name, z_levels, all_image_sources,
            images_per_z or 1, is_grayscale, is_rgb,
            actual_channel_count=actual_channels,
            um_per_pixel=um_per_pixel,
        )

        _set_progress(
            step=step, done=True,
            message="Generation complete!",
            viewer_path=viewer_path,
        )

    except Exception as e:
        _set_progress(error=str(e))


def _parse_z_value(name):
    """Extract numeric z-value from a folder name like '4.0um' -> 4.0."""
    m = re.search(r"([\d.]+)", name)
    return float(m.group(1)) if m else 0.0


def _generate_viewer_html(root_path, dataset_name, z_levels, image_sources,
                          images_per_z, is_grayscale, is_rgb,
                          actual_channel_count=None, um_per_pixel=None):
    """Generate a customized viewer HTML file from the zstackHyper.html template."""
    # Find template in Flask templates folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(script_dir, "templates", "zstackHyper.html")

    if not os.path.isfile(template_path):
        raise FileNotFoundError(
            f"Viewer template not found: {template_path}"
        )

    with open(template_path, "r") as f:
        html = f.read()

    # Compute configuration values
    z_level_count = len(z_levels)
    z_level_values = [_parse_z_value(z.get("name", z["folder"])) for z in z_levels]

    if is_grayscale:
        channels_per_image = 4  # RGBA packing
    elif is_rgb:
        channels_per_image = 3
    else:
        channels_per_image = 4  # RGBA

    total_channels = actual_channel_count or (images_per_z * channels_per_image)
    default_channels = list(range(1, total_channels + 1))

    # Format image sources as JS array entries
    sources_js = ",\n\t\t\t".join(f'"{s}"' for s in image_sources)

    # Build um_per_pixel line for config block
    if um_per_pixel:
        try:
            um_val = max(0.0, float(um_per_pixel))
        except (ValueError, TypeError):
            um_val = 0.0
        um_line = f"\t\tvar umPerPixel = {um_val};          // micrometers per pixel at full resolution\n"
    else:
        um_line = f"\t\tvar umPerPixel = 0;              // no pixel size provided — scale bar disabled\n"

    # Build replacement config block
    config_block = (
        f"// ===== DATASET CONFIGURATION =====\n"
        f"{um_line}"
        f"\t\tvar zLevelCount = {z_level_count};             // number of z-planes\n"
        f"\t\tvar zLevelValues = {json.dumps(z_level_values)};       // z-level labels in micrometers\n"
        f"\t\tvar channelsPerImage = {channels_per_image};        // 3 = RGB (alpha slots locked), 4 = RGBA (all channels active)\n"
        f"\t\tvar defaultChannels = {json.dumps(default_channels)}; // 1-based visible channel numbers ([] for none)\n"
        f"\t\tvar imageSources = [             // DZI tile source paths (grouped by z-level)\n"
        f"\t\t\t{sources_js},\n"
        f"\t\t];\n"
        f"\t\tvar matrixFileName = 'unmix_matrix.txt';  // relative to HTML file, same format as imageSources\n"
        f"\t\t// ===== END DATASET CONFIGURATION ====="
    )

    # Replace configuration block
    pattern = r"// ===== DATASET CONFIGURATION =====.*?// ===== END DATASET CONFIGURATION ====="
    html = re.sub(pattern, config_block, html, flags=re.DOTALL)

    # Set browser tab title to dataset name
    html = html.replace(
        "<title>hyperOSD Multi-Channel Blending</title>",
        f"<title>{dataset_name} — hyperOSD</title>"
    )

    # Set Reinhard toggle to checked (only if not already checked)
    if 'id="reinhardToggle" checked' not in html:
        html = html.replace(
            'id="reinhardToggle"',
            'id="reinhardToggle" checked'
        )

    # Write output — sanitize dataset name for safe filename
    safe_name = re.sub(r"[^\w\-.]", "_", dataset_name)
    viewer_filename = f"{safe_name}_zstackHyper.html"
    viewer_path = os.path.join(root_path, viewer_filename)
    with open(viewer_path, "w") as f:
        f.write(html)

    return viewer_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    auto_open_browser = os.environ.get("HYPEROSG_NO_BROWSER", "").lower() not in {
        "1", "true", "yes",
    }
    if auto_open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    app.run(host="127.0.0.1", port=port, debug=False)
