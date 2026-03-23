#!/usr/bin/env python3
"""
generate_dzi.py — Convert fluorescence microscopy images to DZI pyramids
for hyperOSD visualization.

Two input modes:
  --channels : Single-channel grayscale TIFFs (preferred). Packed into
               RGB or RGBA tile sources automatically.
  --rgb      : Pre-composed RGB/RGBA images. Each image = 1 tile source.

Two output formats (--pack):
  rgba (default) : 4 channels per tile source, PNG tiles.
                   Fewer GPU uploads, max 16 channels. Larger disk.
  rgb            : 3 channels per tile source, JPG tiles.
                   Smaller files, max 12 channels.

Usage:
    python generate_dzi.py --config channels.yaml
    python generate_dzi.py --channels ch1.tif ch2.tif ch3.tif --output ./out
    python generate_dzi.py --channels ch1.tif ch2.tif ch3.tif --pack rgb --output ./out
    python generate_dzi.py --rgb img1.tif img2.tif --output ./out

Requirements:
    pip install "pyvips[binary]" pyyaml
    # or:  conda install -c conda-forge pyvips pyyaml
"""

import argparse
import math
import os
import sys
import textwrap

try:
    import pyvips
except ImportError:
    print("Error: pyvips is not installed.", file=sys.stderr)
    print("Install it with:  pip install 'pyvips[binary]'", file=sys.stderr)
    print("            or:   conda install -c conda-forge pyvips", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants matching hyperOSD expectations
# ---------------------------------------------------------------------------
TILE_SIZE = 1024       # Larger tiles = fewer HTTP requests, better WLAN performance
OVERLAP = 1
LAYOUT = "dz"
MAX_TILE_SOURCES = 4   # hyperOSD supports up to 4 tile sources per z-level

# Pack mode settings
PACK_MODES = {
    "rgba": {"channels_per_tile": 4, "suffix": ".png", "max_channels": 16,
             "band_labels": ["R", "G", "B", "A"]},
    "rgb":  {"channels_per_tile": 3, "suffix": ".jpeg[Q=90]", "max_channels": 12,
             "band_labels": ["R", "G", "B"]},
}


def load_channel(path, force_8bit=True):
    """Load a single-channel image, converting to 8-bit if needed."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Channel image not found: {path}")

    image = pyvips.Image.new_from_file(path, access="sequential")

    # Extract first band if image has multiple bands
    if image.bands > 1:
        print(f"  Warning: {os.path.basename(path)} has {image.bands} bands, "
              f"using band 0 only")
        image = image[0]

    # Convert 16-bit to 8-bit
    if force_8bit and image.format == "ushort":
        print(f"  Converting 16-bit -> 8-bit: {os.path.basename(path)}")
        image = (image / 256).cast("uchar")
    elif force_8bit and image.format not in ("uchar",):
        print(f"  Converting {image.format} -> 8-bit: {os.path.basename(path)}")
        if image.format in ("float", "double"):
            image = (image * 255).cast("uchar")
        else:
            image = image.cast("uchar")

    return image


def pack_channels(channels, pack_mode):
    """Pack single-band images into an RGB or RGBA image.

    Missing channels are zero-filled. Result has exactly 3 or 4 bands
    depending on pack_mode.
    """
    mode = PACK_MODES[pack_mode]
    n_bands = mode["channels_per_tile"]

    if not channels:
        raise ValueError("At least one channel image is required")
    if len(channels) > n_bands:
        raise ValueError(f"Too many channels ({len(channels)}) for one tile "
                         f"source (max {n_bands} in {pack_mode} mode)")

    width = channels[0].width
    height = channels[0].height

    for i, ch in enumerate(channels[1:], start=1):
        if ch.width != width or ch.height != height:
            raise ValueError(
                f"Channel {i} dimensions ({ch.width}x{ch.height}) don't match "
                f"channel 0 ({width}x{height}). All channels must be the same size."
            )

    padded = list(channels)
    while len(padded) < n_bands:
        padded.append(pyvips.Image.black(width, height).cast("uchar"))

    return padded[0].bandjoin(padded[1:])


def load_rgb_tile_source(path, pack_mode, force_8bit=True):
    """Load a pre-composed RGB/RGBA image as a tile source.

    Adjusts band count to match pack_mode (3 for rgb, 4 for rgba).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")

    image = pyvips.Image.new_from_file(path, access="sequential")
    basename = os.path.basename(path)
    target_bands = PACK_MODES[pack_mode]["channels_per_tile"]

    # Convert 16-bit to 8-bit
    if force_8bit and image.format == "ushort":
        print(f"  Converting 16-bit -> 8-bit: {basename}")
        image = (image / 256).cast("uchar")
    elif force_8bit and image.format not in ("uchar",):
        print(f"  Converting {image.format} -> 8-bit: {basename}")
        if image.format in ("float", "double"):
            image = (image * 255).cast("uchar")
        else:
            image = image.cast("uchar")

    # Adjust bands to target
    if image.bands == 1:
        print(f"  {basename}: grayscale -> expanding to {pack_mode.upper()}")
        bands = [image, image, image]
        if target_bands == 4:
            bands.append(pyvips.Image.black(image.width, image.height)
                         .cast("uchar").invert())
        image = bands[0].bandjoin(bands[1:])
    elif image.bands == 3:
        if target_bands == 4:
            print(f"  {basename}: RGB -> adding alpha channel (255)")
            alpha = (pyvips.Image.black(image.width, image.height)
                     .cast("uchar").invert())
            image = image.bandjoin(alpha)
        else:
            print(f"  {basename}: RGB (3 bands, using as-is)")
    elif image.bands == 4:
        if target_bands == 3:
            print(f"  {basename}: RGBA -> dropping alpha for RGB mode")
            image = image[0].bandjoin([image[1], image[2]])
        else:
            print(f"  {basename}: RGBA (4 bands, using as-is)")
    else:
        print(f"  Warning: {basename} has {image.bands} bands, using first {target_bands}")
        bands = [image[i] for i in range(min(image.bands, target_bands))]
        while len(bands) < target_bands:
            bands.append(pyvips.Image.black(image.width, image.height).cast("uchar"))
        image = bands[0].bandjoin(bands[1:])

    return image


def generate_tile_source(image, output_dir, tile_index, suffix):
    """Generate a single DZI tile source in the hyperOSD layout."""
    tile_dir = os.path.join(output_dir, str(tile_index))
    os.makedirs(tile_dir, exist_ok=True)

    dzi_basename = os.path.join(tile_dir, "stitch")

    image.dzsave(
        dzi_basename,
        tile_size=TILE_SIZE,
        overlap=OVERLAP,
        suffix=suffix,
        layout=LAYOUT,
    )

    # Rename stitch.dzi -> stitch.xml (hyperOSD expects .xml)
    dzi_path = dzi_basename + ".dzi"
    xml_path = os.path.join(tile_dir, "stitch.xml")
    if os.path.exists(dzi_path):
        if os.path.exists(xml_path):
            os.remove(xml_path)
        os.rename(dzi_path, xml_path)
        print(f"    Renamed stitch.dzi -> stitch.xml")

    _fix_xml_tile_size(xml_path)


def _fix_xml_tile_size(xml_path):
    """Ensure TileSize matches TILE_SIZE in the DZI XML descriptor.

    pyvips subtracts 2 from the requested tile size (e.g. 1024 → 1022,
    256 → 254). Use a generic regex to catch any incorrect value.
    """
    import re
    with open(xml_path, "r") as f:
        content = f.read()
    fixed = re.sub(r'TileSize="\d+"', f'TileSize="{TILE_SIZE}"', content)
    if fixed != content:
        with open(xml_path, "w") as f:
            f.write(fixed)
        print(f"    Fixed TileSize to {TILE_SIZE} in XML")


def process_z_level(z_name, channel_paths, output_dir, pack_mode="rgba",
                    force_8bit=True):
    """Process single-channel images for one z-level.

    Packs channels into groups of 3 (RGB) or 4 (RGBA) and generates
    DZI pyramids for each group.
    """
    mode = PACK_MODES[pack_mode]
    cpt = mode["channels_per_tile"]
    n_channels = len(channel_paths)
    n_tile_sources = math.ceil(n_channels / cpt)

    if n_channels > mode["max_channels"]:
        raise ValueError(
            f"Too many channels ({n_channels}) for z-level '{z_name}'. "
            f"Maximum is {mode['max_channels']} in {pack_mode} mode."
        )

    z_dir = os.path.join(output_dir, z_name)
    os.makedirs(z_dir, exist_ok=True)

    print(f"\nZ-level: {z_name}  [pack: {pack_mode}]")
    print(f"  Channels: {n_channels} -> {n_tile_sources} "
          f"{pack_mode.upper()} tile source(s)")

    for tile_idx in range(n_tile_sources):
        start = tile_idx * cpt
        end = min(start + cpt, n_channels)
        group_paths = channel_paths[start:end]

        print(f"\n  Tile source {tile_idx + 1}:")
        channels = []
        for i, path in enumerate(group_paths):
            ch_num = start + i
            print(f"    Ch{ch_num} ({mode['band_labels'][i]}) <- {path}")
            channels.append(load_channel(path, force_8bit=force_8bit))

        packed = pack_channels(channels, pack_mode)
        print(f"    {pack_mode.upper()} image: "
              f"{packed.width}x{packed.height}, {packed.bands} bands")

        generate_tile_source(packed, z_dir, tile_idx + 1, mode["suffix"])
        print(f"    Done: {z_dir}/{tile_idx + 1}/stitch.xml")


def process_z_level_rgb(z_name, image_paths, output_dir, pack_mode="rgba",
                        force_8bit=True):
    """Process pre-composed RGB/RGBA images for one z-level."""
    if len(image_paths) > MAX_TILE_SOURCES:
        raise ValueError(
            f"Too many images ({len(image_paths)}) for z-level '{z_name}'. "
            f"Max {MAX_TILE_SOURCES} images per z-level."
        )

    mode = PACK_MODES[pack_mode]
    z_dir = os.path.join(output_dir, z_name)
    os.makedirs(z_dir, exist_ok=True)

    print(f"\nZ-level: {z_name}  [RGB input, pack: {pack_mode}]")
    print(f"  Images: {len(image_paths)} -> {len(image_paths)} tile source(s)")

    for tile_idx, path in enumerate(image_paths):
        print(f"\n  Tile source {tile_idx + 1}: <- {path}")
        img = load_rgb_tile_source(path, pack_mode, force_8bit=force_8bit)
        print(f"    {pack_mode.upper()} image: "
              f"{img.width}x{img.height}, {img.bands} bands")

        generate_tile_source(img, z_dir, tile_idx + 1, mode["suffix"])
        print(f"    Done: {z_dir}/{tile_idx + 1}/stitch.xml")


def load_yaml_config(config_path):
    """Load and validate a YAML configuration file."""
    try:
        import yaml
    except ImportError:
        print("Error: PyYAML is required for config file support.", file=sys.stderr)
        print("Install it with:  pip install pyyaml", file=sys.stderr)
        sys.exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if not config:
        raise ValueError("Config file is empty")

    output_dir = config.get("output_dir")
    if not output_dir:
        raise ValueError("Config file must specify 'output_dir'")

    z_levels = config.get("z_levels", [])
    if not z_levels:
        raise ValueError("Config file must specify at least one z-level")

    config_dir = os.path.dirname(os.path.abspath(config_path))

    for z in z_levels:
        if "name" not in z:
            raise ValueError("Each z-level must have a 'name' field")
        if "channels" not in z or not z["channels"]:
            raise ValueError(f"Z-level '{z['name']}' must have at least one channel")
        for ch in z["channels"]:
            if "path" not in ch:
                raise ValueError(f"Each channel in z-level '{z['name']}' must have "
                                 f"a 'path' field")
            if not os.path.isabs(ch["path"]):
                ch["path"] = os.path.join(config_dir, ch["path"])

    if not os.path.isabs(output_dir):
        output_dir = os.path.join(config_dir, output_dir)

    # Top-level pack mode (can be overridden per z-level)
    global_pack = config.get("pack", "rgba")

    return output_dir, z_levels, global_pack


def main():
    parser = argparse.ArgumentParser(
        description="Convert fluorescence microscopy images to DZI pyramids "
                    "for hyperOSD visualization.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:

              # Config file with single-channel TIFFs (recommended):
              python generate_dzi.py --config channels.yaml

              # Direct CLI, RGBA output (default — 4ch/tile, PNG):
              python generate_dzi.py --channels ch1.tif ch2.tif ch3.tif ch4.tif \\
                                     --z-level 0.0um --output ./output

              # Direct CLI, RGB output (3ch/tile, JPG, smaller files):
              python generate_dzi.py --channels ch1.tif ch2.tif ch3.tif \\
                                     --pack rgb --z-level 0.0um --output ./output

              # Pre-composed RGB images:
              python generate_dzi.py --rgb img1.tif img2.tif --output ./output

            Channel packing:
              RGBA mode: Tile1=[Ch0,Ch1,Ch2,Ch3] Tile2=[Ch4,Ch5,Ch6,Ch7] ... (max 16ch)
              RGB  mode: Tile1=[Ch0,Ch1,Ch2]     Tile2=[Ch3,Ch4,Ch5]     ... (max 12ch)
        """),
    )

    parser.add_argument(
        "--config", "-c",
        help="Path to YAML config file (recommended). "
             "See example_config.yaml.",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        help="Single-channel grayscale images in order (Ch0, Ch1, ...).",
    )
    parser.add_argument(
        "--rgb",
        nargs="+",
        metavar="IMAGE",
        help="Pre-composed RGB/RGBA images (up to 4). "
             "Each image = 1 tile source.",
    )
    parser.add_argument(
        "--pack", "-p",
        choices=["rgba", "rgb"],
        default="rgba",
        help="Output packing: rgba (4ch/tile, PNG, default) or "
             "rgb (3ch/tile, JPG, smaller files).",
    )
    parser.add_argument(
        "--z-level", "-z",
        default="0.0um",
        help="Z-level name (default: '0.0um').",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output directory.",
    )
    parser.add_argument(
        "--no-convert",
        action="store_true",
        help="Skip auto 16-bit to 8-bit conversion.",
    )

    args = parser.parse_args()

    # Validate argument combinations
    mode_count = sum(1 for x in [args.config, args.channels, args.rgb] if x)
    if mode_count > 1:
        parser.error("Use only one of --config, --channels, or --rgb.")
    if mode_count == 0:
        parser.error("Provide one of --config, --channels, or --rgb.")
    if (args.channels or args.rgb) and not args.output:
        parser.error("--output is required when using --channels or --rgb.")

    force_8bit = not args.no_convert
    pack_mode = args.pack
    mode = PACK_MODES[pack_mode]
    tile_fmt = "PNG" if "png" in mode["suffix"] else "JPG"

    print("=" * 60)
    print("hyperOSD DZI Generator")
    print("=" * 60)
    print(f"  pyvips {pyvips.__version__}, "
          f"libvips {pyvips.version(0)}.{pyvips.version(1)}.{pyvips.version(2)}")
    print(f"  Tile size: {TILE_SIZE}, Overlap: {OVERLAP}")
    print(f"  Pack: {pack_mode.upper()} ({mode['channels_per_tile']}ch/tile, "
          f"{tile_fmt} tiles)")

    if args.config:
        output_dir, z_levels, global_pack = load_yaml_config(args.config)
        # CLI --pack overrides config only if explicitly set
        if "--pack" not in sys.argv and "-p" not in sys.argv:
            pack_mode = global_pack
        print(f"  Config: {args.config}")
        print(f"  Output: {output_dir}")

        for z in z_levels:
            paths = [ch["path"] for ch in z["channels"]]
            z_pack = z.get("pack", pack_mode)
            z_mode = z.get("mode", "channels")
            if z_mode == "rgb":
                process_z_level_rgb(z["name"], paths, output_dir,
                                    pack_mode=z_pack, force_8bit=force_8bit)
            else:
                process_z_level(z["name"], paths, output_dir,
                                pack_mode=z_pack, force_8bit=force_8bit)
    elif args.rgb:
        output_dir = args.output
        print(f"  Output: {output_dir}")
        process_z_level_rgb(args.z_level, args.rgb, output_dir,
                            pack_mode=pack_mode, force_8bit=force_8bit)
    else:
        output_dir = args.output
        print(f"  Output: {output_dir}")
        process_z_level(args.z_level, args.channels, output_dir,
                        pack_mode=pack_mode, force_8bit=force_8bit)

    print("\n" + "=" * 60)
    print("Done! DZI pyramids generated successfully.")
    print(f"Output directory: {os.path.abspath(output_dir)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
