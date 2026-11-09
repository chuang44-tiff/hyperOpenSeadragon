---
title: 'hyperOpenSeadragon: a browser-based viewer for multi-channel fluorescence and hyperspectral microscopy'
tags:
  - JavaScript
  - Python
  - WebGL
  - microscopy
  - fluorescence
  - hyperspectral imaging
  - spectral unmixing
  - OpenSeadragon
authors:
  - name: Chi Z. Huang
    orcid: 0009-0002-5555-1232
    affiliation: 1
  - name: Michael G. Giacomelli
    orcid: 0000-0002-2570-0770
    affiliation: 1
affiliations:
  - name: Department of Biomedical Engineering, University of Rochester, Rochester, NY, United States
    index: 1
    ror: 022kthw22
date: 2 November 2026  # TODO: set on submission
bibliography: paper.bib
---

# Summary

Fluorescence and hyperspectral microscopes can record a dozen or more spectral channels per focal plane, in gigapixel-scale images. hyperOpenSeadragon (hyperOSD) is a viewer for such images that runs in a web browser. It builds on OpenSeadragon [@gilman2026openseadragon] and adds a WebGL [@khronos2011webgl1] renderer that blends up to 16 channels per z-plane, each with its own hue and gain, and tone-maps the result.

The renderer can also unmix the channels, either linearly, with a matrix editor and a spectrum inspector, or with PICASSO, whose matrix comes from an optimiser running in a browser Web Worker. Optional passes add a virtual H&E stain based on the Beer–Lambert model of @giacomelli2016virtual, a Masson trichrome stain that extends that model, and a guided-filter denoise [@he2013guided] that changes only the display. An annotation tool exports per-class label masks.

Two companion Python tools, a command-line program and a small Flask [@flask] web app, both built on libvips/pyvips [@cupitt2025libvips; @pyvips], convert microscope TIFFs into the Deep Zoom tile pyramids the viewer reads. The web app also writes a ready-to-open viewer page. The resulting folder can be served by any ordinary web server.

# Statement of need

Multi-channel fluorescence datasets are hard to share and hard to inspect. Desktop tools such as Fiji [@schindelin2012fiji], napari [@napari2026] and QuPath [@bankhead2017qupath] are powerful, but they must be installed and are usually run on a local copy of the data. Sharing a dataset therefore means copying the full-resolution files, which often run to hundreds of gigabytes, again and again: one copy for each collaborator, and a further copy for every workstation, laptop or other device on which the images are viewed. Hyperspectral data adds a second problem. The raw channels are spectrally mixed, and the structures of interest separate only after unmixing [@dickinson2001multispectral; @zimmermann2003spectral; @keshava2002spectral]. In most workflows unmixing is an offline step that writes a new set of files, so the unmixing matrix cannot be adjusted while the user is looking at the tissue.

hyperOSD addresses both problems. A dataset becomes a folder of tiles and one HTML page, hosted once on a web server. Sharing it then means sending a link rather than copying files. Collaborators open the link in any WebGL-capable browser, and the viewer downloads tiles as they are needed rather than the whole dataset. Blending, unmixing, virtual staining and denoising all run as GPU passes over those tiles, so a user can:

- change a channel's colour or gain;
- edit a cell of the unmixing matrix, or add a spectrum picked from a pixel;
- generate a PICASSO matrix from the current field of view;
- switch between fluorescence and virtual-histology rendering.

Every such change is applied at display time; the source tiles are never modified.

The intended users are microscopists and their collaborators: biologists, pathologists and clinicians who need to examine multiplexed or hyperspectral images without setting up software on their own machines.

# State of the field

Several open-source tools already display multiplexed images on the web. Minerva [@hoffer2020minerva; @rashid2022narrative] is a narrative image browser for multiplexed tissue images, such as those from the MCMICRO pipeline [@schapiro2022mcmicro]. It builds on OpenSeadragon and centres on authored stories over channel groups; those groups are pre-rendered when a story is published. Viv [@manz2022viv] renders multiscale OME-TIFF and OME-Zarr images in the browser with WebGL, and it provides the image-rendering layer of the Vitessce framework for single-cell and spatial data [@keller2025vitessce]. Two further tools display images through OpenSeadragon and rely on a server back end: HistoJS [@masoud2024histojs], an analysis tool for multiplexed immunofluorescence images, and Scope2Screen [@jessup2022scope2screen], which adds focus+context lenses for multivariate tissue images. OMERO [@allan2012omero] provides server-backed data management with a web viewer.

On the desktop, napari, QuPath and Fiji offer extensive analysis functionality and plugin ecosystems. They run as local installations but can also reach remote data through connectors such as the OMERO plugins for Fiji and QuPath, or napari-ome-zarr for napari. Unmixing is also available inside desktop viewers. It is provided by Fiji plugins such as LUMoS [@mcrae2019robust], by the acquisition software of some commercial spectral microscopes, and by napari-PICASSO, a third-party plugin that brings PICASSO [@seo2022picasso] into napari. Outside napari, PICASSO, originally implemented in MATLAB, and mosaic-PICASSO [@cang2024mosaic] are applied as offline analysis steps. For virtual staining, FalseColor-Python [@serafin2020falsecolor] is an offline Python package that renders digital H&E from two-channel fluorescence images with a Beer–Lambert model.

hyperOSD is a lightweight viewer designed specifically for hyperspectral fluorescence imaging: a dataset is a static folder on a web server, and viewing it needs only a browser. Viv already recolours channels on the GPU in the browser; hyperOSD extends live, in-browser rendering from colour mapping to the full processing chain. In hyperOSD, unmixing, blending, tone mapping, denoising and virtual staining all run on the GPU, so every parameter, including the unmixing matrix and the stain colours, can be adjusted while the image is being viewed. The stages, in order, are:

1. unmixing (optional): linear, with a matrix editable in the viewer and a spectrum inspector that picks spectra from pixels; mutual-information-based PICASSO, with the matrix generated in the browser from the current field of view; or linear followed by PICASSO;
2. additive blending of up to 16 channels per z-plane, with knee-curve or Reinhard tone mapping;
3. guided-filter denoising (optional);
4. Beer–Lambert virtual H&E or Masson trichrome staining with editable stain colours (optional).

Because these stages change the per-pixel shader and the tile-compositing path, and must re-run on tiles already held on the GPU whenever a parameter changes, we built them as a custom OpenSeadragon drawer rather than as a plugin for an existing viewer. OpenSeadragon's tile scheduling and Deep Zoom support are kept unchanged. The drawer, the PICASSO module and the annotator are separate JavaScript files, and other OpenSeadragon applications can reuse them.

# Software design

**Rendering.** `HyperBlendWebGLDrawer` (\autoref{fig:pipeline}a), an OpenSeadragon `DrawerBase` subclass, caches visible tiles as GPU textures and composites them into one framebuffer per source. Each RGBA source carries four channels, for up to 16 channels per z-plane (12 when the tiles are RGB). A shader multiplies each enabled channel's intensity by its colour and gain, sums the products and tone-maps the result. The knee curve leaves a pixel unchanged while its brightest colour channel is below 0.8 and compresses it above, keeping colour ratios; the Reinhard operator [@reinhard2002photographic] compresses at all intensities. Changes to colour, gain, tone mapping, staining or denoising re-run only the final passes; linear-unmixing and PICASSO matrix changes re-composite the tiles. Adjacent z-planes are prefetched on the first z-slider move, or once the visible plane has loaded and the network is idle. Without WebGL, the viewer falls back to a CPU blend built on the NIST OpenSeadragonFiltering plugin [@vandecreme2024filtering].

**Unmixing and denoising.** The passes run in the order unmixing, blend, denoise, stain; all but the blend are optional. Linear unmixing applies a pseudoinverse computed in JavaScript, for up to 16 inputs and 2–8 outputs. PICASSO applies an N×N matrix with N ≤ 8. Denoising uses a guided filter with the image as its own guide [@he2013guided].

**Virtual staining.** The stains use the Beer–Lambert transillumination model and the default attenuation k = 2.5 of @giacomelli2016virtual but read the blended, tone-mapped colour channels, not raw fluorescence. H&E takes blue as nuclear density and green as stromal density; our Masson trichrome adds a third stain, with green as nuclear, red as stromal and blue as collagen density. The hues assigned in the blend therefore decide which channels act as which stain. Their default colours are our choice and editable; a dataset can set its own stain style in the configuration block.

**PICASSO in the browser.** "Generate from current view" reads the field of view back from the GPU into a Web Worker running our JavaScript implementation of the mosaic-PICASSO optimiser [@cang2024mosaic], which builds on PICASSO [@seo2022picasso] and which we adapted to hyperspectral data [@huang2025hyperpicasso]. It minimises normalised mutual information, computed from 2-D histograms of the pixels in the tiles with the lowest structural similarity (SSIM) [@wang2004image], using a bounded golden-section line search [@kiefer1953sequential]. The matrix fills the editor to be applied, edited or saved.

**Annotation.** `OSDAnnotator` keeps a multi-class indexed label canvas aligned to the image, with polygon and brush tools and stroke-level undo/redo. It saves and reloads one binary PNG per class plus a JSON class manifest; a single label PNG can also be loaded and traced into polygons, simplified Visvalingam–Whyatt-style [@visvalingam1993line].

**Tiling.** The web app (`dzi_app.py`) accepts single-channel TIFFs, multi-page TIFF stacks and RGB(A) images; the CLI (`generate_dzi.py`) accepts single-channel TIFFs and RGB(A) images. Single channels are packed four per RGBA source, 16-bit data becomes 8-bit, and libvips writes 1024-pixel Deep Zoom pyramids. The web app also writes the viewer page, which needs the `openseadragon/` script folder beside it.

**Quality control.** A deterministic synthetic demo (16 channels, two z-planes) opens as shipped. GitHub Actions tests the tile pipeline, viewer-page generation, the demo data and headless-browser rendering on pushes to `main` and on pull requests.

![hyperOSD rendering pipeline. **(a)** Data flow from microscope TIFFs to the screen; dashed boxes are optional GPU passes that are off by default. **(b)** The twelve raw spectral channels with signal (of the detector's 16; the first four are empty and hidden) of a mouse knee section at the femoral growth plate [@huang2025highthroughput] (each channel contrast-stretched for display; the framed channels 2, 5, 7 and 12, numbered among the twelve shown, are the ones blended, in their assigned hues) and the same field of view rendered in the viewer as a four-channel blended composite, a virtual H&E stain and a virtual Masson trichrome stain.\label{fig:pipeline}](figure1.png)

# Research impact statement

Our group used hyperOSD to view and share the hyperspectral data of two published studies: the hyperPICASSO unmixing method [@huang2025hyperpicasso] and high-throughput hyperspectral imaging with a silicon photomultiplier array [@huang2025highthroughput]. A public deployment serves the mouse knee dataset from the second study, the data shown in \autoref{fig:pipeline}b (<https://imstore.circ.rochester.edu/hyperOSD/zstackHyper.html>). So far, all use has been within our group.

# AI usage disclosure

Generative AI tools were used as follows:

- **Claude** (Anthropic), through Claude Code, March to November 2026. Models: Claude Opus 4.6, 4.7, 4.8, 5 and 5.5; Sonnet 4.6; Haiku 4.5 and 5.5; Fable 5 and 5.1. Claude wrote and refactored code, wrote tests and the figure-capture scripts, and drafted documentation. Multi-agent Claude workflows ran proposal, coding and audit rounds.
- **Claude GitHub app** (`claude[bot]`) posted automated review comments on public pull requests.
- **Codex** (OpenAI), mostly with GPT-5.5 and GPT-6, audited code as an independent reviewer.
- **Qwen3.8-27B**, run locally (vLLM, via opencode), ran automated code and documentation checks and some audits.

For this paper, Claude wrote an initial draft. Claude agents then fact-checked each section: claims against the code, citations against Crossref and publisher records, and descriptions of other tools against their documentation. Under the first author's direction, Claude agents redrafted and edited the wording. The first author reviewed and approved every change.

The first author (C.Z.H.) set the scientific and software design, chose and verified the algorithms, reviewed and edited all AI-generated code and text, and validated behaviour with the automated test suite (including pixel-level render tests; the headless-browser tests run in CI) and by visual inspection of real datasets. The authors take full responsibility for the software and this paper.

# Acknowledgements

This work was supported by the National Institutes of Health (R37-CA258376).

# References
