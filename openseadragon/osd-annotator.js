/**
 * OSDAnnotator — Increment 2 (PAINTING)
 *
 * Clean-room paintbrush-annotation overlay for OpenSeadragon 6.0.2.
 * Reads ONLY the OSD public API: viewer.canvas, viewer.world (add-item/remove-item,
 * getItemAt/getItemCount), viewer events (update-viewport/animation/resize),
 * viewer.forceRedraw, and TiledImage.imageToViewerElementCoordinates.
 *
 * NO dependency on hyperOSD globals, the NIST filter bundle, or hyperblend-webgl.js.
 *
 * Inc1 scope: module skeleton + _view overlay + _mask alloc/deferred-init +
 * basis-vector affine _render + viewport-event repaint + drawer-swap re-mount +
 * lazy-z freeze/unfreeze. NO painting (Inc2), NO export (Inc3), NO host-HTML edit.
 *
 * Inc2 scope: pointer FSM, brush/eraser real, setBrushSize real, filled-disc dab
 * raster + interpolation, flip-aware pointer->image->mask mapping, _scheduleRender
 * rAF coalescer + destroy-cancel, _maskDirty on first paint.
 *
 * Spec: doc/specs/2026-06-14-osd-annotator-inc1-build-spec.md (8 GATE-1 fixes folded).
 *       doc/specs/2026-06-15-osd-annotator-inc2-build-spec.md (Inc2 implementation).
 */
(function (global) {
    'use strict';

    var DEFAULTS = { maxMaskEdge: 4096, brushSize: 25, tool: 'polygon', active: false, maxClasses: 16, undoDepth: 20 };

    // ---- Inc-A: multi-class registry constants ----
    var MAX_CLASSES = 16;
    var PALETTE = ['#4a7c8a', '#c0504d', '#9bbb59', '#8064a2', '#4bacc6', '#f79646',
                   '#2c4d75', '#a5a5a5', '#d99694', '#7f6084', '#4f81bd', '#c3d69b'];

    // ---- Inc-15: raster → vector tracing constants (D3/D4) — NOT options, NOT runtime-tunable ----
    var TRACE_TAU = 2.0;        // VW altitude floor, mask px (Inc-17 D3: 2.0 halves the handle count at <= ~1 % round-trip error; 3.0 destroys r8 blobs)
    var TRACE_CAP = 24;         // hard vertex cap per ring (outer AND hole), after the τ phase
    var TRACE_MIN_AREA = 16;    // mask px²: components below are DROPPED, holes below are FILLED

    // ---- Inc-7: polygon freehand travel threshold (CSS px) — HARD-CODED, not an option ----
    var POLY_DRAG_PX = 6;

    // ---- Inc-8: vertex handle square side AND grab radius (CSS px) — HARD-CODED, not an option ----
    var POLY_HANDLE_PX = 8;
    // ---- Inc-8: selection outline + active-handle colour (QuPath PathPrefs.colorSelectedObject) ----
    var POLY_SEL_COLOR = '#ffff00';

    // ---- Inc-9: typing-guard negative list (§4). <input> types that are NOT text entry, i.e. the
    // ONLY ones whose focus must NOT swallow the annotator's keys. Polarity is deliberate: an
    // unknown/future type resolves to 'text' behaviour in browsers, so it stays GUARDED by default.
    var NON_TEXT_INPUT_TYPES = /^(button|checkbox|radio|range|color|file|submit|reset|image|hidden)$/;

    function attach(viewer, options) {
        var opts = {};
        var k;
        for (k in DEFAULTS) { if (DEFAULTS.hasOwnProperty(k)) { opts[k] = DEFAULTS[k]; } }
        if (options) { for (k in options) { if (options.hasOwnProperty(k)) { opts[k] = options[k]; } } }

        // ---- instance fields ----
        var _viewer = viewer;
        var _canvas = viewer.canvas;          // the .openseadragon-canvas DIV
        var _view = null;                     // overlay <canvas>
        var _ctx = null;                      // _view 2D context
        var _mask = null;                     // offscreen mask <canvas> (image-space)
        var _mctx = null;                     // _mask 2D context
        var _refTI = null;                    // reference TiledImage (getItemAt(0))
        var _maskScale = 1;
        var _imgW = 0, _imgH = 0, _maskW = 0, _maskH = 0;
        var _active = opts.active;
        var _maskDirty = false;
        var _frozen = false;
        var _destroyed = false;
        var _firedWarnKeys = new Set();       // fired-key set backing for _warnOnce
        var _handlers = [];                   // {target,event,fn} for teardown

        // ---- E0: Inc2 instance fields ----
        var _tool = opts.tool;
        var _brushImg = opts.brushSize;       // SCREEN-space diameter (CSS px); footprint scales with zoom
        var _painting = false;
        var _activePointerId = null;
        var _lastPt = null;                   // {x,y} in MASK px
        var _rafId = null;
        // ---- right-drag PAN (secondary button pans the viewer while annotating) ----
        var _panning = false;
        var _panPointerId = null;
        var _panLastX = 0, _panLastY = 0;     // last client coords during a pan drag

        // ---- E-INC4: appearance + hover state ----
        var _color = '#4a7c8a';        // default fill/outline color (matches .osd-ann-dot)
        var _fillAlpha = 0.6;          // translucent fill opacity (R1 default ~60%)
        var _outlineW = 3;             // outline thickness in CSS px (constant screen-space hairline)
        var _lastSPerImg = 1;          // CSS px per 1 image px, captured each _render (R3 ring)
        var _hoverCss = null;          // {x,y} last hover pos in CSS px rel to _view, or null

        // ---- Inc-A: multi-class registry state ----
        var _classes = [];             // [{id,name,color}]
        var _activeClassId = 0;        // 1 after init
        var _silh = {};                // id -> {canvas, ctx}  GROUND TRUTH (Inc-14): one binary
                                       // plane per class; classes may overlap. _mask below it is a
                                       // demoted staging/sentinel canvas — pixels not maintained.
        var _scratch = null, _sctx = null;   // ONE reusable _view-sized per-class composite scratch

        // ---- Inc-B: stroke-level undo/redo state (non-frozen) ----
        var _undoStack = [];           // [{x,y,w,h, before:ImageData, after:ImageData}]  bbox in MASK px
        var _redoStack = [];
        var _undoPre   = null;         // Inc-14: the ONE RETAINED SPARE shadow <canvas> (§7.2)
        var _upctx     = null;         // its 2d ctx (imageSmoothingEnabled=false)
        var _strokeBBox = null;        // {x0,y0,x1,y1} inclusive, accumulated during the current stroke
        // ---- Inc-14 (M1/§7): COPY-ON-WRITE per-plane undo state (non-frozen) ----
        // _histBegin is called argument-less from the FROZEN _onPointerDown and cannot know which
        // plane a stroke will touch, so the pre-image is taken lazily, per plane, at that plane's
        // FIRST write (_shadow). At _histCommit all but ONE canvas is released back to _undoPre —
        // steady-state memory equals today's single pre-stroke canvas.
        var _shadowPool = {};          // String(sid) -> {canvas, ctx}  pre-image of _silh[sid] this stroke
        var _strokeSilhIds = [];       // plane ids shadowed during the current stroke, in first-touch order

        // ---- Inc-7: polygon draft state (IN-PROGRESS draft only — there is no committed-polygon list) ----
        // The polygon is an INPUT METHOD for the indexed label mask: on close it rasterizes into
        // _mask exactly as the brush does. ONE representation (pixels), no sidecar, no vectors.
        var _polyDraft = null;         // null | {mode:'click'|'freehand', pts:[{x,y}]} — pts in MASK px, ring implicitly closed
        var _polyDownCss = null;       // null | {x,y} CSS px rel _view — current press origin (travel threshold)
        var _polyLastCss = null;       // null | {x,y} CSS px — last KEPT vertex (freehand spacing origin)
        var _polyCurMask = null;       // null | {x,y} mask px — latest freehand sample (release-tail vertex)
        var _polyLastClick = null;     // null | {t,x,y} previous click-mode press (double-click detection)
        var _polyOpts = { dblClickMs: 300, dblClickPx: 6, stepPx: 8 };   // the three setPolygonOptions tunables
        var _polyLastCommit = false;   // did the most recent _closeDraft() write pixels? (a flag, NOT a vertex record)
        var _renderCount = 0;          // incremented once per _render() that passes the bail (test hook)

        // ---- Inc-8: committed-polygon EDIT RECORDS + selection + in-progress vertex drag ----
        // The mask stays AUTHORITATIVE: these records are an EDIT AFFORDANCE only. They are never
        // rendered as fill, never re-asserted at export/save/fill time, and may legally diverge
        // from the mask (brush over, eraser through, a later polygon on top, undo). Their ONLY
        // power is the symmetric-difference repaint of their OWN class when a vertex is dragged.
        var _polys = [];               // [{id,classId,pts:[{x,y}],holes?:[[{x,y}]],traced?:true}] — array order =
                                       // z-order (last = topmost). Inc-15: `holes` (inner rings, NOT editable) and
                                       // `traced` (machine-made marker) are OPTIONAL and absent on hand-drawn records.
                                       // Inc-16: a brush/eraser stroke that meets a `traced` record re-traces the
                                       // neighbourhood (`_retraceStroke`); hand-drawn records are never re-traced.
        // ---- Inc-15 (§4.7): diagnostics record of the LAST _trace call — NOT state (I-28); read-only
        // via __lastTraceStats. traceActiveClass overwrites `ms` with its end-to-end time (§5.2).
        var _lastTraceStats = { regions: 0, holes: 0, dropped: 0, holesFilled: 0, capped: 0, ms: 0 };
        // ---- Inc-16 (§5.7): end-to-end diagnostics of STROKE re-traces — NOT state; read-only via
        // __retraceStats. _lastTraceStats stays the tracer's own (overwritten per _trace call).
        var _retraceStats = { runs: 0, ms: 0, crop: null, grown: 0 };
        var _polySeq = 0;              // monotonic id source (++_polySeq per append); reset only by destroy
        var _selPoly = null;           // null | record reference into _polys — the selected polygon
        var _dragVert = null;          // null | {poly, idx, pts, downCss, moved} — pts is a WORKING COPY (no mask
                                       // write); downCss/moved: Inc-12 deadzone (CSS press origin + sticky crossed flag)
        var _selVert = null;           // Inc-13: null | {polyId, idx} — the SELECTED anchor (yellow highlight +
                                       // Backspace target). polyId makes stale values inert: an id mismatch means
                                       // nothing highlights and nothing removes — never the wrong anchor destroyed.
        var _polyScratch = null, _psctx = null;   // reusable full-mask binary scratch (silhouette-clipped erase)
        // ---- Inc-9 (R3): Alt-hold temporary-eraser latch. Non-null ⇔ an Alt hold is latched and the
        // saved tool must be restored on keyup / window blur. Under the Inc-9 mode split the latch can
        // only engage from Brush mode, so the ONLY value ever stored is 'brush'; it is kept as a
        // SAVED-TOOL variable (not a boolean) so the restore reads setTool(_altSavedTool) and the
        // mechanism survives any future mode being added.
        var _altSavedTool = null;                 // null | 'brush'

        // ---- Inc-11 (A2/A3/M1/M2): eraser-protection cache + null-op suppression flag ----
        // _protectRows: row y -> flat MERGED half-open [x0,x1) interval pairs covering the
        // pixels of ANY committed polygon on that row, EVERY class (A3). Rebuilt LAZILY in
        // _protectFilter whenever the derived signature (M1) differs — no mutation site is
        // ever asked to remember to invalidate it, including in-place rec.pts writes and a
        // mask resize. An open draft contributes nothing (it has written no pixels).
        var _protectRows = null;                  // null | Object.create(null) of y -> pairs
        var _protectSig1 = 0, _protectSig2 = 0;   // M1 two-lane 32-bit signature
        // M2: true iff some _protectFilter call this stroke EMITTED a non-empty run list
        // (zero-polygon pass-through INCLUDED). Cleared by _histBegin; read by _histCommit.
        var _eraserDidErase = false;
        var _sigF64 = new Float64Array(1);        // exact float-bit folding for the signature
        var _sigI32 = new Int32Array(_sigF64.buffer);

        function _warnOnce(key, msg) {
            if (_firedWarnKeys.has(key)) { return; }
            _firedWarnKeys.add(key);
            if (typeof console !== 'undefined' && console.warn) { console.warn(msg); }
        }

        function _registerHandler(target, event, fn) {
            target.addHandler(event, fn);
            _handlers.push({ target: target, event: event, fn: fn });
        }

        // ---- B3: _view mount (FIX 1 + FIX 4) ----
        function _assertStack() {
            // FIX 4: exclude _view itself so the drawer's removeChild transient
            // never selects _view and grows z-index unbounded across swaps.
            var dc = _canvas.querySelector('canvas:not(.osd-annotator-view)');
            _view.style.zIndex = dc
                ? String((Number.parseInt(getComputedStyle(dc).zIndex, 10) || 0) + 1)
                : '1';
        }

        function _ensureMounted() {
            if (_view.parentNode !== _canvas) { _canvas.appendChild(_view); }
            _assertStack();
        }

        function _mount() {
            _view = document.createElement('canvas');
            _view.style.position = 'absolute';
            _view.style.left = '0';
            _view.style.top = '0';
            _view.style.width = '100%';
            _view.style.height = '100%';
            _view.style.pointerEvents = _active ? 'auto' : 'none';
            _view.className = 'osd-annotator-view';
            // FIX 1: assign the visible 2D context at mount; bail in _render if null.
            _ctx = _view.getContext('2d');
            if (!_ctx) { _warnOnce('no-2d-context', 'OSDAnnotator: 2D context unavailable — overlay rendering disabled'); }
            _assertStack();
            _canvas.appendChild(_view);
        }

        // ---- B4: deferred init ----
        function _onAddInit() {
            // one-shot: remove ITSELF from world + _handlers, then init.
            _viewer.world.removeHandler('add-item', _onAddInit);
            for (var i = 0; i < _handlers.length; i++) {
                if (_handlers[i].fn === _onAddInit) { _handlers.splice(i, 1); break; }
            }
            _initImage();
        }

        function _tryInit() {
            if (_viewer.world.getItemCount() > 0) {
                _initImage();
            } else {
                _registerHandler(_viewer.world, 'add-item', _onAddInit);
            }
        }

        function _initImage() {
            _refTI = _viewer.world.getItemAt(0);
            var dim = _refTI.source.dimensions;
            _imgW = dim.x;
            _imgH = dim.y;
            _maskScale = Math.min(1, opts.maxMaskEdge / Math.max(_imgW, _imgH));
            _maskW = Math.round(_imgW * _maskScale);
            _maskH = Math.round(_imgH * _maskScale);
            _mask = document.createElement('canvas');
            _mask.width = _maskW;
            _mask.height = _maskH;
            _mctx = _mask.getContext('2d');
            _mctx.imageSmoothingEnabled = false;   // set once; _mask never resized in Inc1
            _frozen = false;
            // ---- Inc-A: seed the default class registry ----
            _classes = [{ id: 1, name: 'Class 1', color: PALETTE[0] }];
            _activeClassId = 1;
            // First _render must run inside a real frame so imageToViewerElementCoordinates
            // returns laid-out coords (not NaN). forceRedraw → update-viewport → _render.
            _viewer.forceRedraw();
        }

        // ---- B5: _render() + affine (FIX 2 + FIX 5) ----
        function _render() {
            if (_destroyed || _frozen || !_mask || !_refTI || !_ctx) { return; }  // FIX 1 bail
            _renderCount++;                                                      // Inc-7 test hook
            var DPR = window.devicePixelRatio || 1;
            var w = _canvas.clientWidth, h = _canvas.clientHeight;
            // FIX 5: reset BOTH dimensions on width- OR height-only resize, BEFORE transform.
            if (_view.width !== Math.round(w * DPR) || _view.height !== Math.round(h * DPR)) {
                _view.width = Math.round(w * DPR);
                _view.height = Math.round(h * DPR);
            }
            var P = OpenSeadragon.Point;
            var o = _refTI.imageToViewerElementCoordinates(new P(0, 0));
            var ex = _refTI.imageToViewerElementCoordinates(new P(1, 0));
            var ey = _refTI.imageToViewerElementCoordinates(new P(0, 1));
            var a = (ex.x - o.x) * (_imgW / _maskW);
            var b = (ex.y - o.y) * (_imgW / _maskW);
            var c = (ey.x - o.x) * (_imgH / _maskH);   // FIX 2: (ey.x - o.x)
            var d = (ey.y - o.y) * (_imgH / _maskH);
            _ctx.setTransform(1, 0, 0, 1, 0, 0);
            _ctx.clearRect(0, 0, _view.width, _view.height);
            // Viewport FLIP handling (verified against OSD 6.0.2):
            // viewport._pixelFromPoint (openseadragon.js:27881) accounts for ROTATION
            // ONLY — NOT flip — so the basis vectors ex/ey above are UN-flipped. The
            // drawer applies flip as the OUTERMOST transform at draw start (canvas
            // drawer: getFlip() check → _flip() at ~23621, BEFORE _setRotations),
            // mirroring about the canvas centre (_flip → _getCanvasCenter:
            // translate(cx,0); scale(-1,1); translate(-cx,0)). To co-register the
            // overlay with the flipped image we re-apply that SAME centre mirror as the
            // outermost transform, composed with the affine. (This is NOT a double flip:
            // the coordinate API does not encode flip, so without this the overlay would
            // render un-mirrored over a mirrored image.)
            var vp = _viewer.viewport;
            if (vp && vp.getFlip && vp.getFlip()) {
                _ctx.setTransform(-1, 0, 0, 1, _view.width, 0);   // x -> _view.width - x (mirror about centre)
                _ctx.transform(a * DPR, b * DPR, c * DPR, d * DPR, o.x * DPR, o.y * DPR);
            } else {
                _ctx.setTransform(a * DPR, b * DPR, c * DPR, d * DPR, o.x * DPR, o.y * DPR);
            }
            _ctx.imageSmoothingEnabled = false;        // setTransform above clears ctx state
            _lastSPerImg = Math.hypot(ex.x - o.x, ex.y - o.y);   // CSS px per 1 image px (R3 ring)
            // ---- R1 (v2): crisp OUTLINE FIRST — CONSTANT SCREEN-SPACE hairline ----
            // Prior version offset-dilated in MASK px, so the rim ballooned when zoomed IN
            // and went sub-pixel/faint when zoomed OUT. Here the dilation is done in DEVICE
            // px (the offset added to the transform's e/f translation, NOT to the mask
            // coordinates), so the rim is a fixed ~_outlineW CSS-px band at every zoom.
            // The base affine [A,B,C,D | E,F] (device space, flip folded in) is rebuilt so a
            // device-px offset (offX,offY) is just (E+offX, F+offY).
            var A, B, C, D, E0, F0;
            if (vp && vp.getFlip && vp.getFlip()) {
                // net of the centre-mirror composed with the affine (see FLIP note above)
                A = -a * DPR; B = b * DPR; C = -c * DPR; D = d * DPR;
                E0 = _view.width - o.x * DPR; F0 = o.y * DPR;
            } else {
                A = a * DPR; B = b * DPR; C = c * DPR; D = d * DPR;
                E0 = o.x * DPR; F0 = o.y * DPR;
            }
            var rimDev = _outlineW * DPR;                        // hairline band width in DEVICE px
            var NDIR = 12;                                       // dense enough that the ring has no gaps
            // ---- Inc-A: per-class scratch-isolated fill+outline (avoids whole-canvas cross-tint) ----
            // Each class's fill+outline is composited onto ONE reusable _scratch canvas (the
            // whole-canvas source-atop tints would cross-recolor earlier classes if run on _ctx),
            // then blitted onto _view. Per-class binary silhouettes (se.canvas, _maskW×_maskH) are
            // the SHAPE source, kept in sync incrementally at stroke time — ZERO getImageData here.
            if (!_scratch) { _scratch = document.createElement('canvas'); _sctx = _scratch.getContext('2d'); }
            if (_sctx && (_scratch.width !== _view.width || _scratch.height !== _view.height)) {
                _scratch.width = _view.width; _scratch.height = _view.height;
            }
            if (_sctx) {
                for (var ci = 0; ci < _classes.length; ci++) {
                    // LAW C (Inc-14): PASS 1 accumulates each class's tint at alpha 1 with 'lighter'
                    // (colour-additive, order-independent — classes may overlap), then ONE
                    // 'destination-in' fillRect at _fillAlpha sets a CONSTANT overlay alpha wherever
                    // any class is present. PASS 2 draws the rims opaque on top. The passes are split
                    // because the alpha normalise must apply to the fill layer ALONE.
                    var cls = _classes[ci];
                    var se = _silh[cls.id];
                    if (!se || !se.ctx) { continue; }            // no pixels / no ctx → skip
                    _sctx.setTransform(1, 0, 0, 1, 0, 0);
                    _sctx.clearRect(0, 0, _scratch.width, _scratch.height);
                    _sctx.imageSmoothingEnabled = false;
                    _sctx.globalAlpha = 1;
                    _sctx.globalCompositeOperation = 'source-over';
                    // (1) the silhouette shape at FULL alpha (base transform) — NO globalAlpha here:
                    // the single normalise below owns the overlay alpha
                    _sctx.setTransform(A, B, C, D, E0, F0);      // base (un-offset) transform
                    _sctx.drawImage(se.canvas, 0, 0);
                    // (2) tint fill → cls.color
                    _sctx.setTransform(1, 0, 0, 1, 0, 0);        // identity: tint whole canvas
                    _sctx.globalCompositeOperation = 'source-atop';
                    _sctx.fillStyle = cls.color;
                    _sctx.fillRect(0, 0, _scratch.width, _scratch.height);
                    // (3) restore scratch defaults
                    _sctx.globalCompositeOperation = 'source-over';
                    _sctx.globalAlpha = 1;
                    // (4) ACCUMULATE this class's tint onto _view — 'lighter' is commutative and
                    // associative under clamping, so the fill blend is order-independent
                    _ctx.setTransform(1, 0, 0, 1, 0, 0);
                    _ctx.globalCompositeOperation = 'lighter';
                    _ctx.globalAlpha = 1;
                    _ctx.drawImage(_scratch, 0, 0);
                }
                // ---- THE CONSTANT-ALPHA STEP: ONE uniform normalise over the whole fill layer ----
                // Every pass-1 contribution landed at alpha 1, so accumulated alpha is 255 wherever
                // ANY class covers and 0 elsewhere. 'destination-in' against a uniform alpha source
                // is a per-pixel premultiplied scalar multiply: the result is exactly
                // round(255 * _fillAlpha) wherever a class is present — never 306-clamped — and the
                // colour ratios survive. fillStyle is opaque because destination-in ignores source RGB.
                _ctx.setTransform(1, 0, 0, 1, 0, 0);
                _ctx.globalCompositeOperation = 'destination-in';
                _ctx.globalAlpha = _fillAlpha;
                _ctx.fillStyle = '#fff';
                _ctx.fillRect(0, 0, _view.width, _view.height);
                _ctx.globalAlpha = 1;
                // ---- PASS 2 — rims: opaque, ON TOP of every fill, registry order ----
                for (var ri = 0; ri < _classes.length; ri++) {
                    var rcls = _classes[ri];
                    var rse = _silh[rcls.id];
                    if (!rse || !rse.ctx) { continue; }          // no pixels / no ctx → skip
                    _sctx.setTransform(1, 0, 0, 1, 0, 0);
                    _sctx.clearRect(0, 0, _scratch.width, _scratch.height);
                    _sctx.imageSmoothingEnabled = false;
                    _sctx.globalAlpha = 1;
                    _sctx.globalCompositeOperation = 'source-over';
                    // (1) 12-dir dilation: build the outward rim in DEVICE space
                    for (var di = 0; di < NDIR; di++) {
                        var ang = (Math.PI * 2 * di) / NDIR;
                        _sctx.setTransform(A, B, C, D, E0 + Math.cos(ang) * rimDev, F0 + Math.sin(ang) * rimDev);
                        _sctx.drawImage(rse.canvas, 0, 0);       // dilate silhouette in DEVICE space
                    }
                    // (2) recolor dilated silhouette → rcls.color
                    _sctx.setTransform(1, 0, 0, 1, 0, 0);        // identity: tint whole canvas
                    _sctx.globalCompositeOperation = 'source-atop';
                    _sctx.fillStyle = rcls.color;
                    _sctx.fillRect(0, 0, _scratch.width, _scratch.height);
                    // (3) punch out interior → outward rim only
                    _sctx.setTransform(A, B, C, D, E0, F0);      // base (un-offset) transform
                    _sctx.globalCompositeOperation = 'destination-out';
                    _sctx.drawImage(rse.canvas, 0, 0);
                    // (4) restore scratch defaults
                    _sctx.globalCompositeOperation = 'source-over';
                    _sctx.globalAlpha = 1;
                    // blit this class's rim onto _view, opaque, above every fill
                    _ctx.setTransform(1, 0, 0, 1, 0, 0);
                    _ctx.globalCompositeOperation = 'source-over';
                    _ctx.globalAlpha = 1;
                    _ctx.drawImage(_scratch, 0, 0);
                }
            }
            _ctx.setTransform(1, 0, 0, 1, 0, 0);                 // reset to defaults before the ring
            _ctx.globalCompositeOperation = 'source-over';       // restore default
            _ctx.globalAlpha = 1;
            // ---- Inc-7: polygon DRAFT preview (identity transform, device px) ----
            // Vertices are stored in MASK px and pushed through the CURRENT A..F0 here, so the
            // preview stays co-registered under pan/zoom/rotate/flip with zero extra plumbing
            // (_onView re-renders on every viewport event). Line widths are screen-constant.
            // Antialiasing is fine in this PREVIEW — the mask write is the integer scanline in
            // _polyRuns/_polyCommit. No preview fill of the open draft.
            if (_active && _polyDraft && _polyDraft.pts.length > 0) {
                var DPRp = window.devicePixelRatio || 1;
                var pcol = _activeColor();
                var dpts = [];
                for (var pi = 0; pi < _polyDraft.pts.length; pi++) {
                    var pv = _polyDraft.pts[pi];
                    dpts.push({ x: A * pv.x + C * pv.y + E0, y: B * pv.x + D * pv.y + F0 });
                }
                _ctx.setTransform(1, 0, 0, 1, 0, 0);
                _ctx.globalCompositeOperation = 'source-over';
                _ctx.globalAlpha = 1;
                _ctx.setLineDash([]);
                // (a) solid polyline through the placed vertices (halo under the class-color line)
                if (dpts.length >= 2) {
                    _ctx.beginPath();
                    _ctx.moveTo(dpts[0].x, dpts[0].y);
                    for (var li = 1; li < dpts.length; li++) { _ctx.lineTo(dpts[li].x, dpts[li].y); }
                    _ctx.lineWidth = 4 * DPRp;
                    _ctx.strokeStyle = 'rgba(0,0,0,0.6)';
                    _ctx.stroke();
                    _ctx.lineWidth = 2 * DPRp;
                    _ctx.strokeStyle = pcol;
                    _ctx.stroke();
                }
                // (b) vertex markers
                _ctx.fillStyle = pcol;
                var vSide = 5 * DPRp;
                for (var vi = 0; vi < dpts.length; vi++) {
                    _ctx.fillRect(dpts[vi].x - vSide / 2, dpts[vi].y - vSide / 2, vSide, vSide);
                }
                // (c) rubber band: ONE dashed segment from the last vertex to the cursor
                if (_hoverCss) {
                    var lastD = dpts[dpts.length - 1];
                    _ctx.setLineDash([6 * DPRp, 4 * DPRp]);
                    _ctx.strokeStyle = pcol;
                    _ctx.lineWidth = 2 * DPRp;
                    _ctx.beginPath();
                    _ctx.moveTo(lastD.x, lastD.y);
                    _ctx.lineTo(_hoverCss.x * DPRp, _hoverCss.y * DPRp);
                    _ctx.stroke();
                }
                // (c2) Inc-8: CLOSING EDGE preview — LAST placed vertex → FIRST vertex, dashed at
                // half alpha. Drawn whether or not the pointer is over the canvas (independent of
                // _hoverCss); it and the rubber band are two distinct segments and may both show.
                // Still NO preview fill of the open draft.
                if (dpts.length >= 2) {
                    _ctx.setLineDash([6 * DPRp, 4 * DPRp]);
                    _ctx.strokeStyle = pcol;
                    _ctx.lineWidth = 2 * DPRp;
                    _ctx.globalAlpha = 0.5;
                    _ctx.beginPath();
                    _ctx.moveTo(dpts[dpts.length - 1].x, dpts[dpts.length - 1].y);
                    _ctx.lineTo(dpts[0].x, dpts[0].y);
                    _ctx.stroke();
                    _ctx.globalAlpha = 1;
                }
                // (d) restore defaults for whatever draws next
                _ctx.setLineDash([]);
                _ctx.globalCompositeOperation = 'source-over';
                _ctx.globalAlpha = 1;
            }
            // ---- Inc-8: SELECTION outline + vertex HANDLES (identity transform, device px) ----
            // Vertices are MASK px pushed through the CURRENT A..F0, so handles stay co-registered
            // under pan/zoom/rotate/flip. NO vector fill — the fill on screen is the class
            // silhouette's (i.e. the mask's). Line widths and the handle square are screen-constant.
            // _selPoly is only ever non-null in the polygon tool, so no _tool test is needed.
            if (_active && _selPoly) {
                var DPRs = window.devicePixelRatio || 1;
                var spts = _dragVert ? _dragVert.pts : _selPoly.pts;   // the working copy during a drag
                var sdev = [];
                for (var qi = 0; qi < spts.length; qi++) {
                    var qv = spts[qi];
                    sdev.push({ x: A * qv.x + C * qv.y + E0, y: B * qv.x + D * qv.y + F0 });
                }
                _ctx.setTransform(1, 0, 0, 1, 0, 0);
                _ctx.globalCompositeOperation = 'source-over';
                _ctx.globalAlpha = 1;
                _ctx.setLineDash([]);
                // (a) CLOSED outline (halo under the selection-color line)
                if (sdev.length >= 2) {
                    _ctx.beginPath();
                    _ctx.moveTo(sdev[0].x, sdev[0].y);
                    for (var qli = 1; qli < sdev.length; qli++) { _ctx.lineTo(sdev[qli].x, sdev[qli].y); }
                    _ctx.closePath();
                    _ctx.lineWidth = 4 * DPRs;
                    _ctx.strokeStyle = 'rgba(0,0,0,0.6)';
                    _ctx.stroke();
                    _ctx.lineWidth = 2 * DPRs;
                    _ctx.strokeStyle = POLY_SEL_COLOR;
                    _ctx.stroke();
                }
                // (b) handles: one filled square per vertex (the SELECTED one in POLY_SEL_COLOR)
                var hSide = POLY_HANDLE_PX * DPRs;
                for (var qhi = 0; qhi < sdev.length; qhi++) {
                    var hx0 = sdev[qhi].x - hSide / 2, hy0 = sdev[qhi].y - hSide / 2;
                    _ctx.fillStyle = (_selVert && qhi === _selVert.idx) ? POLY_SEL_COLOR : '#fff';
                    _ctx.fillRect(hx0, hy0, hSide, hSide);
                    _ctx.lineWidth = 1 * DPRs;
                    _ctx.strokeStyle = 'rgba(0,0,0,0.8)';
                    _ctx.strokeRect(hx0, hy0, hSide, hSide);
                }
                // (c) restore defaults for whatever draws next
                _ctx.setLineDash([]);
                _ctx.globalCompositeOperation = 'source-over';
                _ctx.globalAlpha = 1;
            }
            // ---- R3: brush ring cursor (identity transform, screen px) ----
            // SCREEN-SPACE brush: ring radius is CONSTANT on screen (independent of zoom).
            // _brushImg is a CSS-px diameter; the painted image footprint scales with zoom
            // inside _getRPx (screen px ÷ _lastSPerImg), NOT here. Rendered as a bold,
            // high-contrast cursor (dark halo under a bright ring) so it reads on any
            // background and makes brush-size changes obvious.
            // Inc-7: the polygon tool shows NO brush ring (_hoverCss is still tracked, for the band).
            if (_active && _hoverCss && !_frozen && _tool !== 'polygon') {
                var DPRr = window.devicePixelRatio || 1;
                var rDev = (_brushImg * DPRr) / 2;
                if (isFinite(rDev) && rDev > 0) {
                    if (rDev < 1.5) { rDev = 1.5; }                              // floor (tiny brushes)
                    var capR = 4 * Math.max(_view.width, _view.height);
                    if (rDev > capR) { rDev = capR; }                           // sanity cap
                    var hx = _hoverCss.x * DPRr, hy = _hoverCss.y * DPRr;
                    _ctx.setTransform(1, 0, 0, 1, 0, 0);
                    _ctx.globalCompositeOperation = 'source-over';
                    _ctx.globalAlpha = 1;
                    _ctx.beginPath();
                    _ctx.arc(hx, hy, rDev, 0, 2 * Math.PI);
                    _ctx.lineWidth = 4 * DPRr;                                   // dark contrast halo
                    _ctx.strokeStyle = 'rgba(0,0,0,0.6)';
                    _ctx.stroke();
                    _ctx.beginPath();
                    _ctx.arc(hx, hy, rDev, 0, 2 * Math.PI);
                    _ctx.lineWidth = 2 * DPRr;                                   // bright inner ring
                    _ctx.strokeStyle = _activeColor();                          // ring in ACTIVE class color
                    _ctx.stroke();
                }
            }
        }

        // ---- B6: repaint triggers ----
        function _onView() { _ensureMounted(); _render(); }

        // ---- B8: lazy-z guard (FIX 3 + FIX 7) ----
        function _onRemove(e) {
            if (e.item !== _refTI) { return; }
            _refTI = _viewer.world.getItemAt(0) || null;   // try to recover NOW (FIX 3)
            if (_refTI) { _viewer.forceRedraw(); return; }  // replacement present → no freeze
            _frozen = true;
            if (_active) { setActive(false); }              // FIX 7: leave _active=false
            _warnOnce('refti-removed', 'OSDAnnotator: reference image evicted — painting frozen until re-add');
        }

        function _onAddPersist() {
            if (!_frozen) { return; }
            _refTI = _viewer.world.getItemAt(0);
            if (_refTI) {
                _frozen = false;
                _firedWarnKeys.delete('refti-removed');     // re-arm warn for a future eviction
                _viewer.forceRedraw();
                // FIX 7: do NOT auto-restore _active — user must manually re-arm.
            }
        }

        // ---- B1: public methods ----
        function destroy() {
            if (_destroyed) { return; }
            _active = false; _maskDirty = false;
            _painting = false; _activePointerId = null; _lastPt = null;   // FSM reset
            _endPan();                                                    // release any pan capture + reset
            if (_rafId !== null) { window.cancelAnimationFrame(_rafId); _rafId = null; }  // cancel in-flight rAF
            for (var i = 0; i < _handlers.length; i++) {
                var h = _handlers[i];
                if (h.dom) { h.target.removeEventListener(h.event, h.fn); }  // E7(a): DOM listeners
                else { h.target.removeHandler(h.event, h.fn); }              // Inc1 entries (dom absent → OSD)
            }
            _handlers = [];
            if (_view && _view.parentNode) { _view.parentNode.removeChild(_view); }
            // ---- Inc-A: free the multi-class render caches + registry ----
            _scratch = null; _sctx = null;
            _silh = {};
            _classes = []; _activeClassId = 0;
            // ---- Inc-B: free undo/redo state ----
            _undoStack = []; _redoStack = []; _undoPre = null; _upctx = null; _strokeBBox = null;
            _shadowPool = {}; _strokeSilhIds = [];   // Inc-14: drop the COW shadow pool with them
            // ---- Inc-7: drop the polygon draft INLINE (not via _dropDraft — that would schedule
            // an rAF this teardown has already cancelled) ----
            _polyDraft = null; _polyDownCss = null; _polyLastCss = null;
            _polyCurMask = null; _polyLastClick = null; _polyLastCommit = false;
            // ---- Inc-8: drop the edit records, selection, drag and the reshape scratch INLINE
            // (not via _dropDraft — same rAF reason as above) ----
            _polys = []; _polySeq = 0; _selPoly = null; _dragVert = null; _selVert = null;
            _retraceStats = { runs: 0, ms: 0, crop: null, grown: 0 };   // ---- Inc-16 (§5.7): drop the re-trace diagnostics ----
            _polyScratch = null; _psctx = null;
            _altSavedTool = null;                            // ---- Inc-9: drop any latched Alt hold ----
            _protectRows = null; _protectSig1 = 0; _protectSig2 = 0; _eraserDidErase = false;   // ---- Inc-11: drop the protect cache ----
            _viewer = null; _canvas = null; _view = null; _ctx = null;
            _mask = null; _mctx = null; _refTI = null;
            _destroyed = true;
        }

        // ---- E1: canonical radius helper ----
        // _brushImg is a SCREEN-space diameter (CSS px). Convert to MASK px at the CURRENT
        // zoom: screen px -> image px (÷ _lastSPerImg, CSS px per image px) -> mask px (× _maskScale).
        // So the painted image footprint scales with zoom (zoom in = finer), while the on-screen
        // ring stays a constant size (see _render). _lastSPerImg is refreshed every _render.
        function _getRPx() {
            var sPerImg = (_lastSPerImg > 0) ? _lastSPerImg : 1;
            return Math.max(1, Math.round((_brushImg * _maskScale) / (2 * sPerImg)));
        }

        // ---- E2: flip-aware pointer→mask mapping ----
        function _pointerToMask(cssX, cssY) {
            if (_destroyed || !_mask || !_refTI) { return null; }      // OBJ#9 / Codex#11: no-throw post-destroy
            var vp = _viewer.viewport;
            var x = cssX;
            // FLIP INVERSE (verified osd.js 27914-27919 / 29102-29105 / 13036): viewerElementToImage
            // ignores flip; the drawer mirrors about the overlay centre. Un-mirror the pointer x about
            // the OVERLAY width (inverse of Inc1 _render's mirror about _view.width) BEFORE converting.
            if (vp && vp.getFlip && vp.getFlip()) { x = _view.clientWidth - cssX; }
            var ip = _refTI.viewerElementToImageCoordinates(new OpenSeadragon.Point(x, cssY)); // rotation handled here
            var mx = ip.x * _maskScale;
            var my = ip.y * _maskScale;
            var rPx = _getRPx();
            // off-mask-beyond-brush rejection (uses canonical rPx):
            if (mx < -rPx || my < -rPx || mx > _maskW + rPx || my > _maskH + rPx) { return null; }
            return { x: mx, y: my };
        }

        // ---- Inc-A: apply pre-computed integer runs to a 2D ctx (shared by _stampDisc targets) ----
        function _fillRuns(ctx, runs, gco, fill) {
            ctx.globalCompositeOperation = gco;
            ctx.fillStyle = fill;
            for (var i = 0; i < runs.length; i++) {
                ctx.fillRect(runs[i].x, runs[i].y, runs[i].w, 1);
            }
            ctx.globalCompositeOperation = 'source-over';       // restore per-target
        }

        // ---- Inc-11 (A2, PURE): bucket a run list into per-row merged half-open interval
        // pairs. No state reads; shared by _protectRebuild and the __runsSubtract oracle.
        function _mergeRunsToRows(runs) {
            var rows = Object.create(null);
            var i, r;
            for (i = 0; i < runs.length; i++) {
                r = runs[i];
                if (!r || !(r.w > 0)) { continue; }
                (rows[r.y] || (rows[r.y] = [])).push(r.x, r.x + r.w);
            }
            for (var y in rows) {
                var flat = rows[y];
                if (flat.length <= 2) { continue; }               // single interval: already merged
                var iv = [];
                for (i = 0; i + 1 < flat.length; i += 2) { iv.push([flat[i], flat[i + 1]]); }
                iv.sort(function (u, v) { return u[0] - v[0]; });
                var merged = [iv[0][0], iv[0][1]];
                for (i = 1; i < iv.length; i++) {
                    if (iv[i][0] <= merged[merged.length - 1]) {  // overlap or abutment
                        if (iv[i][1] > merged[merged.length - 1]) { merged[merged.length - 1] = iv[i][1]; }
                    } else {
                        merged.push(iv[i][0], iv[i][1]);
                    }
                }
                rows[y] = merged;
            }
            return rows;
        }

        // ---- Inc-11 (A2, PURE): emit the sub-runs of `runs` NOT covered by `rows` (a
        // _mergeRunsToRows result). Half-open arithmetic throughout; never mutates inputs;
        // output preserves the input run order. This is THE complement walk A1 applies.
        function _runsSubtract(runs, rows) {
            var out = [];
            for (var i = 0; i < runs.length; i++) {
                var r = runs[i];
                var row = rows[r.y];
                if (!row || row.length === 0) { out.push({ x: r.x, y: r.y, w: r.w }); continue; }
                var x = r.x, end = r.x + r.w;
                for (var j = 0; j + 1 < row.length && x < end; j += 2) {
                    var p0 = row[j], p1 = row[j + 1];
                    if (p1 <= x) { continue; }
                    if (p0 >= end) { break; }
                    if (p0 > x) { out.push({ x: x, y: r.y, w: p0 - x }); }
                    if (p1 > x) { x = p1; }
                }
                if (x < end) { out.push({ x: x, y: r.y, w: end - x }); }
            }
            return out;
        }

        // ---- Inc-11 (M1): derived geometry signature — it READS THE DATA ITSELF, so no
        // mutation site can be missed: in-place rec.pts writes, loadPolygons repopulation
        // landing on the same length, undo/redo fixups, removeClass's array reassignment,
        // and a mask resize all change it. Inc-14 (§6/A12): the ACTIVE classId IS folded —
        // protection is scoped to the erase target, so the lazy cache re-keys on a class
        // switch. Inc-15: hole rings are folded too — a re-trace that only changes holes
        // must invalidate. Inc-16: the `traced` marker is folded — a record that only flips
        // the marker (hand-drawn → traced, or undo of it) must invalidate.
        // Two independent 32-bit lanes (multipliers 31/37), both must
        // match: integer-only math, no strings, collision odds ~2^-64 per real change.
        // Cost O(total vertices) per ERASER dab, only while a polygon exists — the price
        // of invalidation that cannot be forgotten (per-dab, NEVER hoisted: loadLabel /
        // loadPolygons / clear / removeClass are not _painting-guarded and can replace the
        // geometry mid-stroke through the host's async img.onload).
        function _protectSignature() {
            var h1 = 0, h2 = 0;
            function fold(v) {
                _sigF64[0] = v;
                var lo = _sigI32[0], hi = _sigI32[1];
                h1 = (Math.imul(h1, 31) + lo) | 0; h1 = (Math.imul(h1, 31) + hi) | 0;
                h2 = (Math.imul(h2, 37) + lo) | 0; h2 = (Math.imul(h2, 37) + hi) | 0;
            }
            fold(_polys.length); fold(_maskW); fold(_maskH); fold(_activeClassId);   // Inc-14: re-key on class switch
            for (var i = 0; i < _polys.length; i++) {
                var pp = _polys[i].pts;
                fold(pp.length);
                for (var k = 0; k < pp.length; k++) { fold(pp[k].x); fold(pp[k].y); }
                var hh = _polys[i].holes;
                fold(hh ? hh.length : 0);                                   // Inc-15 (A12): holes re-key the cache
                if (hh) { for (var hi = 0; hi < hh.length; hi++) { var hr = hh[hi]; fold(hr.length);
                    for (var hk = 0; hk < hr.length; hk++) { fold(hr[hk].x); fold(hr[hk].y); } } }
                fold(_polys[i].traced === true ? 1 : 0);                    // Inc-16 (A2): the marker alone re-keys the cache
            }
            return { h1: h1, h2: h2 };
        }

        // ---- Inc-11 (A2/A3), Inc-14 (§6/A12): rebuild the protect set from every committed
        // record OF THE ACTIVE CLASS. The eraser is active-class-only (user decision 2), so the
        // only records a dab can desynchronise are the active class's; guarding the others is
        // over-broad by construction and unrecoverable (class-2 brush paint inside class 1's
        // committed polygon could never be erased by anyone).
        // Cost O(active-class polygon runs) — perimeter-ish — once per invalidation.
        // Degenerate records need no guarding: _polyRuns returns [] for n<3 / non-finite,
        // clamps to the mask, and rasterizes self-intersections under NONZERO winding —
        // the protected set is BY CONSTRUCTION the exact pixel set each record committed.
        function _protectRebuild() {
            var all = [];
            for (var i = 0; i < _polys.length; i++) {
                if (_polys[i].classId !== _activeClassId || _polys[i].traced === true) { continue; }   // Inc-14: active class only; Inc-16: traced records protect nothing
                var runs = _recRuns(_polys[i].pts, _polys[i].holes);   // Inc-15 (I-30): outer − holes
                for (var q = 0; q < runs.length; q++) { all.push(runs[q]); }
            }
            _protectRows = _mergeRunsToRows(all);
        }

        // ---- Inc-11 (A1/A2/M2): filter an eraser dab's runs down to the sub-runs OUTSIDE
        // every committed polygon. Called UNCONDITIONALLY from the eraser branch of
        // _stampDisc. The M2 suppression flag is decided HERE, on the EMITTED list,
        // INCLUDING the zero-polygon pass-through — ResearchA's draft set it only past the
        // early return, which would have discarded every eraser stroke of a brush-only
        // user. Protection is GEOMETRIC, and Inc-14 (§6/A12) SCOPES it to the ACTIVE class:
        // any pixel inside a committed ring OF THE ERASE TARGET is unerasable, whoever painted it.
        function _protectFilter(runs) {
            if (_polys.length === 0) {
                if (runs.length > 0) { _eraserDidErase = true; }
                return runs;                       // brush-only users pay nothing (A2)
            }
            var sig = _protectSignature();         // M1: per dab, reads the data itself
            if (_protectRows === null || sig.h1 !== _protectSig1 || sig.h2 !== _protectSig2) {
                _protectRebuild();
                _protectSig1 = sig.h1; _protectSig2 = sig.h2;
            }
            var out = _runsSubtract(runs, _protectRows);
            if (out.length > 0) { _eraserDidErase = true; }
            return out;
        }

        // ---- E4: filled-disc raster (integer horizontal runs) — indexed multi-class ----
        function _stampDisc(mx, my) {
            var r = _getRPx();
            var cx = Math.round(mx), cy = Math.round(my);
            var id = _activeClassId;
            // Compute the integer horizontal runs ONCE (no arc/AA), then apply to all targets.
            var runs = [];
            for (var dy = -r; dy <= r; dy++) {
                var yy = cy + dy;
                if (yy < 0 || yy >= _maskH) { continue; }
                var span = Math.floor(Math.sqrt(r*r - dy*dy));   // dx with dx²+dy² <= r²
                var x0 = cx - span, x1 = cx + span;              // inclusive
                if (x0 < 0) { x0 = 0; }
                if (x1 > _maskW - 1) { x1 = _maskW - 1; }
                if (x1 < x0) { continue; }
                runs.push({ x: x0, y: yy, w: (x1 - x0 + 1) });   // one integer run
                // ---- Inc-B: accumulate CLAMPED run into the stroke bbox (undo tracking) ----
                if (_strokeBBox) {
                    if (x0 < _strokeBBox.x0) { _strokeBBox.x0 = x0; }
                    if (x1 > _strokeBBox.x1) { _strokeBBox.x1 = x1; }
                    if (yy < _strokeBBox.y0) { _strokeBBox.y0 = yy; }
                    if (yy > _strokeBBox.y1) { _strokeBBox.y1 = yy; }
                } else {
                    _strokeBBox = { x0: x0, y0: yy, x1: x1, y1: yy };   // [AUDIT 1] EXPLICIT keys — x0/x1 are locals, NOT shorthand
                }
            }
            if (_tool === 'eraser') {
                // ---- Inc-11 (A1), SCOPED BY Inc-14 (§6/A12, user decision 2): the eraser must
                // never clear a pixel inside a committed polygon OF THE ERASE TARGET — i.e. of
                // the ACTIVE class. Protection is no longer class-agnostic. Filter the runs at
                // the ONE place a dab becomes pixels; the mask write and the silhouette
                // loop below share this `runs` local, so both are filtered by construction.
                // _strokeBBox (accumulated in the loop above) stays UNFILTERED — a superset
                // bbox restores identical pixels through putImageData, just larger buffers.
                runs = _protectFilter(runs);
                // Inc-14: erase the ACTIVE class only (user decision 2) — other classes' planes
                // are independent ground truth and an eraser dab must not touch them.
                if (_silh[_activeClassId] && _silh[_activeClassId].ctx) {
                    _shadow(_activeClassId);
                    _fillRuns(_silh[_activeClassId].ctx, runs, 'destination-out', '#fff');
                }
            } else {
                // (b) add these pixels to the active class silhouette (null-guarded). Inc-14: no
                // indexed _mask write and no exclusivity punch — classes may overlap freely.
                _shadow(id);                                    // COW pre-image before this plane's first write
                var se = _ensureSilh(id);
                if (se) { _fillRuns(se.ctx, runs, 'source-over', '#fff'); }
            }
            _mctx.globalCompositeOperation = 'source-over';     // restore
        }

        // ---- E6: interpolation, spacing = radius ----
        function _stamp(from, to) {
            if (!from) { _stampDisc(to.x, to.y); return; }
            var step = _getRPx();                       // spacing = radius
            var ddx = to.x - from.x, ddy = to.y - from.y;
            var dist = Math.sqrt(ddx*ddx + ddy*ddy);
            var n = Math.floor(dist / step);
            for (var i = 1; i <= n; i++) {
                var t = i / n;                            // n>=1 here
                _stampDisc(from.x + ddx*t, from.y + ddy*t);
            }
            _stampDisc(to.x, to.y);                     // always stamp the endpoint
        }

        // ---- Inc-7: nonzero-winding integer scanline rasterizer (pure — no ctx, no side effects) ----
        // Mask pixel (i,j) covers [i,i+1) × [j,j+1); its centre is (i+0.5, j+0.5). Pixel (i,j) is
        // filled IFF its centre lies inside the polygon under the NONZERO winding rule (matching
        // QuPath's PolygonROI WIND_NON_ZERO) — no coverage, no AA, no vertex rounding.
        // ctx.fill() on _mctx is FORBIDDEN: Canvas antialiases path fills regardless of
        // imageSmoothingEnabled, and a fractional R in an indexed label mask is an invalid class id.
        function _polyRuns(pts, w, h) {
            var out = [];
            if (!pts) { return out; }
            var W = _maskW, H = _maskH;
            if (isFinite(w) && w > 0 && isFinite(h) && h > 0) { W = w; H = h; }   // Inc-15: explicit dims (pre-init tests)
            var n = pts.length;
            if (n < 3) { return out; }                       // < 3 vertices encloses nothing
            var minY = Infinity, maxY = -Infinity;
            var i, p;
            for (i = 0; i < n; i++) {
                p = pts[i];
                if (!p || !isFinite(p.x) || !isFinite(p.y)) { return []; }   // any non-finite coord → nothing
                if (p.y < minY) { minY = p.y; }
                if (p.y > maxY) { maxY = p.y; }
            }
            var j0 = Math.max(0, Math.floor(minY));
            var j1 = Math.min(H - 1, Math.ceil(maxY));   // rows outside the mask are never visited
            for (var j = j0; j <= j1; j++) {
                var yc = j + 0.5;                            // scanline at the pixel CENTRE
                var cr = [];
                for (var k = 0; k < n; k++) {
                    var a = pts[k], b = pts[(k + 1) % n];     // edges close the ring implicitly
                    if (a.y === b.y) { continue; }            // horizontal (and zero-length) edges never cross
                    var lo = (a.y < b.y) ? a.y : b.y;
                    var hi = (a.y < b.y) ? b.y : a.y;
                    if (yc < lo || yc >= hi) { continue; }    // HALF-OPEN: upper endpoint inclusive, lower exclusive
                    cr.push({
                        x: a.x + (yc - a.y) * (b.x - a.x) / (b.y - a.y),
                        dir: (b.y > a.y) ? 1 : -1             // +1 = edge runs downward in traversal order
                    });
                }
                if (cr.length < 2) { continue; }
                cr.sort(function (u, v) { return u.x - v.x; });   // ties are empty intervals — order irrelevant
                var wind = 0;
                for (var c = 0; c < cr.length - 1; c++) {
                    wind += cr[c].dir;
                    if (wind === 0) { continue; }             // NONZERO rule (even-odd is NOT the rule)
                    // columns whose centre lies in the half-open span [cr[c].x, cr[c+1].x)
                    var i0 = Math.max(0, Math.ceil(cr[c].x - 0.5));
                    var i1 = Math.min(W - 1, Math.ceil(cr[c + 1].x - 0.5) - 1);
                    if (i1 < i0) { continue; }
                    out.push({ x: i0, y: j, w: (i1 - i0 + 1) });   // one integer run, clamped to the mask
                }
            }
            return out;                                       // row-major
        }

        // ---- Inc-8: deep copy of a vertex list (records NEVER share point objects) ----
        function _copyPts(pts) {
            var out = [];
            if (!pts) { return out; }
            for (var i = 0; i < pts.length; i++) { out.push({ x: pts[i].x, y: pts[i].y }); }
            return out;
        }

        // ---- Inc-10 (§5.9): pure vertex-list equality. Used ONLY by the _reshapeCommit zero-diff
        // branch to tell a real (pixel-identical) vertex move from a no-move handle grab-release,
        // which reaches that branch on every click and must NOT touch history.
        function _ptsEqual(a, b) {
            if (!a || !b || a.length !== b.length) { return false; }
            for (var i = 0; i < a.length; i++) {
                if (a[i].x !== b[i].x || a[i].y !== b[i].y) { return false; }
            }
            return true;
        }

        // ---- Inc-8: point-in-polygon (pure) — the SAME nonzero left-to-right crossing walk as
        // _polyRuns, evaluated at ONE point. Normative equivalence (oracle O8-03):
        //   _ptInPoly(pts, i + 0.5, j + 0.5) === (pixel (i,j) is covered by _polyRuns(pts))
        function _ptInPoly(pts, x, y) {
            if (!pts) { return false; }
            var n = pts.length;
            if (n < 3) { return false; }                         // < 3 vertices encloses nothing
            if (!isFinite(x) || !isFinite(y)) { return false; }
            var wind = 0;
            for (var k = 0; k < n; k++) {
                var a = pts[k], b = pts[(k + 1) % n];             // edges close the ring implicitly
                if (!a || !b || !isFinite(a.x) || !isFinite(a.y) || !isFinite(b.x) || !isFinite(b.y)) { return false; }
                if (a.y === b.y) { continue; }                    // horizontal (and zero-length) edges never cross
                var lo = (a.y < b.y) ? a.y : b.y;
                var hi = (a.y < b.y) ? b.y : a.y;
                if (y < lo || y >= hi) { continue; }              // HALF-OPEN, exactly as _polyRuns
                var xc = a.x + (y - a.y) * (b.x - a.x) / (b.y - a.y);
                if (xc <= x) { wind += (b.y > a.y) ? 1 : -1; }    // crossings at or left of the sample
            }
            return wind !== 0;                                    // NONZERO rule (even-odd is NOT the rule)
        }

        // ---- Inc-11 (A4, PURE): point-to-segment projection. t clamped to [0,1]; null for
        // a degenerate (L2 <= 0) edge — its distance equals the distance to vertex a, which
        // the grab branch owns. isFinite inputs are the caller's concern (record vertices
        // are finite by loadPolygons/_ptFromEvent construction).
        function _projectPointToSeg(p, a, b) {
            var vx = b.x - a.x, vy = b.y - a.y;
            var L2 = vx * vx + vy * vy;
            if (L2 <= 0) { return null; }
            var t = ((p.x - a.x) * vx + (p.y - a.y) * vy) / L2;
            if (t < 0) { t = 0; } else if (t > 1) { t = 1; }
            var qx = a.x + t * vx, qy = a.y + t * vy;
            return { t: t, x: qx, y: qy, d: Math.hypot(p.x - qx, p.y - qy) };
        }

        // ---- Inc-11 (A4, PURE): nearest qualifying ring edge. Edge i runs pts[i] ->
        // pts[(i+1) % n] (the closing segment is i === n-1). Strict `<` keeps the LOWEST
        // edge index on a tie. Returns {idx, x, y, d} or null when no edge is within tol.
        function _nearestEdge(pts, p, tol) {
            var best = null;
            for (var i = 0; i < pts.length; i++) {
                var pr = _projectPointToSeg(p, pts[i], pts[(i + 1) % pts.length]);
                if (!pr) { continue; }
                if (pr.d <= tol && (!best || pr.d < best.d)) {
                    best = { idx: i, x: pr.x, y: pr.y, d: pr.d };
                }
            }
            return best;
        }

        // ---- Inc-8: symmetric difference of two run lists, BY PIXEL SET (pure) ----
        // fill = N \ O, erase = O \ N; O ∩ N appears in neither. Correct even when the input runs
        // within a row overlap or are unordered — the semantics are pixel sets, not run shapes.
        // Output runs are integer, inside the mask, row-major (ascending y, then ascending x).
        function _runsDiff(oldRuns, newRuns) {
            var fill = [], erase = [];
            var W = _maskW;
            if (!(W > 0)) { return { fill: fill, erase: erase }; }
            var rowMap = {};                                      // y -> {o:[runs], n:[runs]}
            var ys = [];
            function _bucket(runs, which) {
                if (!runs) { return; }
                for (var q = 0; q < runs.length; q++) {
                    var rr = runs[q];
                    if (!rr || !(rr.w > 0)) { continue; }
                    if (rr.y < 0 || rr.y >= _maskH) { continue; }  // outside the mask is not a pixel
                    var bk = rowMap[rr.y];
                    if (!bk) { bk = rowMap[rr.y] = { o: [], n: [] }; ys.push(rr.y); }
                    bk[which].push(rr);
                }
            }
            _bucket(oldRuns, 'o');
            _bucket(newRuns, 'n');
            ys.sort(function (u, v) { return u - v; });           // rows visited in ascending y
            var scan = new Uint8Array(W);                         // ONE reusable row accumulator
            for (var i = 0; i < ys.length; i++) {
                var y = ys[i];
                var b = rowMap[y];
                var j, r, c0, c1, c;
                scan.fill(0);
                for (j = 0; j < b.o.length; j++) {                 // OLD pixels -> bit 1
                    r = b.o[j];
                    c0 = r.x < 0 ? 0 : r.x; c1 = r.x + r.w; if (c1 > W) { c1 = W; }
                    for (c = c0; c < c1; c++) { scan[c] |= 1; }
                }
                for (j = 0; j < b.n.length; j++) {                 // NEW pixels -> bit 2
                    r = b.n[j];
                    c0 = r.x < 0 ? 0 : r.x; c1 = r.x + r.w; if (c1 > W) { c1 = W; }
                    for (c = c0; c < c1; c++) { scan[c] |= 2; }
                }
                var cf = -1, ce = -1;                              // open fill / erase run starts
                for (c = 0; c < W; c++) {
                    var v = scan[c];
                    if (v === 2) { if (cf < 0) { cf = c; } }
                    else if (cf >= 0) { fill.push({ x: cf, y: y, w: c - cf }); cf = -1; }
                    if (v === 1) { if (ce < 0) { ce = c; } }
                    else if (ce >= 0) { erase.push({ x: ce, y: y, w: c - ce }); ce = -1; }
                }
                if (cf >= 0) { fill.push({ x: cf, y: y, w: W - cf }); }
                if (ce >= 0) { erase.push({ x: ce, y: y, w: W - ce }); }
            }
            return { fill: fill, erase: erase };
        }

        // ---- Inc-8: inclusive bbox over a run list, or null for [] (same arithmetic as
        // _polyCommit's, factored so it can be applied to the erase list too) ----
        function _runsBBox(runs) {
            if (!runs || runs.length === 0) { return null; }
            var x0 = runs[0].x, x1 = runs[0].x + runs[0].w - 1;
            var y0 = runs[0].y, y1 = runs[0].y;
            for (var i = 1; i < runs.length; i++) {
                var r = runs[i];
                var rx1 = r.x + r.w - 1;
                if (r.x < x0) { x0 = r.x; }
                if (rx1 > x1) { x1 = rx1; }
                if (r.y < y0) { y0 = r.y; }
                if (r.y > y1) { y1 = r.y; }
            }
            return { x0: x0, y0: y0, x1: x1, y1: y1 };            // EXPLICIT keys
        }

        // =====================================================================================
        // ---- Inc-15 ---- raster → vector TRACING (Inc-15 build spec, 2026-08-27)
        // Everything in this block is NEW and NON-FROZEN. Order (spec §4.1): _trace + its private
        // _trace* helpers (§4), _recRuns (§6.2), _copyRec / _applyPolySet (§7.2), _validRing (§8.2),
        // traceActiveClass (§5).
        // =====================================================================================

        function _traceNow() {
            return (typeof performance !== 'undefined' && performance && typeof performance.now === 'function')
                ? performance.now() : Date.now();
        }

        // Signed shoelace area, ½·Σ(x_k·y_{k+1} − x_{k+1}·y_k). In y-down mask coordinates a ring
        // walked with the filled side on its RIGHT is > 0 for an OUTER ring and < 0 for a HOLE (§3).
        function _traceShoelace(pts) {
            var s = 0, n = pts.length;
            for (var i = 0; i < n; i++) {
                var a = pts[i], b = pts[(i + 1) % n];
                s += a.x * b.y - b.x * a.y;
            }
            return s / 2;
        }

        // ---- Inc-15 (§4.3): crack-lattice ring walk on the PADDED label plane `lab` (stride W).
        // Starts at padded corner (x0,y0) = top-left corner of an L pixel whose top edge is a
        // boundary edge, heading E, and keeps label L on the RIGHT: at every corner, with d the
        // incoming direction, `lab(pR) ≠ L` → turn right; else `lab(pL) = L` → turn left; else
        // straight (saddles turn RIGHT — SEPARATING, matching 4-connectivity). A vertex is emitted
        // at every direction change (collinear corners never); the start corner is always a turn
        // and is emitted first. Every E-heading edge marks visitedTop of the pixel below it. The
        // ring ends when the walker is back at c0 heading E. Coordinates are de-padded (−1) to
        // MASK px. Directions: 0 = E, 1 = S, 2 = W, 3 = N; right of d = (d+1)&3, left = (d+3)&3.
        function _traceWalk(lab, visitedTop, W, x0, y0, L) {
            var pts = [{ x: x0 - 1, y: y0 - 1 }];
            var x = x0, y = y0, d = 0;
            visitedTop[y * W + x] = 1; x++;                     // first edge: E along the start pixel's top
            for (;;) {
                var pR, pL;                                       // the two pixels AHEAD of corner (x,y), §4.3 table
                if (d === 0)      { pR = y * W + x;             pL = (y - 1) * W + x; }
                else if (d === 1) { pR = y * W + x - 1;         pL = y * W + x; }
                else if (d === 2) { pR = (y - 1) * W + x - 1;   pL = y * W + x - 1; }
                else              { pR = (y - 1) * W + x;       pL = (y - 1) * W + x - 1; }
                var nd;
                if (lab[pR] !== L)      { nd = (d + 1) & 3; }     // turn right
                else if (lab[pL] === L) { nd = (d + 3) & 3; }     // turn left
                else                    { nd = d; }               // straight
                if (x === x0 && y === y0 && nd === 0) { break; }  // back at c0 heading E: the ring is closed
                if (nd !== d) { pts.push({ x: x - 1, y: y - 1 }); d = nd; }
                if (d === 0)      { visitedTop[y * W + x] = 1; x++; }
                else if (d === 1) { y++; }
                else if (d === 2) { x--; }
                else              { y--; }
            }
            return pts;
        }

        // ---- Inc-15 (§4.5.2): Visvalingam–Whyatt on ONE ring. Circular doubly-linked list over the
        // ring's vertex indices; binary min-heap keyed on ALTITUDE alt(v) = 2·area(prev,v,next) /
        // |prev − next| (0 when |prev − next| = 0), ties broken by the LOWER original index; lazy
        // invalidation by per-vertex version stamp. Phase τ: while count > 3 and min altitude ≤ tau
        // (INCLUSIVE — §0a L2, QuPath's own rule), remove. Phase cap: while count > cap (and > 3),
        // remove the minimum. Guard: never below 3. `stats.capped` counts rings still > cap when
        // phase τ ended. Vertices are REMOVED, never moved — the output is a subset of the input in
        // ring order, freshly allocated. Deliberately keyed on altitude (A7) with guard 3 (D3), NOT
        // QuPath's area queue / max(n/100, 3) — approved divergences, not to be "corrected".
        function _traceSimplify(ring, tau, cap, stats) {
            var n = ring.length, i;
            var out = [];
            if (n <= 3) {
                if (n > cap) { stats.capped++; }
                for (i = 0; i < n; i++) { out.push({ x: ring[i].x, y: ring[i].y }); }
                return out;
            }
            var prev = new Int32Array(n), next = new Int32Array(n), ver = new Int32Array(n);
            var alive = new Uint8Array(n);
            var heap = [];
            function altOf(v) {
                var a = ring[prev[v]], b = ring[v], c = ring[next[v]];
                var dx = c.x - a.x, dy = c.y - a.y;
                var L2 = dx * dx + dy * dy;
                if (L2 === 0) { return 0; }
                return Math.abs((b.x - a.x) * dy - (b.y - a.y) * dx) / Math.sqrt(L2);
            }
            function less(e, f) { return e.a < f.a || (e.a === f.a && e.i < f.i); }
            function push(e) {
                heap.push(e);
                var k = heap.length - 1;
                while (k > 0) {
                    var pk = (k - 1) >> 1;
                    if (!less(heap[k], heap[pk])) { break; }
                    var t = heap[k]; heap[k] = heap[pk]; heap[pk] = t; k = pk;
                }
            }
            function pop() {
                var top = heap[0];
                var last = heap.pop();
                if (heap.length) {
                    heap[0] = last;
                    var k = 0, len = heap.length;
                    for (;;) {
                        var l = 2 * k + 1, r = l + 1, m = k;
                        if (l < len && less(heap[l], heap[m])) { m = l; }
                        if (r < len && less(heap[r], heap[m])) { m = r; }
                        if (m === k) { break; }
                        var t = heap[k]; heap[k] = heap[m]; heap[m] = t; k = m;
                    }
                }
                return top;
            }
            function peek() {                                     // discard stale entries, return the live minimum
                while (heap.length) {
                    var e = heap[0];
                    if (alive[e.i] && e.v === ver[e.i]) { return e; }
                    pop();
                }
                return null;
            }
            for (i = 0; i < n; i++) { prev[i] = (i + n - 1) % n; next[i] = (i + 1) % n; alive[i] = 1; }
            for (i = 0; i < n; i++) { push({ a: altOf(i), i: i, v: 0 }); }
            var count = n;
            function removeMin() {                                // caller guarantees peek() !== null
                var e = pop();
                alive[e.i] = 0;
                var pi = prev[e.i], ni = next[e.i];
                next[pi] = ni; prev[ni] = pi;
                count--;
                ver[pi]++; push({ a: altOf(pi), i: pi, v: ver[pi] });
                ver[ni]++; push({ a: altOf(ni), i: ni, v: ver[ni] });
            }
            while (count > 3) {                                   // phase τ (inclusive)
                var e1 = peek();
                if (!e1 || !(e1.a <= tau)) { break; }
                removeMin();
            }
            if (count > cap) { stats.capped++; }
            while (count > cap && count > 3) {                    // phase cap, guard 3
                if (!peek()) { break; }
                removeMin();
            }
            var s = 0;
            while (s < n && !alive[s]) { s++; }
            var cur = s;
            do { out.push({ x: ring[cur].x, y: ring[cur].y }); cur = next[cur]; } while (cur !== s);
            return out;
        }

        // ---- Inc-15 (§4): THE PURE TRACER. `occ` is a Uint8Array(w*h), row-major, index y*w+x,
        // NONZERO = filled; w, h ≥ 1; opts optional {tau, cap, minArea} (each defaults to its
        // constant). Returns [{pts:[{x,y}], holes:[[{x,y}],…], area}] — `holes` ALWAYS an array —
        // ordered (nesting rank ASC, area DESC, label ASC). Reads NO module state, never mutates
        // occ, allocates fresh output per call; its ONE side effect is overwriting _lastTraceStats
        // (§4.7). Callable pre-init. Steps: (1) 4-connected component labelling on a 1-px-padded
        // Int32Array; (2) crack-lattice ring walk (outer shoelace > 0, hole < 0 — the label IS the
        // parent link); (3) nesting rank by 8-connected CAVITY labelling of the background in the
        // SAME array with negative ids; (4) min-area, VW per ring, the ≥ 1-px invariant; (5) order.
        function _trace(occ, w, h, opts) {
            var t0 = _traceNow();
            var tau = (opts && opts.tau !== undefined) ? opts.tau : TRACE_TAU;
            var cap = (opts && opts.cap !== undefined) ? opts.cap : TRACE_CAP;
            var minArea = (opts && opts.minArea !== undefined) ? opts.minArea : TRACE_MIN_AREA;
            var stats = { regions: 0, holes: 0, dropped: 0, holesFilled: 0, capped: 0, ms: 0 };
            var out = [];
            if (!occ || !(w >= 1) || !(h >= 1)) {
                stats.ms = _traceNow() - t0; _lastTraceStats = stats;
                return out;
            }
            var W = w + 2, H = h + 2;                             // padded plane; the 1-px border is background
            var lab = new Int32Array(W * H);                      // 0 = unlabelled; L ≥ 1 foreground; −k background (step 3)
            var stack = [];
            var p, px, py, q, qx, qy, prow, i;
            // ---- step 1: 4-connected component labelling. Foreground is read from occ directly ----
            var area = [0], topLeft = [0];                        // indexed by label L; slot 0 unused
            var L = 0;
            for (py = 1; py <= h; py++) {
                prow = py * W;
                var orow = (py - 1) * w - 1;                      // occ index of padded (px,py) = orow + px
                for (px = 1; px <= w; px++) {
                    p = prow + px;
                    if (lab[p] !== 0 || occ[orow + px] === 0) { continue; }
                    L++;
                    lab[p] = L; topLeft.push(p);
                    var cnt = 0;
                    stack.push(p);
                    while (stack.length) {
                        q = stack.pop(); cnt++;
                        qy = (q / W) | 0; qx = q - qy * W;
                        var oq = (qy - 1) * w + (qx - 1);         // occ index of q (always interior)
                        // the pad is never foreground: the coordinate tests keep every read inside occ
                        if (qx > 1 && lab[q - 1] === 0 && occ[oq - 1] !== 0) { lab[q - 1] = L; stack.push(q - 1); }
                        if (qx < w && lab[q + 1] === 0 && occ[oq + 1] !== 0) { lab[q + 1] = L; stack.push(q + 1); }
                        if (qy > 1 && lab[q - W] === 0 && occ[oq - w] !== 0) { lab[q - W] = L; stack.push(q - W); }
                        if (qy < h && lab[q + W] === 0 && occ[oq + w] !== 0) { lab[q + W] = L; stack.push(q + W); }
                    }
                    area.push(cnt);
                }
            }
            // ---- step 2: ring enumeration + walk. Start at every L pixel whose top edge is a boundary
            // edge and has not been traversed; classify by the sign of the raw shoelace area ----
            var outerOf = new Array(L + 1), holesOf = new Array(L + 1);
            var visitedTop = new Uint8Array(W * H);
            for (py = 1; py <= h; py++) {
                prow = py * W;
                for (px = 1; px <= w; px++) {
                    p = prow + px;
                    var lp = lab[p];
                    if (lp <= 0 || lab[p - W] === lp || visitedTop[p] !== 0) { continue; }
                    var ring = _traceWalk(lab, visitedTop, W, px, py, lp);
                    if (_traceShoelace(ring) > 0) { outerOf[lp] = ring; }
                    else { (holesOf[lp] || (holesOf[lp] = [])).push(ring); }
                }
            }
            // ---- step 3: nesting rank by 8-connected CAVITY labelling (negative ids, same array).
            // Seed at padded index 0 so the OUTSIDE is −1; parent[k] = the label directly above
            // cavity k's topmost-leftmost pixel; rank[L] = 0 if the pixel above topLeft[L] is
            // outside, else rank[parent] + 1 — one pass in label order, parents always ranked first ----
            var parent = [0, 0];                                  // indexed by k = −lab; slots 0/1 unused
            var K = 0, total = W * H, Wm1 = W - 1, Hm1 = H - 1;
            for (p = 0; p < total; p++) {
                if (lab[p] !== 0) { continue; }
                K++;
                var neg = -K;
                lab[p] = neg; stack.push(p);
                while (stack.length) {
                    q = stack.pop();
                    qy = (q / W) | 0; qx = q - qy * W;
                    var xl = qx > 0, xr = qx < Wm1, yu = qy > 0, yd = qy < Hm1;
                    if (xl && lab[q - 1] === 0)          { lab[q - 1] = neg;     stack.push(q - 1); }
                    if (xr && lab[q + 1] === 0)          { lab[q + 1] = neg;     stack.push(q + 1); }
                    if (yu && lab[q - W] === 0)          { lab[q - W] = neg;     stack.push(q - W); }
                    if (yd && lab[q + W] === 0)          { lab[q + W] = neg;     stack.push(q + W); }
                    if (yu && xl && lab[q - W - 1] === 0) { lab[q - W - 1] = neg; stack.push(q - W - 1); }
                    if (yu && xr && lab[q - W + 1] === 0) { lab[q - W + 1] = neg; stack.push(q - W + 1); }
                    if (yd && xl && lab[q + W - 1] === 0) { lab[q + W - 1] = neg; stack.push(q + W - 1); }
                    if (yd && xr && lab[q + W + 1] === 0) { lab[q + W + 1] = neg; stack.push(q + W + 1); }
                }
                if (K >= 2) { parent.push(lab[p - W]); }          // the seed is the cavity's top-left; above it is foreground
            }
            var rank = new Int32Array(L + 1);
            for (i = 1; i <= L; i++) {
                var above = lab[topLeft[i] - W];                  // always background (§4.4)
                if (above === -1) { rank[i] = 0; continue; }
                var pl = parent[-above];
                rank[i] = (pl > 0 ? rank[pl] : 0) + 1;
            }
            // ---- step 4: min-area, VW per ring (outer AND holes), the ≥ 1-px invariant ----
            var recs = [];
            for (i = 1; i <= L; i++) {
                if (area[i] < minArea) { stats.dropped++; continue; }
                var outer = outerOf[i];
                if (!outer) { stats.dropped++; continue; }        // unreachable: every component has one outer ring
                var hl = holesOf[i] || [];
                var sholes = [];
                for (var hi = 0; hi < hl.length; hi++) {
                    if (Math.abs(_traceShoelace(hl[hi])) < minArea) { stats.holesFilled++; continue; }   // FILLED
                    sholes.push(_traceSimplify(hl[hi], tau, cap, stats));
                }
                var spts = _traceSimplify(outer, tau, cap, stats);
                if (_recRuns(spts, sholes, w, h).length === 0) { stats.dropped++; continue; }   // A19: never emit an empty record
                recs.push({ pts: spts, holes: sholes, area: area[i], rank: rank[i], label: i });
            }
            // ---- step 5: order (rank ASC, area DESC, label ASC) and output ----
            recs.sort(function (u, v) {
                if (u.rank !== v.rank) { return u.rank - v.rank; }
                if (u.area !== v.area) { return v.area - u.area; }
                return u.label - v.label;
            });
            for (i = 0; i < recs.length; i++) {
                out.push({ pts: recs[i].pts, holes: recs[i].holes, area: recs[i].area });
                stats.holes += recs[i].holes.length;
            }
            stats.regions = out.length;
            stats.ms = _traceNow() - t0;
            _lastTraceStats = stats;
            return out;
        }

        // ---- Inc-15 (§6.2, A5): rasterise a RECORD — runs(outer) − ∪runs(holes). Pure beyond
        // _polyRuns's dims fallback. A hole partly or wholly outside the outer subtracts nothing
        // there — no special case. Every site that rasterises a record goes through here (I-30).
        function _recRuns(pts, holes, w, h) {
            var outer = _polyRuns(pts, w, h);
            if (!holes || holes.length === 0 || outer.length === 0) { return outer; }
            var all = [];
            for (var i = 0; i < holes.length; i++) {
                var hr = _polyRuns(holes[i], w, h);
                for (var q = 0; q < hr.length; q++) { all.push(hr[q]); }
            }
            return _runsSubtract(outer, _mergeRunsToRows(all));
        }

        // ---- Inc-15 (§7.2): deep copy of a record WITH its id. Emits `holes` ONLY when non-empty
        // and `traced` ONLY when exactly true — never `holes: []`, never `traced: false` (§3).
        function _copyRec(rec) {
            var c = { id: rec.id, classId: rec.classId, pts: _copyPts(rec.pts) };
            if (rec.holes && rec.holes.length) {
                var hs = [];
                for (var i = 0; i < rec.holes.length; i++) { hs.push(_copyPts(rec.holes[i])); }
                c.holes = hs;
            }
            if (rec.traced === true) { c.traced = true; }
            return c;
        }

        // ---- Inc-15 (§7.2): REPLACE class `cid`'s record set with deep copies of `recs` (ids kept).
        // Never writes pixels, never schedules a render (undo/redo call _render() themselves, as
        // _applyPolyFixup documents). Restored records go to the END of _polys (top of z-order) in
        // their own relative order; records whose class no longer exists are skipped silently.
        function _applyPolySet(cid, recs) {
            _polys = _polys.filter(function (p) { return p.classId !== cid; });
            for (var i = 0; i < recs.length; i++) {
                var c = _copyRec(recs[i]);
                if (c.id > _polySeq) { _polySeq = c.id; }
                if (_classById(c.classId)) { _polys.push(c); }
            }
            if (_selPoly && _selPoly.classId === cid) { _selPoly = null; _selVert = null; }   // _dragVert is null under undo/redo's guard
        }


        // ---- Inc-16 ----
        // (§5.3) The INCLUSIVE pixel bbox {x0,y0,x1,y1} of a ring. EXACT for a traced record's
        // integer crack vertices; a SUPERSET after a hand reshape of a traced record (float
        // vertices), because _polyRuns fills centre-inside pixels only (E7). Holes lie inside the
        // outer, so `pts` alone suffices. A degenerate result (x1 < x0 or y1 < y0) is SKIPPED by
        // every caller BEFORE dilating — dilation would otherwise widen it into a 1-px strip.
        function _recBBox(pts) {
            var x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
            for (var i = 0; i < pts.length; i++) {
                var p = pts[i];
                if (p.x < x0) { x0 = p.x; } if (p.x > x1) { x1 = p.x; }
                if (p.y < y0) { y0 = p.y; } if (p.y > y1) { y1 = p.y; }
            }
            return { x0: Math.floor(x0), y0: Math.floor(y0), x1: Math.ceil(x1) - 1, y1: Math.ceil(y1) - 1 };
        }

        // ---- Inc-16 (§6, A6/A7/M5): swap class `cid`'s records — drop `drop` BY ID, append deep
        // copies of `add`, and re-lay the class as the stable partition
        // `others ⧺ untouched-traced ⧺ add ⧺ hand-drawn`. By id, never identity (the
        // _applyPolyFixup lesson): undo/redo hold copies. Hand-drawn records of the class end up on
        // TOP so a polygon drawn by hand inside a traced region keeps winning the click walk
        // (:2049-2055, last-index-first, class-gated — interleaving with OTHER classes is not
        // preserved and is unobservable). Never writes pixels, never schedules a render.
        function _applyPolyDiff(cid, drop, add) {
            var dropIds = {}, i, c;
            for (i = 0; i < drop.length; i++) { dropIds[drop[i].id] = true; }
            var rest = [], tr = [], hd = [];
            for (i = 0; i < _polys.length; i++) {
                var p = _polys[i];
                if (p.classId !== cid) { rest.push(p); }
                else if (p.traced === true) { if (!dropIds[p.id]) { tr.push(p); } }
                else { hd.push(p); }
            }
            for (i = 0; i < add.length; i++) {
                c = _copyRec(add[i]);
                if (c.id > _polySeq) { _polySeq = c.id; }
                if (_classById(c.classId)) { tr.push(c); }
            }
            _polys = rest.concat(tr, hd);
            if (_selPoly && _selPoly.classId === cid) { _selPoly = null; _selVert = null; }   // _dragVert is null under undo/redo's guard and during a stroke commit
        }

        // ---- Inc-16 (§5, A3-A5/A8-A12, M1/M2): PIXELS ARE THE TRUTH (FP1). A brush/eraser stroke
        // whose bbox meets a `traced` record of the ACTIVE class re-derives that neighbourhood's
        // traced records from the LIVE plane and returns the swap for the stroke's OWN history
        // entry. It writes no plane, schedules no render, mints no record for a hand-drawn one and
        // never touches a hand-drawn record object (I-35). The predicate is bbox-only and allocates
        // NOTHING before its last check (I-38): a null return leaves _polys, _retraceStats and the
        // entry exactly as they were.
        function _retraceStroke(cid, x, y, w, h) {
            var t0 = _traceNow();                                // discarded on a null return; lets `ms` be end-to-end
            var se = _silh[cid];
            if (!se || !se.ctx) { return null; }                 // §5.2 check 3
            // §5.2 check 4: S = the UNFILTERED stroke bbox (INCLUSIVE mask px; _strokeBBox accumulates
            // every CLAMPED dab run before the eraser filter). The meet uses the 1-px-DILATED record
            // bbox (E1): a brush pixel 4-adjacent to a record pixel merges with it without overlapping
            // its bbox. No memcmp of planes (R2), no _tool test (R7), no rPx term (R8).
            var sx0 = x, sy0 = y, sx1 = x + w - 1, sy1 = y + h - 1;
            var i, r, b, met = false;
            for (i = 0; i < _polys.length; i++) {
                r = _polys[i];
                if (r.classId !== cid || r.traced !== true) { continue; }
                b = _recBBox(r.pts);
                if (!(b.x1 >= b.x0 && b.y1 >= b.y0)) { continue; }          // §5.3 degenerate guard
                if (b.x0 - 1 <= sx1 && sx0 <= b.x1 + 1 && b.y0 - 1 <= sy1 && sy0 <= b.y1 + 1) { met = true; break; }
            }
            if (!met) { return null; }                           // E11 — before any readback or allocation
            // ---- §5.4: the crop — PULL-AND-GROW. C starts as clamp(dilate(S, 1)) (E8: the dilation
            // at a plane corner yields -1). Each pass pulls every traced record of `cid` whose
            // clamped dilated bbox meets C and unions that bbox in (L1: the 1-px margin keeps a
            // pulled record's own pixels off C's border), re-scanning until nothing is added so the
            // set is closed under ancestors AND descendants (E2). Hand-drawn runs are subtracted
            // from the occupancy BEFORE the border scan and clipped to C (E3), so hand-drawn pixels
            // on the border never force growth. A dirty side that is not a plane edge (E4) grows by
            // `step` (64, doubling) and the pull re-runs (E5). Terminates: |Tset| and C both grow
            // monotonically and are bounded by the record count and the plane.
            var C = { x0: sx0 - 1, y0: sy0 - 1, x1: sx1 + 1, y1: sy1 + 1 };
            if (C.x0 < 0) { C.x0 = 0; }
            if (C.y0 < 0) { C.y0 = 0; }
            if (C.x1 > _maskW - 1) { C.x1 = _maskW - 1; }
            if (C.y1 > _maskH - 1) { C.y1 = _maskH - 1; }
            var Tset = {}, step = 64, grown = 0, Cw = 0, Ch = 0, occ = null;
            var pulled, d, p, q, k, runs, run, xs, xe, rowOff, yy, xx;
            var bx0, by0, bx1, by1, dirtyL, dirtyR, dirtyT, dirtyB;
            for (;;) {
                // (1) PULL
                pulled = false;
                for (i = 0; i < _polys.length; i++) {
                    r = _polys[i];
                    if (r.classId !== cid || r.traced !== true || Tset[r.id]) { continue; }
                    b = _recBBox(r.pts);
                    if (!(b.x1 >= b.x0 && b.y1 >= b.y0)) { continue; }      // §5.3 degenerate guard
                    bx0 = b.x0 - 1; by0 = b.y0 - 1; bx1 = b.x1 + 1; by1 = b.y1 + 1;
                    if (bx0 < 0) { bx0 = 0; }
                    if (by0 < 0) { by0 = 0; }
                    if (bx1 > _maskW - 1) { bx1 = _maskW - 1; }
                    if (by1 > _maskH - 1) { by1 = _maskH - 1; }
                    if (!(bx0 <= C.x1 && C.x0 <= bx1 && by0 <= C.y1 && C.y0 <= by1)) { continue; }
                    Tset[r.id] = true; pulled = true;
                    if (bx0 < C.x0) { C.x0 = bx0; }
                    if (by0 < C.y0) { C.y0 = by0; }
                    if (bx1 > C.x1) { C.x1 = bx1; }
                    if (by1 > C.y1) { C.y1 = by1; }
                }
                if (pulled) { continue; }                        // re-scan until no record is added
                // (2) READBACK — one per iteration that reaches here; the LIVE plane (E12): the
                // stroke's own pixels are already in it, and planes[]/_strokeSilhIds are not consulted.
                Cw = C.x1 - C.x0 + 1; Ch = C.y1 - C.y0 + 1;
                d = se.ctx.getImageData(C.x0, C.y0, Cw, Ch).data;
                occ = new Uint8Array(Cw * Ch);
                for (p = 0, q = 3; p < occ.length; p++, q += 4) { occ[p] = (d[q] > 0) ? 1 : 0; }
                d = null;
                // (3) HAND-DRAWN SUBTRACTION — their pixels are invisible to the tracer; the record
                // OBJECTS keep their identity and their ids (I-35). _recRuns runs are half-open
                // (x .. x+w-1 inclusive), so the `+ 1` on fill's end index is the conversion.
                for (i = 0; i < _polys.length; i++) {
                    r = _polys[i];
                    if (r.classId !== cid || r.traced === true) { continue; }
                    b = _recBBox(r.pts);
                    if (!(b.x0 <= C.x1 && C.x0 <= b.x1 && b.y0 <= C.y1 && C.y0 <= b.y1)) { continue; }
                    runs = _recRuns(r.pts, r.holes);             // module dims; half-open runs {x, y, w}
                    for (k = 0; k < runs.length; k++) {
                        run = runs[k];
                        if (run.y < C.y0 || run.y > C.y1) { continue; }
                        xs = (run.x < C.x0) ? C.x0 : run.x;
                        xe = run.x + run.w - 1; if (xe > C.x1) { xe = C.x1; }
                        if (xs > xe) { continue; }
                        rowOff = (run.y - C.y0) * Cw;
                        occ.fill(0, rowOff + (xs - C.x0), rowOff + (xe - C.x0) + 1);
                    }
                }
                // (4) BORDER SCAN — a side that IS a plane edge is never dirty (E4).
                dirtyL = false; dirtyR = false; dirtyT = false; dirtyB = false;
                if (C.x0 > 0) { for (yy = 0; yy < Ch; yy++) { if (occ[yy * Cw]) { dirtyL = true; break; } } }
                if (C.x1 < _maskW - 1) { for (yy = 0; yy < Ch; yy++) { if (occ[yy * Cw + Cw - 1]) { dirtyR = true; break; } } }
                if (C.y0 > 0) { for (xx = 0; xx < Cw; xx++) { if (occ[xx]) { dirtyT = true; break; } } }
                if (C.y1 < _maskH - 1) { rowOff = (Ch - 1) * Cw; for (xx = 0; xx < Cw; xx++) { if (occ[rowOff + xx]) { dirtyB = true; break; } } }
                if (dirtyL || dirtyR || dirtyT || dirtyB) {
                    if (dirtyL) { C.x0 = Math.max(0, C.x0 - step); }
                    if (dirtyR) { C.x1 = Math.min(_maskW - 1, C.x1 + step); }
                    if (dirtyT) { C.y0 = Math.max(0, C.y0 - step); }
                    if (dirtyB) { C.y1 = Math.min(_maskH - 1, C.y1 + step); }
                    step *= 2; grown++; occ = null;
                    continue;                                    // the pull re-runs on the grown C
                }
                break;
            }
            // ---- §5.5: trace the crop and mint the replacements. _trace is UNCHANGED and called
            // with its DEFAULTS (τ 2.0, cap 24, min-area 16); it overwrites _lastTraceStats, which
            // stays the TRACER's own record. The crop offset is added to `pts` AND to every hole
            // ring (E6) in place — _trace's arrays are freshly allocated. res.length === 0 is an
            // ANSWER, not a refusal (E10): the pulled records are removed and nothing is added. A
            // >= 16 px² orphan component inside C is minted as a traced record (E9; ticket §1).
            var res = _trace(occ, Cw, Ch);
            occ = null;
            var N = [], ring, nr;
            for (i = 0; i < res.length; i++) {
                for (k = 0; k < res[i].pts.length; k++) { res[i].pts[k].x += C.x0; res[i].pts[k].y += C.y0; }
                for (q = 0; q < res[i].holes.length; q++) {
                    ring = res[i].holes[q];
                    for (k = 0; k < ring.length; k++) { ring[k].x += C.x0; ring[k].y += C.y0; }
                }
                nr = { id: ++_polySeq, classId: cid, pts: res[i].pts, traced: true };   // fresh id (R5), `holes` only when non-empty
                if (res[i].holes.length) { nr.holes = res[i].holes; }
                N.push(nr);
            }
            // ---- §5.6: apply and return. `removed` is the pulled set in _polys ORDER (L5) — never
            // in pull order, or undo would re-add an island above its parent. Both lists are
            // _copyRec deep copies WITH ids; the live _polys holds its own copies.
            var removed = [], added = [];
            for (i = 0; i < _polys.length; i++) { if (Tset[_polys[i].id]) { removed.push(_copyRec(_polys[i])); } }
            _applyPolyDiff(cid, removed, N);
            for (i = 0; i < N.length; i++) { added.push(_copyRec(N[i])); }
            _retraceStats = { runs: _retraceStats.runs + 1, ms: _traceNow() - t0, crop: { x: C.x0, y: C.y0, w: Cw, h: Ch }, grown: grown };
            return { classId: cid, removed: removed, added: added };
        }

        // ---- Inc-15 (§8.2): FILE-form ring validator — true iff an Array of ≥ 3 [x, y] pairs of
        // finite numbers (loadPolygons' exact pre-Inc-15 point test, factored; used for pts AND holes).
        function _validRing(a) {
            if (!Array.isArray(a) || a.length < 3) { return false; }
            for (var k = 0; k < a.length; k++) {
                var ent = a[k];
                if (!Array.isArray(ent) || ent.length !== 2
                    || typeof ent[0] !== 'number' || !isFinite(ent[0])
                    || typeof ent[1] !== 'number' || !isFinite(ent[1])) { return false; }
            }
            return true;
        }

        // ---- Inc-15 (§5) / Inc-17 (§5): the ONE trace body, shared by the public traceActiveClass()
        // (pushHist === true: ONE history entry, exactly as Inc-15) and by traceLoadedClasses()
        // (pushHist === false: NO shadow, NO before/after copies, NO _histCommit — the caller flushes).
        // Trace the class's silhouette plane into polygon records (outer ring + hole rings),
        // re-rasterise the plane from the simplified records (D1: mask == vectors by construction),
        // REPLACE the class's records. Returns `null` (nothing to trace) or {regions, holes, capped}.
        // NOTHING is written before the last refusal passes (I-28). Refusals 1 and 2 (no image /
        // busy) belong to traceActiveClass — this body assumes the caller has decided them.
        function _traceClass(cid, pushHist) {
            var t0 = _traceNow();
            var se = _silh[cid];
            if (!se || !se.ctx) { return null; }                 // refusal 3a — plane unallocated
            // readback: the module's SINGLE full-plane read (J5); occ + inclusive pixel bbox `ob`
            var d = se.ctx.getImageData(0, 0, _maskW, _maskH).data;
            var occ = new Uint8Array(_maskW * _maskH);
            var ob = null;
            var x, y, p = 0, di = 3;
            for (y = 0; y < _maskH; y++) {
                for (x = 0; x < _maskW; x++, p++, di += 4) {
                    if (d[di] > 0) {
                        occ[p] = 1;
                        if (ob === null) { ob = { x0: x, y0: y, x1: x, y1: y }; }   // EXPLICIT keys
                        else {
                            if (x < ob.x0) { ob.x0 = x; }
                            if (x > ob.x1) { ob.x1 = x; }
                            if (y > ob.y1) { ob.y1 = y; }               // rows are visited ascending: y0 is the first hit
                        }
                    }
                }
            }
            d = null;
            if (ob === null) { return null; }                    // refusal 3b — no alpha > 0 pixel
            var res = _trace(occ, _maskW, _maskH);              // defaults (τ 2.0, cap 24, min-area 16)
            occ = null;
            if (res.length === 0) { return null; }               // refusal 4 — every component dropped (_lastTraceStats updated)
            // records: fresh ids, `holes` only when non-empty, `traced` always
            var recs = [], i, holesTotal = 0;
            for (i = 0; i < res.length; i++) {
                var nr = { id: ++_polySeq, classId: cid, pts: res[i].pts, traced: true };
                if (res[i].holes.length) { nr.holes = res[i].holes; }
                holesTotal += res[i].holes.length;
                recs.push(nr);
            }
            var before = [];
            if (pushHist) { for (i = 0; i < _polys.length; i++) { if (_polys[i].classId === cid) { before.push(_copyRec(_polys[i])); } } }
            // commit — clear-and-refill through the existing _polyCommit (A3, "Commit shape: A")
            _histBegin();                                        // Inc-15 order: release the previous stroke's shadows FIRST (kept unconditional — Inc-17 L4)
            if (pushHist) { _shadow(cid); }                      // pre-image BEFORE the clear — only when an entry will be pushed
            se.ctx.globalCompositeOperation = 'source-over';
            se.ctx.clearRect(0, 0, _maskW, _maskH);              // plane := ∅
            for (i = 0; i < recs.length; i++) { _polyCommit(_recRuns(recs[i].pts, recs[i].holes), cid); }   // sets _maskDirty
            _strokeBBox = { x0: ob.x0, y0: ob.y0, x1: ob.x1, y1: ob.y1 };   // OVERRIDE _polyCommit's last-record bbox: the union
            _polys = _polys.filter(function (q) { return q.classId !== cid; }).concat(recs);   // REPLACED (J2); precedent removeClass
            if (_selPoly && _selPoly.classId === cid) { _selPoly = null; _selVert = null; }   // _dragVert is null by refusal 2 (button) or irrelevant (load)
            if (pushHist) {
                var after = [];
                for (i = 0; i < recs.length; i++) { after.push(_copyRec(recs[i])); }
                _histCommit({ polys: { classId: cid, before: before, after: after } });
            }
            _lastTraceStats.ms = _traceNow() - t0;               // END-TO-END (readback through the commit)
            _scheduleRender();
            return { regions: recs.length, holes: holesTotal, capped: _lastTraceStats.capped };
        }

        // ---- Inc-15 (§5): PUBLIC — trace the ACTIVE class; ONE history entry (D4). Returns `false`
        // (busy / no image), `null` (nothing to trace) or {regions, holes, capped}. Inc-17 (D2): the
        // hosts no longer call this (the Trace button is gone); kept as module API for tests
        // and scripts. The body is _traceClass.
        function traceActiveClass() {
            // refusal 1 — no image / destroyed / frozen (mirrors loadPolygons' guard)
            if (_destroyed || !_mask || _frozen) {
                if (!_mask && !_destroyed) { _warnOnce('no-image', 'OSDAnnotator: no image loaded yet'); }
                return false;
            }
            // refusal 2 — busy. `_tool === 'eraser'` is the :1288 trap: _histCommit would drop the
            // entry of a trace made during an Alt latch or after setTool('eraser').
            if (_painting || _polyDraft || _dragVert || _tool === 'eraser') { return false; }
            return _traceClass(_activeClassId, true);
        }

        // ---- Inc-17 (D1/D2, spec §6): PUBLIC — trace EVERY class that has a plane, for the host's
        // json-less Load Mask lane ("without json everything is a polygon"). A load is not a gesture,
        // so there is NO _tool / _polyDraft / _painting / _dragVert refusal (the eraser latch and an
        // open draft are left exactly as they were). Pushes NO history — N full-plane entries would be
        // ~1.57 GB at maxClasses — and FLUSHES both stacks, like loadLabel / loadMasks / clear.
        // Returns null (no image / destroyed / frozen) or the AGGREGATE {classes, regions, holes,
        // capped}; `classes` counts classes that produced >= 1 record; `capped` is SUMMED across
        // classes (_lastTraceStats is per _trace call and describes the LAST class only).
        function traceLoadedClasses() {
            if (_destroyed || !_mask || _frozen) { return null; }
            var tot = { classes: 0, regions: 0, holes: 0, capped: 0 }, i, r;
            for (i = 0; i < _classes.length; i++) {
                _histBegin();                       // PER CLASS (I-41, belt-and-braces): frees the previous class's _polyCommit
                                                    // shadow BEFORE this class's 49 MB readback. The pool bound itself comes from
                                                    // _traceClass's own UNCONDITIONAL _histBegin() — never guard that one by pushHist.
                r = _traceClass(_classes[i].id, false);
                if (r) { tot.classes++; tot.regions += r.regions; tot.holes += r.holes; tot.capped += r.capped; }
            }
            _histBegin();                           // LOAD-BEARING (I-40): nulls _strokeBBox and releases the last shadow, so a
                                                    // pointer-up landing after an async load early-returns in _histCommit (:1881)
                                                    // instead of pushing an entry over the tracer's union bbox. Do not "simplify".
            _undoStack.length = 0; _redoStack.length = 0; _fireHistory();   // a load is never undoable
            _scheduleRender();
            return tot;
        }

        // ---- Inc-7: commit a closed polygon into its class plane (mirrors _stampDisc's write) ----
        // _stampDisc stays byte-identical (frozen); this reuses the same non-frozen _fillRuns helper.
        function _polyCommit(runs, cid) {
            if (!runs || runs.length === 0) { return; }       // makes the _maskDirty gate explicit
            // Inc-8: `cid` is OPTIONAL — every Inc-7 caller passes one argument and behaves as
            // before (class taken at CLOSE time); _reshapeCommit passes the RECORD's classId so a
            // reshape always writes the polygon's own class, never the active one.
            var id = (typeof cid === 'number') ? cid : _activeClassId;
            // (b) add these pixels to the class silhouette (null-guarded). Inc-14: no indexed
            // _mask write and no exclusivity punch — classes may overlap freely.
            _shadow(id);                                     // COW pre-image before this plane's first write
            var se = _ensureSilh(id);
            if (se) { _fillRuns(se.ctx, runs, 'source-over', '#fff'); }
            // ---- Inc-B undo bbox, computed HERE over the CLAMPED runs ----
            // The brush accumulates _strokeBBox inside _stampDisc (frozen), which a polygon fill
            // never calls; raw vertices must not be used (an off-mask vertex would give an
            // out-of-range getImageData rect in _histCommit).
            var x0 = runs[0].x, x1 = runs[0].x + runs[0].w - 1;
            var y0 = runs[0].y, y1 = runs[0].y;
            for (var i = 1; i < runs.length; i++) {
                var r = runs[i];
                var rx1 = r.x + r.w - 1;
                if (r.x < x0) { x0 = r.x; }
                if (rx1 > x1) { x1 = rx1; }
                if (r.y < y0) { y0 = r.y; }
                if (r.y > y1) { y1 = r.y; }
            }
            _strokeBBox = { x0: x0, y0: y0, x1: x1, y1: y1 };   // EXPLICIT keys
            _maskDirty = true;              // set DIRECTLY: _maskDirtyOnBrush is brush-gated AND frozen
            _mctx.globalCompositeOperation = 'source-over';     // belt-and-braces (_fillRuns already restores)
        }

        // ---- E3: pointer FSM helpers ----
        function _ptFromEvent(e) {
            var r = _view.getBoundingClientRect();   // LIVE per event
            return _pointerToMask(e.clientX - r.left, e.clientY - r.top);
        }

        function _endStroke() {
            // ---- Inc-8: a vertex RESHAPE drag commits on ANY press end (the four frozen handlers
            // all funnel here argument-less), exactly as a brush stroke and Inc-7 freehand do. This
            // is the ONLY place the reshape write happens; nulling _dragVert FIRST makes the call
            // re-entrancy-safe and makes "null _dragVert before _endStroke runs" the abort
            // primitive every teardown path already gets for free through _dropDraft.
            if (_dragVert) {
                var dv = _dragVert; _dragVert = null;
                if (_polys.indexOf(dv.poly) >= 0) { _reshapeCommit(dv.poly, dv.pts); }   // skipped if the record vanished
                _scheduleRender();
            }
            // ---- Inc-7: a FREEHAND polygon press commits on ANY press end (up/cancel/lostcapture/
            // leave all funnel here argument-less), exactly as a brush stroke does. A CLICK-mode
            // draft survives untouched — that is what makes click-to-place work.
            if (_polyDraft && _polyDraft.mode === 'freehand') {
                var lp = _polyDraft.pts[_polyDraft.pts.length - 1];
                if (_polyCurMask && (!lp || _polyCurMask.x !== lp.x || _polyCurMask.y !== lp.y)) {
                    _polyDraft.pts.push(_polyCurMask);       // release-tail vertex is always kept
                }
                _closeDraft();
            }
            _polyDownCss = null; _polyLastCss = null; _polyCurMask = null;
            _histCommit();   // Inc-B: commit the stroke's undo entry (no-op when _strokeBBox null)
            if (_activePointerId !== null && _view && _view.hasPointerCapture && _view.hasPointerCapture(_activePointerId)) {
                _view.releasePointerCapture(_activePointerId);
            }
            _painting = false; _activePointerId = null; _lastPt = null;
        }

        function _endPan() {
            if (_panPointerId !== null && _view && _view.hasPointerCapture && _view.hasPointerCapture(_panPointerId)) {
                _view.releasePointerCapture(_panPointerId);
            }
            _panning = false; _panPointerId = null;
        }

        function _maskDirtyOnBrush() {
            if (_tool === 'brush') { _maskDirty = true; }
        }

        // ---- Inc-7: polygon draft lifecycle (non-frozen) ----
        // Drops the open draft WITHOUT writing the mask. Does NOT touch _painting/_activePointerId/
        // pointer capture — a held press still ends through _endStroke as usual.
        function _dropDraft() {
            if (_polyDraft !== null) { _polyDraft = null; _scheduleRender(); _fireHistory(); }   // Inc-11 (A6): un-grey — AFTER nulling
            // ---- Inc-8: this is the LOAD-BEARING abort for a hijacked vertex drag. Every teardown
            // path (setTool on change, setActive(false), clear, cancelPolygon/Esc) already calls
            // _dropDraft BEFORE _endStroke, so nulling _dragVert here means the _endStroke prologue
            // sees nothing to commit — no mask write, no history entry. _polys is NOT touched here:
            // Esc must deselect, never delete committed polygons.
            if (_selPoly !== null || _dragVert !== null) { _selPoly = null; _dragVert = null; _selVert = null; _scheduleRender(); }
            _polyLastClick = null; _polyDownCss = null; _polyLastCss = null; _polyCurMask = null;
        }

        // Closes the open draft: rasterize + commit as ONE undo entry. Returns true iff pixels
        // were written. Degeneracy is simply "zero runs" — there is no min-area epsilon.
        function _closeDraft() {
            var d = _polyDraft;
            _dropDraft();                                    // draft nulled FIRST (re-entrancy safety)
            if (!d || d.pts.length < 3) { _polyLastCommit = false; _fireHistory(); return false; }   // Inc-11 (A6)
            var runs = _polyRuns(d.pts);
            if (runs.length === 0) { _polyLastCommit = false; _fireHistory(); return false; }   // Inc-11 (A6): collinear/degenerate/off-mask
            _histBegin();                                    // one _histBegin per POLYGON, not per click
            _polyCommit(runs);
            // ---- Inc-8: append the EDIT RECORD. A shallow copy of the point objects is enough —
            // _polyDraft.pts has just been dropped and is never mutated again. The record does NOT
            // become selected (_dropDraft above already deselected); closing never auto-selects.
            var rec = { id: ++_polySeq, classId: _activeClassId, pts: d.pts.slice() };
            _polys.push(rec);
            _histCommit({ polyId: rec.id, classId: rec.classId, ptsBefore: null, ptsAfter: _copyPts(rec.pts) });
            _polyLastCommit = true;
            _scheduleRender();
            return true;
        }

        // ---- Inc-8: THE reshape write — the ONLY mask write of this increment. Runs ONLY from the
        // _endStroke prologue, with _dragVert already nulled, inside ONE synchronous
        // _histBegin -> writes -> _histCommit triple. Symmetric difference only: O ∩ N (including
        // holes the eraser carved) is left untouched — reshaping never restores erased pixels.
        function _reshapeCommit(rec, newPts) {
            if (!rec || !_mask || !_mctx) { return; }
            // ---- Inc-9 (R4): newPts === null means DELETE. The delete lane reuses this reviewed erase
            // machinery literally — N = ∅, so _runsDiff yields fill = [] and erase = O's pixel set — and
            // then REMOVES the record. It is never emptied: a pts:[] record would poison the NEXT
            // loadPolygons, whose ≥3-vertex validation is atomic.
            var deleting = (newPts === null);
            var O = _recRuns(rec.pts, rec.holes);             // Inc-15 (I-30): holes ride the record (D2)
            var N = deleting ? [] : _recRuns(newPts, rec.holes);   // may be [] if the drag made the ring degenerate
            var d = _runsDiff(O, N);
            // (2) O △ N empty at the RUNS level → no pixel changes hands. Inc-11 (M3): BOTH
            // zero-footprint lanes now push a TAG-ONLY entry (null raster) — the record
            // mutation is undoable and redo is invalidated through the same helper. The
            // _ptsEqual guard stays load-bearing: a zero-movement handle grab-release
            // reaches this branch on EVERY click and must not touch history at all.
            if (d.fill.length === 0 && d.erase.length === 0) {
                if (deleting) {
                    // delete lane: tag BEFORE _removePolyRec so rec.pts is still live.
                    // Undo re-adds the record via _applyPolyFixup's not-found branch (at
                    // the TOP of the z-order, the documented pre-existing fixup rule);
                    // redo removes it via the null branch.
                    _histPushTag({ polyId: rec.id, classId: rec.classId, ptsBefore: _copyPts(rec.pts), ptsAfter: null, holes: rec.holes, traced: rec.traced });
                    _removePolyRec(rec);
                    return;
                }
                if (!_ptsEqual(rec.pts, newPts)) {
                    // reshape lane (e.g. a zero-drag A4 edge insert): tag then mutate. The
                    // helper subsumes the old `_redoStack.length = 0; _fireHistory();` pair.
                    _histPushTag({ polyId: rec.id, classId: rec.classId, ptsBefore: _copyPts(rec.pts), ptsAfter: _copyPts(newPts), holes: rec.holes, traced: rec.traced });
                    rec.pts = _copyPts(newPts);
                }
                return;
            }
            _histBegin();
            _shadow(rec.classId);   // Inc-14: COW pre-image — ONE plane covers both halves (I-21)
            // (4) erase O \ N, CLIPPED to pixels this class still owns (readback-free, via the
            // silhouette). Other classes' silhouettes need no touch: Inc-14 makes the planes
            // independent, so a reshape of THIS record can only ever move THIS class's pixels.
            var se = _silh[rec.classId];
            if (d.erase.length > 0 && se && se.ctx) {
                if (!_polyScratch || _polyScratch.width !== _maskW || _polyScratch.height !== _maskH) {
                    _polyScratch = document.createElement('canvas');
                    _polyScratch.width = _maskW; _polyScratch.height = _maskH;
                    _psctx = _polyScratch.getContext('2d');
                    if (_psctx) { _psctx.imageSmoothingEnabled = false; }
                }
                if (_psctx) {
                    _psctx.globalCompositeOperation = 'source-over';
                    _psctx.clearRect(0, 0, _maskW, _maskH);
                    _fillRuns(_psctx, d.erase, 'source-over', '#fff');
                    _psctx.globalCompositeOperation = 'destination-in';
                    _psctx.drawImage(se.canvas, 0, 0);        // scratch = erase ∩ {mask == classId}
                    _psctx.globalCompositeOperation = 'source-over';
                    // identity drawImage of an integer-sized BINARY canvas → no AA, no fractional ids
                    se.ctx.globalCompositeOperation = 'destination-out';
                    se.ctx.drawImage(_polyScratch, 0, 0);
                    se.ctx.globalCompositeOperation = 'source-over';
                }
            }
            // (5) fill N \ O in the RECORD's class (no-op when empty; sets _strokeBBox + _maskDirty)
            _polyCommit(d.fill, rec.classId);
            // (6) _strokeBBox = bbox(O △ N) — NEVER bbox(O ∪ N)
            var eb = _runsBBox(d.erase);
            if (eb) {
                if (!_strokeBBox) {
                    _strokeBBox = { x0: eb.x0, y0: eb.y0, x1: eb.x1, y1: eb.y1 };   // EXPLICIT keys
                } else {
                    if (eb.x0 < _strokeBBox.x0) { _strokeBBox.x0 = eb.x0; }
                    if (eb.y0 < _strokeBBox.y0) { _strokeBBox.y0 = eb.y0; }
                    if (eb.x1 > _strokeBBox.x1) { _strokeBBox.x1 = eb.x1; }
                    if (eb.y1 > _strokeBBox.y1) { _strokeBBox.y1 = eb.y1; }
                }
            }
            _maskDirty = true;                                // covers the erase-only case
            // (7) ONE undo entry, carrying the absolute before/after vertex lists. Inc-9: for a DELETE
            // ptsAfter is null — the exact MIRROR of a creation entry, so _applyPolyFixup's existing
            // null branch drops the record on redo and its not-found branch re-adds it on undo, with
            // ZERO changes to undo/redo/_applyPolyFixup.
            _histCommit({ polyId: rec.id, classId: rec.classId, ptsBefore: _copyPts(rec.pts), ptsAfter: deleting ? null : _copyPts(newPts), holes: rec.holes, traced: rec.traced });
            if (deleting) { _removePolyRec(rec); } else { rec.pts = _copyPts(newPts); }
            _mctx.globalCompositeOperation = 'source-over';   // belt-and-braces
        }

        // ---- Inc-9 (R4): drop an edit record from _polys by IDENTITY. Never writes the mask, never
        // touches history — the caller (_reshapeCommit's delete lane) owns both. Clears the selection
        // and any drag that pointed at the record so no handle survives it.
        function _removePolyRec(rec) {
            for (var i = 0; i < _polys.length; i++) {
                if (_polys[i] === rec) { _polys.splice(i, 1); break; }
            }
            if (_selPoly === rec) { _selPoly = null; _dragVert = null; _selVert = null; }
            _scheduleRender();
        }

        // ---- Inc-B: stroke-level undo/redo helpers (non-frozen) ----
        // Class-op undo (add/remove/rename/recolor) is OUT OF SCOPE for Inc-B; undo scope is strokes only.
        // ---- Inc-14 (M1/§7.2): COPY-ON-WRITE plane shadow. Called from every write path BEFORE
        // that path's first write to plane `sid`, so _histCommit can read a true pre-image even
        // though _histBegin (argument-less, called from the FROZEN _onPointerDown) cannot know
        // which plane the stroke will touch. Idempotent per stroke. Cost = ONE GPU blit per
        // shadowed plane, no readback — exactly today's _undoPre cost. An UNALLOCATED plane
        // shadows as EMPTY, which is its correct pre-image.
        function _shadow(sid) {
            if (!_mask) { return; }
            var key = String(sid);
            if (_shadowPool.hasOwnProperty(key)) { return; }   // already shadowed this stroke
            var cv = _undoPre, cx = _upctx;                    // take the ONE retained spare
            _undoPre = null; _upctx = null;
            if (!cv || !cx || cv.width !== _maskW || cv.height !== _maskH) {   // [AUDIT 5] re-alloc if mask size changed
                cv = document.createElement('canvas');
                cv.width = _maskW; cv.height = _maskH;
                cx = cv.getContext('2d');
                if (!cx) { return; }
                cx.imageSmoothingEnabled = false;
            }
            cx.setTransform(1, 0, 0, 1, 0, 0);
            cx.globalCompositeOperation = 'source-over';
            cx.clearRect(0, 0, _maskW, _maskH);
            var se = _silh[sid];
            if (se && se.canvas) { cx.drawImage(se.canvas, 0, 0); }   // GPU blit, no readback
            _shadowPool[key] = { canvas: cv, ctx: cx };
            _strokeSilhIds.push(sid);
        }

        // Drop the per-stroke shadow pool, keeping exactly ONE canvas in the spare slot — steady
        // state memory is today's single pre-stroke canvas, not a per-class pool (M1).
        function _releaseShadows() {
            var k;
            for (k in _shadowPool) {
                if (_shadowPool.hasOwnProperty(k) && _shadowPool[k] && !_undoPre) {
                    _undoPre = _shadowPool[k].canvas; _upctx = _shadowPool[k].ctx;
                }
            }
            _shadowPool = {};
            _strokeSilhIds.length = 0;
        }

        function _histBegin() {
            if (!_mask) { return; }
            // Inc-14 (§7.3): NO _mask blit here — the pre-image is taken per PLANE, lazily, by
            // _shadow() at that plane's first write. Returns last stroke's canvases to the spare.
            _releaseShadows();
            _strokeBBox = null;
            _eraserDidErase = false;   // Inc-11 (M2): reset the per-stroke suppression flag
        }

        // Inc-8: `tag` is OPTIONAL — {polyId, classId, ptsBefore, ptsAfter}. There is no `kind`
        // discriminator and no second restore path: every entry still carries a raster bbox;
        // the vertex fields are a FIXUP applied after the raster restore (see _applyPolyFixup).
        // Inc-14 (§7.1): the raster payload is `planes: [{sid, before, after}]` — ONE entry
        // restores every plane the stroke touched and no plane it did not. The plane key is
        // `sid`, NEVER `classId`: ent.classId is the Inc-8 POLYGON tag, merged below.
        function _histCommit(tag) {
            if (!_strokeBBox || !_mctx) { return; }   // no paint this stroke (pan/no-op)
            // ---- Inc-11 (M2): a FULLY suppressed eraser stroke (every dab landed inside
            // protected polygon interiors) pushes NO entry — it changed nothing, and
            // pushing would clear the redo chain below. _strokeBBox accumulates BEFORE the
            // eraser branch filters (frozen :584-591), so the bbox alone cannot tell; the
            // flag can. _tool is still 'eraser' during a setTool-driven _endStroke
            // (setTool assigns _tool AFTER _endStroke returns, :1577-1579). The
            // PRE-EXISTING blank-canvas null-op entry (dab emitted runs over empty pixels)
            // is deliberately NOT fixed here — ticket §8.1.
            if (_tool === 'eraser' && !_eraserDidErase) { _strokeBBox = null; return; }
            var x = _strokeBBox.x0, y = _strokeBBox.y0;
            var w = _strokeBBox.x1 - _strokeBBox.x0 + 1;
            var h = _strokeBBox.y1 - _strokeBBox.y0 + 1;
            _strokeBBox = null;                                  // [AUDIT 13] null FIRST — makes re-entry a no-op
            if (w <= 0 || h <= 0) { return; }
            // Inc-14: one {sid, before, after} triple per plane this stroke shadowed. `before` comes
            // from the COW shadow, `after` from the live plane — both over the SAME bbox.
            var planes = [];
            for (var pi = 0; pi < _strokeSilhIds.length; pi++) {
                var sid = _strokeSilhIds[pi];
                var sh = _shadowPool[String(sid)];
                var tse = _ensureSilh(sid);
                if (!sh || !tse) { continue; }
                planes.push({ sid: sid, before: sh.ctx.getImageData(x, y, w, h), after: tse.ctx.getImageData(x, y, w, h) });
            }
            _releaseShadows();                                   // back to ONE retained canvas (M1)
            // Inc-16 (§5): a BARE call is a brush/eraser stroke (only _endStroke calls without a tag);
            // re-trace the neighbourhood if the stroke met a traced record. Runs AFTER the M2 gate
            // and reads the LIVE plane, so no shadow is needed. Writes no pixels.
            var rt = (tag === undefined) ? _retraceStroke(_activeClassId, x, y, w, h) : null;
            var ent = { x: x, y: y, w: w, h: h, planes: planes };
            _undoStack.push(ent);
            // Inc-15 (§7.3): the single-record tag copies exactly as before; `holes`/`traced` ride
            // beside it (deep-copied); a trace entry carries `polys` and NO polyId.
            if (tag) {
                if (tag.polyId !== undefined) { ent.polyId = tag.polyId; ent.classId = tag.classId; ent.ptsBefore = tag.ptsBefore; ent.ptsAfter = tag.ptsAfter; }
                if (tag.holes !== undefined && tag.holes) { ent.holes = tag.holes.map(_copyPts); }
                if (tag.traced === true) { ent.traced = true; }
                if (tag.polys !== undefined) { ent.polys = tag.polys; }
            }
            if (rt) { ent.retrace = rt; }
            while (_undoStack.length > (opts.undoDepth || DEFAULTS.undoDepth)) { _undoStack.shift(); }
            _redoStack.length = 0;
            _fireHistory();
        }

        // ---- Inc-11 (M3): push a TAG-ONLY entry (null raster) for the two zero-footprint
        // _reshapeCommit lanes — no pixel changed but the record did, and a record
        // mutation must invalidate redo AND be undoable. Same trim / redo-clear / fire
        // discipline as _histCommit; undo()/redo() skip the raster restore on an EMPTY
        // planes list (invariant I4). These entries DO count against opts.undoDepth.
        function _histPushTag(tag) {
            var ent = { x: 0, y: 0, w: 0, h: 0, planes: [] };   // Inc-14: tag-only ⇒ no raster planes
            ent.polyId = tag.polyId; ent.classId = tag.classId;
            ent.ptsBefore = tag.ptsBefore; ent.ptsAfter = tag.ptsAfter;
            if (tag.holes !== undefined && tag.holes) { ent.holes = tag.holes.map(_copyPts); }   // Inc-15 (§7.3)
            if (tag.traced === true) { ent.traced = true; }                                   // (a tag-only entry never carries `polys`)
            _undoStack.push(ent);
            while (_undoStack.length > (opts.undoDepth || DEFAULTS.undoDepth)) { _undoStack.shift(); }
            _redoStack.length = 0;
            _fireHistory();
        }

        // Re-derive per-class silhouettes over the bbox from mask ground-truth.
        // [AUDIT 4] the getImageData result AND each byId[id] buffer are BOTH w×h subregion-local
        // and share the same stride — index them with the SAME p; write back with (x,y) offset.
        function _rebuildSilh(x, y, w, h) {
            var k;
            for (k in _silh) {                                    // clear the bbox in EVERY silhouette
                if (_silh.hasOwnProperty(k) && _silh[k] && _silh[k].ctx) {
                    _silh[k].ctx.clearRect(x, y, w, h);
                }
            }
            var md = _mctx.getImageData(x, y, w, h);              // subregion, stride = w
            var src = md.data;
            var byId = {};                                        // id -> Uint8ClampedArray(w*h*4), same subregion stride
            for (var p = 0; p < src.length; p += 4) {            // [AUDIT 4] p indexes the SUBREGION buffer directly
                var id = src[p];
                if (id === 0 || src[p + 3] === 0) { continue; }
                if (!byId[id]) { byId[id] = new Uint8ClampedArray(w * h * 4); }
                var a = byId[id]; a[p] = 255; a[p + 1] = 255; a[p + 2] = 255; a[p + 3] = 255;
            }
            for (var sid in byId) {
                if (!byId.hasOwnProperty(sid)) { continue; }
                var se = _ensureSilh(parseInt(sid, 10));
                if (se) { se.ctx.putImageData(new ImageData(byId[sid], w, h), x, y); }   // (x,y) offset applied here
            }
        }

        // ---- Inc-8: vertex FIXUP applied after a raster restore. Never writes the mask, never
        // schedules its own render (undo/redo call _render() themselves). Inc-15 (§7.3): +holes,
        // +traced — used ONLY by the not-found re-add (a found record keeps its own; a reshape or
        // delete never changes holes). ----
        function _applyPolyFixup(polyId, classId, pts, holes, traced) {
            var rec = null, at = -1, i;
            for (i = 0; i < _polys.length; i++) {
                if (_polys[i].id === polyId) { rec = _polys[i]; at = i; break; }
            }
            if (pts === null || pts === undefined) {              // undo of a CREATION → drop the record
                if (rec) {
                    _polys.splice(at, 1);
                    if (_selPoly === rec) { _selPoly = null; _dragVert = null; _selVert = null; }
                }
                return;
            }
            if (rec) {
                // Inc-13 (OQ-4): the restored pts may have a DIFFERENT length — a surviving idx
                // would name a shifted anchor. Conditional BY ID, not by identity (the not-found
                // branch below re-creates records, so identity can dangle while the id holds) and
                // not unconditional (an undo on a DIFFERENT polygon must not drop the highlight).
                if (_selVert && _selVert.polyId === rec.id) { _selVert = null; }
                rec.pts = _copyPts(pts);
                return;
            }
            // not found + pts present: redo of a creation (or a record dropped by loadPolygons).
            // Re-added at the TOP of the z-order; skipped silently if its class no longer exists
            // (the pixels are still restored — only the edit affordance is missing).
            if (polyId > _polySeq) { _polySeq = polyId; }
            if (_classById(classId)) {
                var nr = { id: polyId, classId: classId, pts: _copyPts(pts) };
                if (holes && holes.length) { nr.holes = holes.map(_copyPts); }   // Inc-15: the re-add carries both
                if (traced === true) { nr.traced = true; }
                _polys.push(nr);
            }
        }

        function undo() {
            if (_destroyed || !_mask) { if (!_mask && !_destroyed) { _warnOnce('no-image', 'OSDAnnotator: no image loaded yet'); } return; }
            if (_painting) { return; }                            // never undo mid-stroke
            if (_polyDraft || _dragVert) { return; }              // Inc-7/8: no-op mid-draft or mid-reshape-drag
            if (!_undoStack.length) { return; }
            var e = _undoStack.pop();
            // Inc-11 (M3): tag-only entries carry an EMPTY planes list and have no raster to
            // restore — skip putImageData/_maskDirty; the vertex fixup below is the whole
            // restore (invariant I4). Inc-14 (§7.3): the restore is PER PLANE and there is NO
            // _rebuildSilh call — re-deriving planes from the demoted _mask was the deterministic
            // overlap destruction this increment exists to remove.
            if (e.planes && e.planes.length) {
                for (var pi = 0; pi < e.planes.length; pi++) {
                    var use = _ensureSilh(e.planes[pi].sid);
                    if (use) { use.ctx.putImageData(e.planes[pi].before, e.x, e.y); }
                }
                _maskDirty = true;                                // isEmpty stays coarse-true latch [AUDIT 10]
            }
            if (e.polyId !== undefined) { _applyPolyFixup(e.polyId, e.classId, e.ptsBefore, e.holes, e.traced); }
            if (e.polys) { _applyPolySet(e.polys.classId, e.polys.before); }   // Inc-15 (§7.3): trace entry → pre-trace record set
            if (e.retrace) { _applyPolyDiff(e.retrace.classId, e.retrace.added, e.retrace.removed); }   // Inc-16 (§7): stroke re-trace → pre-stroke records
            _redoStack.push(e);
            _fireHistory();
            _render();
        }

        function redo() {
            if (_destroyed || !_mask) { if (!_mask && !_destroyed) { _warnOnce('no-image', 'OSDAnnotator: no image loaded yet'); } return; }
            if (_painting) { return; }
            if (_polyDraft || _dragVert) { return; }              // Inc-7/8: no-op mid-draft or mid-reshape-drag
            if (!_redoStack.length) { return; }
            var e = _redoStack.pop();
            // Inc-11 (M3): tag-only entries carry an EMPTY planes list and have no raster to
            // restore — skip putImageData/_maskDirty; the vertex fixup below is the whole
            // restore (invariant I4). Inc-14 (§7.3): per-plane restore, NO _rebuildSilh.
            if (e.planes && e.planes.length) {
                for (var pi = 0; pi < e.planes.length; pi++) {
                    var rse = _ensureSilh(e.planes[pi].sid);
                    if (rse) { rse.ctx.putImageData(e.planes[pi].after, e.x, e.y); }
                }
                _maskDirty = true;
            }
            if (e.polyId !== undefined) { _applyPolyFixup(e.polyId, e.classId, e.ptsAfter, e.holes, e.traced); }
            if (e.polys) { _applyPolySet(e.polys.classId, e.polys.after); }    // Inc-15 (§7.3): trace entry → post-trace record set
            if (e.retrace) { _applyPolyDiff(e.retrace.classId, e.retrace.removed, e.retrace.added); }   // Inc-16 (§7): stroke re-trace → post-stroke records
            _undoStack.push(e);
            _fireHistory();
            _render();
        }

        // Inc-11 (A6): while a draft is open, undo()/redo() early-return (:1099, :1114) —
        // reporting stack length here made the host Undo button render enabled and do
        // nothing (§5.1). Gate on _polyDraft ONLY: a reshape drag holds pointer capture so
        // the button is unreachable mid-drag, and frozen _endStroke cannot guarantee an
        // un-grey on every exit.
        function canUndo() { return !_polyDraft && _undoStack.length > 0; }
        function canRedo() { return !_polyDraft && _redoStack.length > 0; }
        function _fireHistory() {
            if (typeof opts.onHistory === 'function') { opts.onHistory(canUndo(), canRedo()); }
        }

        // ---- Inc-9: host-sync callbacks. Fired ONLY from the bare-key branches of _onKeyDown — never
        // from setTool/setActiveClass (those are host-initiated writes that would echo back at the
        // panel) and never from the Alt path (an Alt hold does not change the MODE, §6.0). onTool
        // therefore only ever receives 'polygon' or 'brush'.
        function _fireTool(t) {
            if (typeof opts.onTool === 'function') { opts.onTool(t); }
        }
        function _fireClass(id) {
            if (typeof opts.onClass === 'function') { opts.onClass(id); }
        }

        // ---- Inc-A: class registry helpers ----
        function _classById(id) {
            for (var i = 0; i < _classes.length; i++) {
                if (_classes[i].id === id) { return _classes[i]; }
            }
            return null;
        }

        function _activeColor() {
            return (_classById(_activeClassId) || {}).color || PALETTE[0];
        }

        function _freeId() {
            for (var id = 1; id <= 255; id++) {
                if (!_classById(id)) { return id; }
            }
            return null;
        }

        function _ensureSilh(id) {
            var se = _silh[id];
            if (se && se.ctx) { return se; }
            var cv = document.createElement('canvas');
            cv.width = _maskW; cv.height = _maskH;
            var cx = cv.getContext('2d');
            if (!cx) { return null; }               // do NOT cache a broken entry
            cx.imageSmoothingEnabled = false;
            se = { canvas: cv, ctx: cx };
            _silh[id] = se;
            return se;
        }

        // ---- Inc-A: two-sentinel CSS-color validator (shared by setColor/addClass/setClassColor) ----
        function _validColor(css) {
            if (typeof css !== 'string' || !_ctx) { return false; }
            var prev = _ctx.fillStyle;
            _ctx.fillStyle = '#000';
            _ctx.fillStyle = css;
            var valid1 = (_ctx.fillStyle !== '#000000');
            _ctx.fillStyle = '#fff';
            _ctx.fillStyle = css;
            var valid2 = (_ctx.fillStyle !== '#ffffff');
            _ctx.fillStyle = prev;
            return valid1 || valid2;
        }

        // ---- E3: pointer FSM event handlers ----
        function _onPointerDown(e) {
            if (_active) { e.stopPropagation(); }
            if (_destroyed || !_active) { return; }
            // SECONDARY (right) button → PAN the viewer while annotating (left paints).
            if (e.button === 2) {
                if (_painting) { return; }                    // never pan mid-stroke
                var vpd = _viewer && _viewer.viewport;
                if (!vpd) { return; }
                e.preventDefault();
                _view.setPointerCapture(e.pointerId);
                _panning = true;
                _panPointerId = e.pointerId;
                _panLastX = e.clientX;
                _panLastY = e.clientY;
                return;
            }
            if (_painting || _panning || !_mask) { return; }  // never paint while panning
            if (e.button !== 0 || !e.isPrimary) { return; }   // PRIMARY mouse button only
            // ---- Inc-7: POLYGON tool — place a vertex / close on double-click. NO _histBegin
            // (that happens once, at close) and NO _stampDisc (clicks paint nothing).
            if (_tool === 'polygon') {
                var rcd = _view.getBoundingClientRect();
                var pcss = { x: e.clientX - rcd.left, y: e.clientY - rcd.top };
                var pnow = e.timeStamp || performance.now();
                // (i) double-click test FIRST, and only against a click-mode draft.
                // 300 ms matches OSD's dblClickTimeThreshold; 6 CSS px NOT OSD's 20 — a deliberate
                // second vertex 15 px from the first would otherwise falsely close the ring.
                if (_polyDraft && _polyDraft.mode === 'click' && _polyLastClick
                    && (pnow - _polyLastClick.t) <= _polyOpts.dblClickMs
                    && Math.hypot(pcss.x - _polyLastClick.x, pcss.y - _polyLastClick.y) <= _polyOpts.dblClickPx) {
                    _closeDraft();                            // the 2nd click of the pair places NO vertex
                    return;
                }
                // (ii) off-mask click → no vertex, no press
                var pmd = _ptFromEvent(e);
                if (!pmd) { return; }
                // ---- Inc-8: SELECTION / RESHAPE predicate (M-2). Evaluated top to bottom; EXACTLY
                // ONE branch fires per click; side-effect-free until it commits to a branch. The
                // grab tolerance is expressed in MASK px via the same screen->mask shape as
                // _getRPx, so it is a screen-constant POLY_HANDLE_PX CSS px at every zoom.
                var tolm = (POLY_HANDLE_PX * _maskScale) / ((_lastSPerImg > 0) ? _lastSPerImg : 1);
                // (1) selected polygon + press within tol of one of ITS vertices -> begin a vertex
                //     drag on the NEAREST such vertex (ties -> lowest index). No _histBegin here:
                //     the whole history triple runs on release, inside _reshapeCommit.
                if (_selPoly) {
                    var bestK = -1, bestD = Infinity;
                    for (var svi = 0; svi < _selPoly.pts.length; svi++) {
                        var svp = _selPoly.pts[svi];
                        var svd = Math.hypot(pmd.x - svp.x, pmd.y - svp.y);
                        if (svd <= tolm && svd < bestD) { bestD = svd; bestK = svi; }
                    }
                    if (bestK >= 0) {
                        // ---- Inc-13: the press SELECTS this anchor (yellow) and, in the same press,
                        // arms the drag below. Selection is written BEFORE the latch so a release that
                        // never moves still leaves the anchor selected; the existing _scheduleRender()
                        // below paints it on this very press. idx mirrors _dragVert.idx (invariant I-13).
                        // Backspace with this selection removes the anchor (_onKeyDown); Esc clears it.
                        _selVert = { polyId: _selPoly.id, idx: bestK };
                        _view.setPointerCapture(e.pointerId);     // the EXISTING latch lines, reused
                        _activePointerId = e.pointerId;
                        _painting = true;
                        _dragVert = { poly: _selPoly, idx: bestK, pts: _copyPts(_selPoly.pts),
                                      downCss: pcss, moved: false };   // Inc-12: deadzone press origin — ON the literal (both sites)
                        _scheduleRender();
                        return;                                   // no vertex placed, no draft
                    }
                    // ---- Inc-11 (A4): EDGE INSERT (user decision 2 — "we click the edge
                    // to add anchor point"). Reached ONLY when the vertex grab above did
                    // not fire; SAME tolm — grab outranks insert by ORDER, not tuning
                    // (every point within tolm of a vertex is within tolm of its adjacent
                    // segments). splice at idx+1 makes the closing segment a plain push;
                    // the drag then runs on the EXISTING _dragVert lanes: _onPointerMove
                    // writes the working copy, _endStroke -> _reshapeCommit commits — a
                    // zero-drag release lands in the zero-diff branch and pushes an M3
                    // tag-only entry. q is a convex combination of two on-mask vertices.
                    var eBest = _nearestEdge(_selPoly.pts, pmd, tolm);
                    if (eBest) {
                        var ipts = _copyPts(_selPoly.pts);
                        ipts.splice(eBest.idx + 1, 0, { x: eBest.x, y: eBest.y });
                        _view.setPointerCapture(e.pointerId);
                        _activePointerId = e.pointerId;
                        _painting = true;
                        // Inc-13: index coherence (invariant I-13) — the NEW anchor is the selected one.
                        // _render draws from _dragVert.pts, which mid-insert holds one EXTRA point, so a
                        // surviving stale _selVert would highlight the wrong handle. Side effect (ruled
                        // intended, D-5): the inserted anchor is yellow and immediately Backspace-removable.
                        _selVert = { polyId: _selPoly.id, idx: eBest.idx + 1 };
                        _dragVert = { poly: _selPoly, idx: eBest.idx + 1, pts: ipts,
                                      downCss: pcss, moved: false };   // Inc-12: deadzone fields — BOTH creation sites (never _polyDownCss)
                        _scheduleRender();
                        return;                                   // no vertex placed, no draft
                    }
                }
                // (2) a DRAFT always wins -> fall through to (iii)(iv) below unchanged.
                if (!_polyDraft) {
                    // (2.9) Inc-13c: a selection only governs INTERIOR clicks for its OWN class.
                    //       Without this, a class-C polygon left selected while class D is active
                    //       swallowed every interior press (3.5 keys on _selPoly identity, 4 on it
                    //       being non-null) and blocked cross-class NESTING — the escape hatch for
                    //       the accepted same-class limitation (ticket §11.4).
                    //       PLACEMENT IS DELIBERATE AND TESTED: this sits BELOW clause (1) and the
                    //       A4 edge insert, so a press in the POLY_HANDLE_PX band along the
                    //       selected polygon's vertices/edges STILL reshapes that record even when
                    //       its class is no longer active — pinned by inc11 I-09
                    //       ("class switch KEEPS selection and insert uses the RECORD class") and
                    //       inc8 R-16 ("reshape writes record's own class"). Moving this above
                    //       clause (1) breaks both. The split is the design: BOUNDARY = edit the
                    //       selected record, INTERIOR = draw in the active class.
                    //       Deliberately NOT done in setActiveClass(): public API, and inc8 R-18
                    //       calls it and then presses Esc against the live selection.
                    if (_selPoly && _selPoly.classId !== _activeClassId) {
                        _selPoly = null; _selVert = null;
                    }
                    // (3) click inside a SAME-CLASS, not-currently-selected polygon -> select the
                    //     TOPMOST such record (last index first). The click is consumed: no latch,
                    //     no press, no vertex. A click inside a DIFFERENT-class polygon fails here
                    //     and places a vertex immediately.
                    var hitp = null;
                    for (var hpi = _polys.length - 1; hpi >= 0; hpi--) {
                        var cand = _polys[hpi];
                        if (cand.classId !== _activeClassId || cand === _selPoly) { continue; }
                        if (_ptInPoly(cand.pts, pmd.x, pmd.y)) { hitp = cand; break; }
                    }
                    if (hitp) { _selPoly = hitp; _selVert = null; _scheduleRender(); return; }   // Inc-13: another polygon clears the anchor
                    // (3.5) Inc-13b: a click INSIDE the currently selected polygon KEEPS it
                    //       selected and is CONSUMED. If an anchor was selected it is peeled
                    //       off (the same end state as Esc #1). It NEVER opens a nested draft:
                    //       clicking the middle of your own selected polygon must be a no-op,
                    //       not a way to accidentally start a new one.
                    //       ORDER: this sits AFTER the clause-3 scan, so a click into the
                    //       OVERLAP of another same-class polygon still SELECTS that other
                    //       record — clause 3 outranks this.
                    if (_selPoly && _ptInPoly(_selPoly.pts, pmd.x, pmd.y)) {
                        if (_selVert) { _selVert = null; _scheduleRender(); }
                        return;                                  // consumed: no latch, no vertex
                    }
                    // (4) Inc-13b: the click hit NOTHING, but something IS selected -> act as
                    //     Esc: DESELECT and CONSUME. It does NOT open a draft. A SECOND click,
                    //     now with nothing selected, starts the new polygon as before. This
                    //     stops "I clicked away to dismiss the selection" from silently
                    //     placing the first vertex of a polygon the user never asked for.
                    if (_selPoly) {
                        _selPoly = null; _selVert = null; _dragVert = null;   // byte-for-byte the Esc-#2 end state
                        _scheduleRender();
                        return;                                  // consumed: no latch, no vertex
                    }
                    // (5) nothing selected -> fall through to (iii)(iv) and start/continue a draft.
                }
                // (iii) latch the press by REUSING the brush latch (_painting + capture), so
                // right-drag pan, undo/redo and setActive(false) all behave mid-press as they do
                // mid-stroke. Between clicks _painting is false, so right-drag pan still works.
                _view.setPointerCapture(e.pointerId);
                _activePointerId = e.pointerId;
                _painting = true;
                // (iv) place the vertex
                if (!_polyDraft) { _polyDraft = { mode: 'click', pts: [pmd] }; }
                else { _polyDraft.pts.push(pmd); }
                _polyDownCss = pcss; _polyLastCss = pcss; _polyCurMask = null;
                _polyLastClick = { t: pnow, x: pcss.x, y: pcss.y };
                _fireHistory();   // Inc-11 (A6): a draft is open -> canUndo/canRedo are now
                                  // false; grey the host buttons THIS click (§5.1 fix)
                _scheduleRender();
                return;
            }
            _view.setPointerCapture(e.pointerId);
            _activePointerId = e.pointerId;
            _painting = true;
            _histBegin();   // Inc-B: snapshot pre-stroke pixels + reset stroke bbox (paint branch only)
            var p = _ptFromEvent(e);
            if (p) {
                _stampDisc(p.x, p.y);
                _lastPt = p;
                _maskDirtyOnBrush();
            }
            _scheduleRender();
        }

        function _onPointerMove(e) {
            if (_active) { e.stopPropagation(); }
            if (_panning && e.pointerId === _panPointerId) {
                var vpm = _viewer && _viewer.viewport;
                if (vpm) {
                    var dx = e.clientX - _panLastX;
                    var dy = e.clientY - _panLastY;
                    _panLastX = e.clientX;
                    _panLastY = e.clientY;
                    // Match OSD's own drag: a flipped (mirrored) viewport negates the
                    // horizontal delta BEFORE the pan negation (openseadragon.js onCanvasDrag),
                    // else right-drag pans the wrong way when flipped.
                    if (vpm.getFlip && vpm.getFlip()) { dx = -dx; }
                    // drag = content follows cursor → pan by the NEGATED pixel delta
                    vpm.panBy(vpm.deltaPointsFromPixels(new OpenSeadragon.Point(-dx, -dy)));
                    vpm.applyConstraints();
                }
                return;
            }
            // ---- Inc-7: POLYGON tool — freehand mode selection + spaced sampling.
            // The rubber band needs no code here: _onHoverMove already tracks _hoverCss and
            // schedules a render on every pointermove (press or no press).
            if (_tool === 'polygon') {
                // ---- Inc-8: vertex RESHAPE drag — the preview is VECTOR-ONLY. The mask is written
                // exactly once, on release (_endStroke -> _reshapeCommit); nothing is rasterized
                // per pointermove. _ptFromEvent is recomputed per event from a live bounding rect +
                // the current viewport, so the vertex tracks the cursor correctly through a zoom.
                if (_dragVert) {
                    if (!_painting || e.pointerId !== _activePointerId) { return; }
                    var pdm = _ptFromEvent(e);
                    if (!pdm) { return; }                         // off-mask sample skipped; the vertex stays put
                    // ---- Inc-12: movement DEADZONE (the shipped jitter-nudge fix). Until the
                    // press has travelled MORE than POLY_DRAG_PX CSS px from its origin, the
                    // working copy is NOT written and nothing renders — a jittery double-click
                    // press leaves rec.pts untouched, _ptsEqual stays true at release, and
                    // _reshapeCommit returns before touching history. STICKY once crossed (a
                    // drag that returns to its origin still counts as moved). The gate computes
                    // its OWN CSS point: the `mcss` below is a hoisted var, still undefined
                    // here. downCss/moved live ON the _dragVert literal (both creation sites) —
                    // NEVER read _polyDownCss, which is null on the edge-insert lane. Applies
                    // to BOTH _dragVert lanes (vertex grab AND Inc-11 edge-insert) uniformly:
                    // a sub-threshold insert-drag now leaves the new vertex at its exact edge
                    // projection. Same shape as the click->freehand promotion below.
                    if (!_dragVert.moved) {
                        var rdz = _view.getBoundingClientRect();
                        var dcss = { x: e.clientX - rdz.left, y: e.clientY - rdz.top };
                        if (Math.hypot(dcss.x - _dragVert.downCss.x, dcss.y - _dragVert.downCss.y) <= POLY_DRAG_PX) { return; }
                        _dragVert.moved = true;
                    }
                    _dragVert.pts[_dragVert.idx] = pdm;           // WORKING COPY only
                    _scheduleRender();
                    return;
                }
                if (!_painting || e.pointerId !== _activePointerId || !_polyDraft) { return; }
                var rcm = _view.getBoundingClientRect();
                var mcss = { x: e.clientX - rcm.left, y: e.clientY - rcm.top };
                if (_polyDraft.mode === 'click') {
                    // ONLY the draft-creating press may become freehand; the mode is then sticky
                    // for the rest of the draft (one mode decision per draft — a later drag must
                    // never silently close a shape the user is still clicking out).
                    if (_polyDraft.pts.length !== 1 || !_polyDownCss
                        || Math.hypot(mcss.x - _polyDownCss.x, mcss.y - _polyDownCss.y) <= POLY_DRAG_PX) { return; }
                    _polyDraft.mode = 'freehand';
                }
                var pmm = _ptFromEvent(e);
                if (!pmm) { return; }                         // off-mask sample skipped; the press continues
                _polyCurMask = pmm;                           // remembered as the release-tail vertex
                if (!_polyLastCss
                    || Math.hypot(mcss.x - _polyLastCss.x, mcss.y - _polyLastCss.y) >= _polyOpts.stepPx) {
                    _polyDraft.pts.push(pmm); _polyLastCss = mcss;
                }
                _scheduleRender();
                return;
            }
            if (!_painting || e.pointerId !== _activePointerId) { return; }
            var p = _ptFromEvent(e);
            if (!p) { return; }
            _stamp(_lastPt, p);
            _lastPt = p;
            _scheduleRender();
        }

        function _onPointerUp(e) {
            if (_active) { e.stopPropagation(); }
            if (_panning && e.pointerId === _panPointerId) { _endPan(); return; }
            if (e.pointerId !== _activePointerId) { return; }
            _endStroke();
        }

        function _onPointerCancel(e) {
            if (_active) { e.stopPropagation(); }
            if (_panning && e.pointerId === _panPointerId) { _endPan(); return; }
            if (e.pointerId !== _activePointerId) { return; }
            _endStroke();
        }

        function _onLostCapture(e) {
            if (_panning && e.pointerId === _panPointerId) { _endPan(); return; }
            if (e.pointerId !== _activePointerId) { return; }
            _endStroke();
        }

        function _onPointerLeave(e) {
            if (_painting && e.pointerId === _activePointerId) { _endStroke(); }
        }

        // ---- E-INC4: R2 belt-and-suspenders click/dblclick swallower ----
        function _onClickSwallow(e) {
            if (_active) { e.preventDefault(); e.stopPropagation(); e.stopImmediatePropagation(); }
        }

        // ---- right-drag pan: suppress the browser context menu while annotating ----
        // (When inactive, _view is pointerEvents:none so this never fires and the host's
        //  canvas-contextmenu → Spectrum Inspector handler runs normally.)
        function _onContextMenu(e) {
            if (_active) { e.preventDefault(); e.stopPropagation(); }
        }

        // ---- E-INC4: R3 hover tracking ----
        function _onHoverMove(e) {
            if (_destroyed || !_active || !_view) { return; }
            var r = _view.getBoundingClientRect();
            _hoverCss = { x: e.clientX - r.left, y: e.clientY - r.top };
            _scheduleRender();
        }

        function _onHoverLeave() {
            if (_hoverCss !== null) { _hoverCss = null; _scheduleRender(); }
        }

        // ---- Shift+wheel: resize the (screen-space) brush while active ----
        // Plain wheel falls through untouched so OSD keeps zooming; only Shift acts.
        function _onWheel(e) {
            if (_destroyed || !_active) { return; }              // inactive/destroyed → OSD zoom untouched (req 7)
            if (!e.shiftKey) { return; }                         // plain wheel → bubbles to OSD zoom (req 1)
            e.preventDefault(); e.stopPropagation();             // Shift+wheel must NOT also zoom (req 2)
            var delta = (e.deltaY !== 0) ? e.deltaY : e.deltaX;  // Chrome/Windows remaps Shift+wheel to deltaX
            if (!delta) { return; }
            var up = delta < 0;                                  // wheel up = larger brush
            var next = Math.round(_brushImg * (up ? 1.15 : 1 / 1.15));  // proportional feel (req 3)
            if (next === _brushImg) { next = _brushImg + (up ? 1 : -1); }  // guarantee ≥1px change near the floor
            if (next < 1) { next = 1; }
            if (next > 100) { next = 100; }                      // clamp+integer [1,100] to match the slider (req 3)
            if (next === _brushImg) { return; }                  // already at clamp → no redundant work
            setBrushSize(next);                                  // reuse frozen setter (sets _brushImg)
            if (_brushImg !== next) { return; }                  // Codex#8: setter no-op'd (no mask yet) — don't desync host slider/ring
            if (_view) {                                          // Codex#1: synthesize hover pos so the ring
                var wr = _view.getBoundingClientRect();          // redraws at the new size on the FIRST scroll,
                _hoverCss = { x: e.clientX - wr.left, y: e.clientY - wr.top };  // before any pointermove (req 5)
            }
            _scheduleRender();                                   // re-render the bold ring at the new size (req 5)
            if (typeof opts.onBrushSize === 'function') { opts.onBrushSize(next); }  // optional host sync (req 4)
        }

        // ---- Inc-9: typing-guard predicate. TRUE only for genuine text-entry targets. The old
        // guard bailed on ANY <input>, so a focused radio/checkbox/range (i.e. after EVERY panel
        // click) silently killed Ctrl+Z (ticket §3.1 road 2). Polarity is a NEGATIVE list of
        // input types: an unknown/future type resolves to 'text' behaviour in browsers, so it
        // must fall on the GUARDED side by default. Backspace is the sharpest case: inside a
        // text field it must delete a character, never a polygon (R4).
        function _isTypingTarget(t) {
            if (!t) { return false; }
            if (t.isContentEditable) { return true; }
            var tag = t.tagName;
            if (tag === 'TEXTAREA' || tag === 'SELECT') { return true; }
            if (tag === 'INPUT') { return !NON_TEXT_INPUT_TYPES.test(t.type); }
            return false;
        }

        // ---- Inc-B: keyboard undo/redo (annotator-scoped = _active-gated) ----
        // ---- Inc-9: + the bare-key map — R1 (P/B mode), R2 (1-9 class), R3 (Alt-hold eraser),
        // R4 (Backspace/Delete deletes the selected polygon). preventDefault ONLY, never
        // stopPropagation: OSD's canvas handler has already run and its key set is disjoint, so
        // every key this handler does not consume reaches OSD/the browser untouched.
        function _onKeyDown(e) {
            if (_destroyed || !_active) { return; }               // [AUDIT 7] a disarmed annotator never hijacks host Ctrl+Z
            var t = e.target;                                      // typing guard: let native text editing win
            if (_isTypingTarget(t)) { return; }                    // Inc-9 §4
            // ---- Inc-7: Esc cancels / Enter finishes an open polygon draft. preventDefault ONLY
            // when a draft was actually consumed, so an idle annotator never hijacks Esc/Enter.
            var key = e.key || '';
            // Inc-8: Esc ALSO deselects (and aborts a held reshape drag, via _dropDraft).
            if (key === 'Escape') {
                // Inc-13: the ANCHOR selection peels off FIRST, leaving the polygon selected;
                // a second Esc then behaves exactly as before. NOT while a drag is live — a
                // mid-drag Esc must still abort the drag through cancelPolygon/_dropDraft.
                if (_selVert && !_dragVert && !_polyDraft) {
                    e.preventDefault(); _selVert = null; _scheduleRender(); return;
                }
                if (_polyDraft || _selPoly) { e.preventDefault(); cancelPolygon(); }
                return;
            }
            if (key === 'Enter') { if (_polyDraft) { e.preventDefault(); finishPolygon(); } return; }
            if (e.ctrlKey || e.metaKey) {
                var k = key.toLowerCase();
                if (k === 'z' && !e.shiftKey) { e.preventDefault(); undo(); }
                else if ((k === 'z' && e.shiftKey) || k === 'y') { e.preventDefault(); redo(); }
                return;                                            // Ctrl/Meta+anything-else: untouched
            }
            // ---- Inc-9: bare keys. Reached only when armed, not typing, no Ctrl/Meta. ----
            if (!_mask) { return; }                                // pre-init: consume nothing, desync nothing
            if (key === 'Alt') {                                   // R3 engage (§6.1)
                if (e.shiftKey) { return; }                        // Shift+Alt combos: not ours
                if (e.repeat) {                                    // OS auto-repeat of a held Alt:
                    if (_altSavedTool !== null) { e.preventDefault(); }   // keep suppressing the menu heuristic
                    return;                                        // captured ONCE, on the first keydown
                }
                if (_altSavedTool !== null) { e.preventDefault(); return; }   // second physical Alt key: no re-latch
                if (_tool !== 'brush') { return; }                 // R6: Alt erases ONLY in Brush mode. In Polygon
                                                                   // mode (or internal 'eraser') Alt does NOTHING —
                                                                   // no latch, not consumed, browser default intact
                if (_painting || _polyDraft) { return; }           // press in flight / draft open: NO-OP (§6.6)
                _altSavedTool = _tool;                             // always 'brush' here (I2)
                setTool('eraser');                                 // frozen setTool, CALLED not edited
                e.preventDefault();                                // menu-bar suppression, half 1 (§6.4)
                return;
            }
            if (e.altKey || e.shiftKey) { return; }                // Alt combos are the browser's (Alt+Backspace =
                                                                   // history-back on some platforms); Shift is OSD zoom's
            if (key === 'Backspace' || key === 'Delete') {         // R4 (§7) + Inc-13 anchor lane
                // ---- Inc-13 (spec §6.2a): ONE physical press = ONE removal. Without this,
                // the OS auto-repeat of a HELD Backspace ESCALATES: keydown #1 removes the
                // anchor and nulls _selVert while _selPoly survives, so keydown #2 sees a
                // polygon-only selection and falls to the delete lane below — a user who
                // held the key a beat too long loses the WHOLE record. Consumed, not
                // ignored, so the browser's history-back default never fires either.
                // Mirrors the Alt branch's own e.repeat guard above.
                if (e.repeat) { e.preventDefault(); return; }
                if (_selPoly && !_painting) {
                    // ---- Inc-13: ANCHOR lane. Removal is a RESHAPE — rpts is a real array, NEVER
                    // null (null is the polygon-delete lane below). The full three-conjunct validity
                    // test guards the increment's only data-destroying path; the range check lives
                    // HERE deliberately, as the counterpart to keeping it OUT of _render.
                    if (_selVert && _selVert.polyId === _selPoly.id
                        && _selVert.idx >= 0 && _selVert.idx < _selPoly.pts.length) {
                        e.preventDefault();                        // consumed either way, refusal included —
                                                                   // BEFORE the return below: a fall-through
                                                                   // would delete the WHOLE polygon
                        if (_selPoly.pts.length <= 3) { return; }  // a triangle keeps its three: refusal is
                                                                   // consumed-and-inert; selection AND anchor
                                                                   // stay (the anchor remains yellow)
                        var rpts = _copyPts(_selPoly.pts);
                        rpts.splice(_selVert.idx, 1);
                        _selVert = null;                           // clearing condition 4, BEFORE the commit
                                                                   // (the _dragVert-nulled-first precedent)
                        _reshapeCommit(_selPoly, rpts);            // symmetric-difference repaint + ONE entry;
                                                                   // a collinear anchor lands in the tag-only
                                                                   // branch automatically
                        _scheduleRender();                         // REQUIRED: _reshapeCommit does not render
                                                                   // on the non-delete lane
                        return;
                    }
                    _reshapeCommit(_selPoly, null);                // UNCHANGED polygon-delete lane
                    e.preventDefault();
                }
                return;                                            // nothing selected / mid-press: NOT consumed
            }
            var lk = key.toLowerCase();
            if (lk === 'b' || lk === 'p') {                        // R1 — 'e' is deliberately ABSENT (R6)
                var tn = (lk === 'b') ? 'brush' : 'polygon';
                _altSavedTool = null;                              // an explicit mode choice dissolves any stale Alt latch (§6.5)
                setTool(tn);
                _fireTool(tn);
                e.preventDefault();
                return;
            }
            if (key.length === 1 && key >= '1' && key <= '9') {    // R2
                var ci = key.charCodeAt(0) - 49;                   // '1' -> 0 … '9' -> 8
                if (ci < _classes.length) {
                    setActiveClass(_classes[ci].id);
                    _fireClass(_classes[ci].id);
                    e.preventDefault();
                }
                return;                                            // out-of-range digit: NOT consumed
            }
        }

        // ---- Inc-9 (R3): Alt-hold release. Deliberately NOT _active-gated and NOT typing-guarded:
        // a latch can only exist if the matching keydown was consumed while armed (§3 I1), and
        // it MUST resolve even if the user disarmed, refocused an input, or clicked a radio
        // mid-hold — otherwise the tool is stranded on the eraser. The annotator still never
        // consumes a keyup whose keydown it did not consume (I5).
        function _onKeyUp(e) {
            if (_destroyed) { return; }
            if ((e.key || '') !== 'Alt') { return; }
            if (_altSavedTool === null) { return; }                // no hold latched: fall through untouched
            var saved = _altSavedTool;
            _altSavedTool = null;                                  // resolve the latch FIRST (re-entrancy)
            if (_tool === 'eraser') {                              // §6.5: restore ONLY if still on the temp eraser
                setTool(saved);                                    // back to 'brush'; no _fireTool (§6.0)
            }
            e.preventDefault();                                    // menu-bar suppression, half 2 (§6.4)
        }

        // ---- Inc-9 (R3): Alt+Tab / any focus loss mid-hold. The keyup will never arrive; restore
        // NOW. Same restore rule as _onKeyUp; blur is not cancelable so there is no
        // preventDefault. Alt+Tab, tab switch, devtools focus and native dialogs all fire
        // window blur before any further annotator pointer input is possible.
        function _onWindowBlur() {
            if (_destroyed) { return; }
            if (_altSavedTool === null) { return; }
            var saved = _altSavedTool;
            _altSavedTool = null;
            if (_tool === 'eraser') {
                setTool(saved);
            }
        }

        // ---- E8: rAF coalescer ----
        function _scheduleRender() {
            if (_destroyed) { return; }
            if (_rafId !== null) { return; }                 // one rAF in flight (coalesced)
            _rafId = window.requestAnimationFrame(function () {
                _rafId = null;
                if (_destroyed) { return; }                    // OBJ#7-guard: never fire into nulled refs
                _render();
            });
        }

        // ---- E7: DOM listener helper ----
        function _addDomHandler(target, event, fn, opts3) {
            target.addEventListener(event, fn, opts3 || false);
            _handlers.push({ target: target, event: event, fn: fn, dom: true });
        }

        function setTool(t) {
            if (_destroyed || !_mask) {
                if (!_mask && !_destroyed) { _warnOnce('no-image', 'OSDAnnotator: no image loaded yet'); }
                return undefined;
            }
            if (t === 'brush' || t === 'eraser' || t === 'polygon') {
                if (t !== _tool) {
                    _dropDraft();                             // Inc-7: a tool change drops any open draft
                    // ...and a LATCHED PRESS must not survive the change either. The polygon branch
                    // of _onPointerDown latches _painting/_activePointerId WITHOUT _histBegin (per
                    // spec §17 clicks are free — no per-click mask blit). A press left latched here
                    // would be adopted by the brush/eraser branch of _onPointerMove, which paints
                    // against a STALE _undoPre (→ a corrupt undo entry that destroys pixels the
                    // stroke never touched) and never sets _maskDirty (→ isEmpty() true, exportPNG()
                    // null, the annotation unsavable). Ordered _dropDraft-then-_endStroke exactly as
                    // setActive(false) is, so a freehand press is DROPPED, not committed; for a real
                    // brush press _endStroke still commits its partial stroke correctly.
                    if (_painting) { _endStroke(); }
                }
                _tool = t;
            }
            return undefined;
        }

        function setBrushSize(px) {
            if (_destroyed || !_mask) {
                if (!_mask && !_destroyed) { _warnOnce('no-image', 'OSDAnnotator: no image loaded yet'); }
                return undefined;
            }
            if (typeof px === 'number' && isFinite(px) && px > 0) { _brushImg = px; }
            return undefined;
        }

        // ---- Inc-7: polygon public API ----
        // Close the open draft. No _tool gate: a draft implies the polygon tool, and a
        // harness-injected draft must also be closeable. Returns true iff pixels were written.
        function finishPolygon() {
            if (_destroyed || !_mask || !_polyDraft) { return false; }
            if (_painting) { _endStroke(); }      // a held FREEHAND press closes inside _endStroke
            return _polyDraft ? _closeDraft() : _polyLastCommit;
        }

        // Discard the open draft AND/OR the selection. Never writes the mask; a held press
        // continues to its normal end and commits nothing (_onPointerMove/_endStroke both guard on
        // _polyDraft, and _dropDraft has already nulled _dragVert). Committed polygons survive:
        // _dropDraft never touches _polys. Returns true iff a draft was dropped or a selection
        // (and any drag) was cleared.
        function cancelPolygon() {
            if (_destroyed || (!_polyDraft && !_selPoly)) { return false; }
            _dropDraft();
            return true;
        }

        // The three M2 tunables only. No _mask guard (pure options, safe pre-open); post-destroy no-op.
        function setPolygonOptions(o) {
            if (_destroyed || !o) { return undefined; }
            var pk = ['dblClickMs', 'dblClickPx', 'stepPx'];
            for (var i = 0; i < pk.length; i++) {
                var v = o[pk[i]];
                if (typeof v === 'number' && isFinite(v) && v >= 0) { _polyOpts[pk[i]] = v; }
            }
            return undefined;
        }

        // ---- Inc-A: class registry public API (MUTATORS guard like setBrushSize) ----
        function addClass(name, color) {
            if (_destroyed || !_mask) {
                if (!_mask && !_destroyed) { _warnOnce('no-image', 'OSDAnnotator: no image loaded yet'); }
                return null;
            }
            if (_classes.length >= (opts.maxClasses || MAX_CLASSES) || _freeId() === null) {
                _warnOnce('max-classes', 'OSDAnnotator: maximum class count reached');
                return null;
            }
            var id = _freeId();
            var nm = (typeof name === 'string' && String(name).trim() !== '') ? String(name).trim() : ('Class ' + id);
            var col = _validColor(color) ? color : PALETTE[(id - 1) % PALETTE.length];
            _classes.push({ id: id, name: nm, color: col });
            _activeClassId = id;                        // new class becomes active
            _scheduleRender();
            return id;
        }

        function removeClass(id) {
            if (_destroyed || !_mask) {
                if (!_mask && !_destroyed) { _warnOnce('no-image', 'OSDAnnotator: no image loaded yet'); }
                return undefined;
            }
            if (_classes.length <= 1) {
                _warnOnce('min-classes', 'OSDAnnotator: cannot remove the last remaining class');
                return undefined;
            }
            if (!_classById(id)) { return undefined; }
            // Inc-14 (A5): dropping the PLANE is the whole operation — the planes are independent
            // ground truth, so there is nothing to punch out of anything else. (The old _mctx
            // punch destroyed OTHER classes' pixels wherever they overlapped this one.)
            // NOTE: does NOT touch _maskDirty — isEmpty stays the coarse first-paint latch (CODEX-2).
            if (_silh[id]) { delete _silh[id]; }         // free the silhouette cache
            // ---- Inc-8: this class's edit records go with its pixels. Nothing holds the _polys
            // ARRAY object (only records, and the getter reads the variable), so a filter is safe.
            // The unconditional nulling costs nothing — a drag cannot be live during a host click
            // (the pointer is captured on _view) — and needs no reasoning about which record went.
            _polys = _polys.filter(function (p) { return p.classId !== id; });
            _selPoly = null; _dragVert = null; _selVert = null;
            for (var i = 0; i < _classes.length; i++) {
                if (_classes[i].id === id) { _classes.splice(i, 1); break; }
            }
            if (_activeClassId === id) { _activeClassId = _classes[0].id; }
            // Inc-B [AUDIT 3]: removeClass destroys this class's pixels (the plane delete above) →
            // a stale undo entry would resurrect deleted-class pixels, so FLUSH history.
            // (Class-op undo — add/remove/rename/recolor — is OUT OF SCOPE for Inc-B.)
            _undoStack.length = 0; _redoStack.length = 0; _fireHistory();
            _scheduleRender();
            return undefined;
        }

        function renameClass(id, name) {
            if (_destroyed || !_mask) {
                if (!_mask && !_destroyed) { _warnOnce('no-image', 'OSDAnnotator: no image loaded yet'); }
                return undefined;
            }
            var cls = _classById(id);
            if (cls) { cls.name = String(name).trim(); }
            return undefined;
        }

        function setClassColor(id, css) {
            if (_destroyed || !_mask) {
                if (!_mask && !_destroyed) { _warnOnce('no-image', 'OSDAnnotator: no image loaded yet'); }
                return undefined;
            }
            var cls = _classById(id);
            if (cls && _validColor(css)) { cls.color = css; }
            _scheduleRender();
            return undefined;
        }

        function setActiveClass(id) {
            if (_destroyed || !_mask) {
                if (!_mask && !_destroyed) { _warnOnce('no-image', 'OSDAnnotator: no image loaded yet'); }
                return undefined;
            }
            if (_classById(id)) { _activeClassId = id; }
            else { _warnOnce('unknown-class', 'OSDAnnotator: setActiveClass — unknown class id'); }
            _scheduleRender();
            return undefined;
        }

        // ---- Inc-B: load an indexed label PNG into the mask (download-only round-trip) ----
        // [AUDIT 2] CONTRACT: `src` MUST be an ALREADY-DECODED ImageBitmap |
        // HTMLImageElement (img.complete===true) | HTMLCanvasElement. The host does all file I/O
        // and calls loadLabel SYNCHRONOUSLY from inside img.onload (§3.2). loadLabel is fully
        // synchronous — it does NO async decode. Passing an un-decoded Image draws a blank canvas
        // (a host bug, not loadLabel's responsibility). Returns boolean.
        function loadLabel(src, classesOrNull) {
            if (_destroyed || !_mask || _frozen) {
                if (!_mask && !_destroyed) { _warnOnce('no-image', 'OSDAnnotator: no image loaded yet'); }
                return false;
            }
            // (2) draw onto a _maskW×_maskH temp canvas with NEAREST (ids never interpolated)
            var tmp = document.createElement('canvas');
            tmp.width = _maskW; tmp.height = _maskH;
            var tctx = tmp.getContext('2d');
            if (!tctx) { return false; }
            tctx.imageSmoothingEnabled = false;
            var sw = src.width || (src.naturalWidth || 0);
            var sh = src.height || (src.naturalHeight || 0);
            if (sw > 0 && sh > 0) {
                var srcAsp = sw / sh, dstAsp = _maskW / _maskH;
                if (Math.abs(srcAsp - dstAsp) / dstAsp > 0.01) {
                    _warnOnce('load-aspect', 'OSDAnnotator: loaded label aspect differs from mask — nearest-scaled to fit');
                }
            }
            tctx.drawImage(src, 0, 0, _maskW, _maskH);
            var imgd = tctx.getImageData(0, 0, _maskW, _maskH);
            var d = imgd.data;
            // (3) collect DISTINCT ids where R>0 && A>0
            var seen = {};
            var ids = [];
            var i;
            for (i = 0; i < d.length; i += 4) {
                var id = d[i];
                if (id > 0 && d[i + 3] > 0 && !seen[id]) { seen[id] = true; ids.push(id); }
            }
            ids.sort(function (p, q) { return p - q; });
            // (4) cap at maxClasses — keep first `cap` distinct ids; drop the rest
            var cap = opts.maxClasses || MAX_CLASSES;
            if (ids.length > cap) {
                _warnOnce('load-overcap', 'OSDAnnotator: loaded label has more classes than the max — extra ids dropped');
            }
            var kept = {};
            var keptIds = [];
            for (i = 0; i < ids.length && keptIds.length < cap; i++) { kept[ids[i]] = true; keptIds.push(ids[i]); }
            // (5) normalize pixels in place: kept id → (id,0,0,255); else → (0,0,0,0)
            for (i = 0; i < d.length; i += 4) {
                var v = d[i];
                if (v > 0 && d[i + 3] > 0 && kept[v]) {
                    d[i + 1] = 0; d[i + 2] = 0; d[i + 3] = 255;
                } else {
                    d[i] = 0; d[i + 1] = 0; d[i + 2] = 0; d[i + 3] = 0;
                }
            }
            _mctx.clearRect(0, 0, _maskW, _maskH);
            _mctx.putImageData(imgd, 0, 0);
            // (6) rebuild _classes — [AUDIT 8] IDS PRESERVED, NEVER RENUMBERED (Save→Load→Save id-stable)
            var newClasses = [];
            var j;
            if (Array.isArray(classesOrNull)) {
                for (j = 0; j < keptIds.length; j++) {
                    var kid = keptIds[j];
                    var match = null;
                    for (var m = 0; m < classesOrNull.length; m++) {
                        if (classesOrNull[m] && classesOrNull[m].id === kid) { match = classesOrNull[m]; break; }
                    }
                    var nm, col;
                    if (match) {
                        nm = (typeof match.name === 'string' && String(match.name).trim() !== '') ? String(match.name).trim() : ('Class ' + kid);
                        col = _validColor(match.color) ? match.color : PALETTE[(kid - 1) % PALETTE.length];
                    } else {
                        // dangling kept-id not covered by JSON → auto-create (every mask id MUST have a class)
                        nm = 'Class ' + kid;
                        col = PALETTE[(kid - 1) % PALETTE.length];
                    }
                    newClasses.push({ id: kid, name: nm, color: col });
                }
            } else {
                for (j = 0; j < keptIds.length; j++) {
                    var aid = keptIds[j];
                    newClasses.push({ id: aid, name: 'Class ' + aid, color: PALETTE[(aid - 1) % PALETTE.length] });
                }
            }
            if (newClasses.length === 0) {
                newClasses.push({ id: 1, name: 'Class 1', color: PALETTE[0] });   // seed Inc-A default (_classes.length>=1 always)
            }
            _classes = newClasses;
            // (7) active = first class
            _activeClassId = _classes[0].id;
            // (8) rebuild silhouettes from scratch over the whole mask
            _silh = {};
            _rebuildSilh(0, 0, _maskW, _maskH);
            // (8b) Inc-8: the loaded pixels have NO vertices — drop every edit record and any
            // selection/drag. This also aborts a drag hijacked by the host's async img.onload
            // (loadLabel is _frozen-guarded, not _painting-guarded). The host re-populates the
            // records afterwards via loadPolygons().
            _polys.length = 0; _selPoly = null; _dragVert = null; _selVert = null;
            // (9) mark dirty + FLUSH history
            _maskDirty = true;
            _undoStack.length = 0; _redoStack.length = 0; _fireHistory();
            // (10) render
            _render();
            return true;
        }

        // ---- Inc-14 (§8.5/A10): load a WHOLE v3 session — one binary mask PNG per class — in ONE
        // call. loadLabel cannot be called N times: it wipes _classes/_silh/_polys and both history
        // stacks per call, so the second call would destroy the first's classes.
        // [AUDIT 2] CONTRACT, same as loadLabel: every `entries[i].image` MUST be an ALREADY-DECODED
        // ImageBitmap | HTMLImageElement (img.complete===true) | HTMLCanvasElement — the host does
        // all file I/O; loadMasks is fully synchronous and does NO async decode.
        //   entries = [{classId, image}]  (one per PNG supplied)
        //   classes = the manifest's `classes` array (the ONLY File→class mapping; no filename parsing)
        // RASTER ONLY: the two-step vector contract is KEPT — the HOST calls
        // loadPolygons(manifest.polygons) afterwards, exactly as the legacy lane does.
        // Never touches _mask; never calls _rebuildSilh (that would flatten the overlap away).
        // Returns boolean; a malformed shape refuses ATOMICALLY (state untouched — loadPolygons precedent).
        function loadMasks(entries, classes) {
            if (_destroyed || !_mask || _frozen) {
                if (!_mask && !_destroyed) { _warnOnce('no-image', 'OSDAnnotator: no image loaded yet'); }
                return false;
            }
            // (1) structural validation FIRST — nothing is mutated until everything has passed
            if (!Array.isArray(entries) || !Array.isArray(classes)) { return false; }
            var i, j;
            for (i = 0; i < entries.length; i++) {
                var en = entries[i];
                if (!en || typeof en.classId !== 'number' || !isFinite(en.classId) ||
                    Math.floor(en.classId) !== en.classId || en.classId < 1 || !en.image) { return false; }
                // Inc-14 [EXT-AUDIT 1]: `!en.image` is a TRUTHINESS test — a string or a plain
                // object passes it and then throws inside drawImage() AFTER _classes/_silh have
                // been replaced, destroying the previous session. Require a DRAWABLE source here,
                // while nothing has been mutated yet (§8.5.1 atomic refusal).
                var _im = en.image;
                var _iw = _im.width || _im.naturalWidth || 0;
                var _ih = _im.height || _im.naturalHeight || 0;
                if (typeof _iw !== 'number' || typeof _ih !== 'number' || !isFinite(_iw) ||
                    !isFinite(_ih) || _iw <= 0 || _ih <= 0) { return false; }
            }
            for (i = 0; i < classes.length; i++) {
                var mc = classes[i];
                if (!mc || typeof mc.id !== 'number' || !isFinite(mc.id) ||
                    Math.floor(mc.id) !== mc.id || mc.id < 1) { return false; }
            }
            // (2) cap at maxClasses — keep the FIRST `cap` manifest classes, in array order
            var cap = opts.maxClasses || MAX_CLASSES;
            if (classes.length > cap) {
                _warnOnce('load-overcap', 'OSDAnnotator: loaded label has more classes than the max — extra ids dropped');
            }
            // (3) rebuild _classes FROM THE MANIFEST — [AUDIT 8] IDS PRESERVED, NEVER RENUMBERED.
            // Name/colour validation is loadLabel's, unchanged. EMPTY classes (a manifest entry with
            // no PNG) are CREATED, with an unallocated plane — the Save→Load→Save round-trip rule.
            var newClasses = [];
            var keptIds = {};
            for (i = 0; i < classes.length && newClasses.length < cap; i++) {
                var cm = classes[i];
                if (keptIds[cm.id]) { continue; }                 // ignore a duplicate id
                var nm = (typeof cm.name === 'string' && String(cm.name).trim() !== '') ? String(cm.name).trim() : ('Class ' + cm.id);
                var col = _validColor(cm.color) ? cm.color : PALETTE[(cm.id - 1) % PALETTE.length];
                keptIds[cm.id] = true;
                newClasses.push({ id: cm.id, name: nm, color: col });
            }
            if (newClasses.length === 0) {
                newClasses.push({ id: 1, name: 'Class 1', color: PALETTE[0] });   // seed Inc-A default (_classes.length>=1 always)
            }
            // (4) pixels: SESSION REPLACE. Each entry is nearest-scaled onto a _maskW×_maskH temp
            // canvas and thresholded at R > 127 (the Save encoding is 255/0; 127 is the midpoint).
            // Entries whose classId is not a kept manifest class are SKIPPED SILENTLY (stale files,
            // not malformed input — the loadPolygons precedent).
            var tmp = document.createElement('canvas');
            tmp.width = _maskW; tmp.height = _maskH;
            var tctx = tmp.getContext('2d');
            if (!tctx) { return false; }
            tctx.imageSmoothingEnabled = false;
            // Inc-14 [EXT-AUDIT 1]: snapshot BEFORE the commit so a throw inside the decode
            // loop (drawImage TypeError, getImageData SecurityError on a tainted canvas)
            // restores the previous session instead of leaving it half-destroyed.
            var _prevClasses = _classes, _prevSilh = _silh, _prevActive = _activeClassId;
            _classes = newClasses;
            _silh = {};
            var loadedAny = false;
            try {
            for (i = 0; i < entries.length; i++) {
                var e2 = entries[i];
                if (!keptIds[e2.classId]) { continue; }
                var src = e2.image;
                var sw = src.width || (src.naturalWidth || 0);
                var sh = src.height || (src.naturalHeight || 0);
                if (sw > 0 && sh > 0) {
                    var srcAsp = sw / sh, dstAsp = _maskW / _maskH;
                    if (Math.abs(srcAsp - dstAsp) / dstAsp > 0.01) {
                        _warnOnce('load-aspect', 'OSDAnnotator: loaded label aspect differs from mask — nearest-scaled to fit');
                    }
                }
                tctx.setTransform(1, 0, 0, 1, 0, 0);
                tctx.globalCompositeOperation = 'source-over';
                tctx.clearRect(0, 0, _maskW, _maskH);
                tctx.drawImage(src, 0, 0, _maskW, _maskH);
                var imgd = tctx.getImageData(0, 0, _maskW, _maskH);
                var d = imgd.data;
                for (j = 0; j < d.length; j += 4) {
                    if (d[j] > 127) { d[j] = 255; d[j + 1] = 255; d[j + 2] = 255; d[j + 3] = 255; }
                    else { d[j] = 0; d[j + 1] = 0; d[j + 2] = 0; d[j + 3] = 0; }
                }
                var se = _ensureSilh(e2.classId);
                if (se) { se.ctx.putImageData(imgd, 0, 0); loadedAny = true; }
            }
            } catch (err) {
                // [EXT-AUDIT 1] atomic refusal: undo the commit and report failure.
                _classes = _prevClasses; _silh = _prevSilh; _activeClassId = _prevActive;
                _warnOnce('load-masks-decode', 'OSDAnnotator: loadMasks could not decode a mask image — load refused, previous session kept');
                return false;
            }
            // (5) active = first kept class; drop every edit record and any selection/drag. The
            // loaded pixels carry NO vertices — the HOST re-populates them via loadPolygons().
            _activeClassId = _classes[0].id;
            _polys.length = 0; _selPoly = null; _dragVert = null; _selVert = null;
            // (6) mark dirty (so isEmpty() goes false and Save works after a load) + FLUSH history
            if (loadedAny) { _maskDirty = true; }
            _undoStack.length = 0; _redoStack.length = 0; _fireHistory();
            _render();
            return true;
        }

        // GETTERS are the INTENTIONAL exception — NO guard/warn; pre-init return []/null (CODEX-3).
        function getClasses() {
            return _classes.map(function (c) { return { id: c.id, name: c.name, color: c.color }; });
        }

        function getActiveClass() {
            var cls = _classById(_activeClassId);
            return cls ? { id: cls.id, name: cls.name, color: cls.color } : null;
        }

        // ---- Inc-8: polygon EDIT-RECORD persistence. The mask stays the deliverable — these two
        // are an additive affordance carried in the EXISTING <label>.classes.json envelope
        // (host-side: {version:2, classes, polygons}). Neither ever writes the mask. ----

        // GETTER (no guard/warn; pre-init / post-destroy -> []). FILE form: ids are per-attach and
        // are NOT exported; array order (= z-order) is preserved.
        function getPolygons() {
            var out = [];
            if (_destroyed) { return out; }
            for (var i = 0; i < _polys.length; i++) {
                var p = _polys[i];
                var pts = [];
                for (var k = 0; k < p.pts.length; k++) { pts.push([p.pts[k].x, p.pts[k].y]); }
                var rec = { classId: p.classId, pts: pts };
                // Inc-15 (§8.1): `holes` ONLY when non-empty, `traced: true` ONLY when set — files
                // without traced/holed records are byte-identical to v3.4.2 output.
                if (p.holes && p.holes.length) {
                    rec.holes = p.holes.map(function (ring) { return ring.map(function (q) { return [q.x, q.y]; }); });
                }
                if (p.traced === true) { rec.traced = true; }
                out.push(rec);
            }
            return out;
        }

        // Replace ALL edit records. Structural validation is ATOMIC (any violation -> _polys stays
        // empty, return false); records naming a class that does not exist are SKIPPED silently
        // (stale vectors, not malformed input). Coordinates are MASK px, floats.
        function loadPolygons(arr) {
            if (_destroyed || !_mask || _frozen) {
                if (!_mask && !_destroyed) { _warnOnce('no-image', 'OSDAnnotator: no image loaded yet'); }
                return false;
            }
            _polys.length = 0; _selPoly = null; _dragVert = null; _selVert = null;   // ALWAYS first
            if (!Array.isArray(arr)) { _scheduleRender(); return false; }
            var i, k, r, ent;
            for (i = 0; i < arr.length; i++) {                      // pass 1: validate EVERYTHING
                r = arr[i];
                if (!r || typeof r !== 'object') { _scheduleRender(); return false; }
                if (typeof r.classId !== 'number' || !isFinite(r.classId)
                    || Math.floor(r.classId) !== r.classId
                    || r.classId < 1 || r.classId > 255) { _scheduleRender(); return false; }
                if (!_validRing(r.pts)) { _scheduleRender(); return false; }
                // Inc-15 (§8.2): OPTIONAL `holes` (Array of valid rings; [] is valid) and OPTIONAL
                // `traced` (boolean) — validated here so the whole file is still refused atomically.
                if (r.holes !== undefined) {
                    if (!Array.isArray(r.holes)) { _scheduleRender(); return false; }
                    for (k = 0; k < r.holes.length; k++) {
                        if (!_validRing(r.holes[k])) { _scheduleRender(); return false; }
                    }
                }
                if (r.traced !== undefined && typeof r.traced !== 'boolean') { _scheduleRender(); return false; }
            }
            for (i = 0; i < arr.length; i++) {                      // pass 2: adopt
                r = arr[i];
                if (!_classById(r.classId)) { continue; }           // unreachable vectors — skipped silently
                var pts = [];
                for (k = 0; k < r.pts.length; k++) { pts.push({ x: r.pts[k][0], y: r.pts[k][1] }); }
                var nrec = { id: ++_polySeq, classId: r.classId, pts: pts };
                // Inc-15 (§8.2): `holes` ONLY if present AND non-empty; `traced: true` ONLY if exactly
                // true (`false` loads UNMARKED — key absent). Loading never INVENTS the marker (I-29).
                if (r.holes !== undefined && r.holes.length) {
                    var hs = [];
                    for (k = 0; k < r.holes.length; k++) {
                        var hring = [];
                        for (var hk = 0; hk < r.holes[k].length; hk++) { hring.push({ x: r.holes[k][hk][0], y: r.holes[k][hk][1] }); }
                        hs.push(hring);
                    }
                    nrec.holes = hs;
                }
                if (r.traced === true) { nrec.traced = true; }
                _polys.push(nrec);
            }
            _scheduleRender();
            return true;
        }

        // GETTER (no guard/warn). ONE full-plane getImageData per call — Export/Save-click only;
        // never called from a render or status path.
        // Inc-14 (A7/I-24): counts the class's OWN plane, alpha > 0. Under overlap the counts
        // DOUBLE-COUNT — a pixel covered by two classes counts once for EACH. That is intended:
        // there is no per-pixel owner any more. An unallocated plane counts 0.
        function getClassPixelCount(id) {
            if (_destroyed || !_mask || !_mctx) { return 0; }
            if (typeof id !== 'number' || !isFinite(id) || Math.floor(id) !== id) { return 0; }
            var se = _silh[id];
            if (!se || !se.ctx) { return 0; }
            var d = se.ctx.getImageData(0, 0, _maskW, _maskH).data;
            var n = 0;
            for (var i = 0; i < d.length; i += 4) { if (d[i + 3] > 0) { n++; } }
            return n;
        }

        // setColor (re-based, NON-frozen) → delegate to the ACTIVE class.
        function setColor(css) {
            setClassColor(_activeClassId, css);
            return undefined;
        }

        function setFillAlpha(a) {
            if (_destroyed || !_mask) {
                if (!_mask && !_destroyed) { _warnOnce('no-image', 'OSDAnnotator: no image loaded yet'); }
                return undefined;
            }
            if (typeof a === 'number' && isFinite(a)) {
                _fillAlpha = a < 0 ? 0 : (a > 1 ? 1 : a);   // clamp [0,1]
                _scheduleRender();
            }
            return undefined;
        }

        function setActive(b) {
            if (_destroyed) { return undefined; }
            var next = !!b;
            // Inc-7: drop the draft BEFORE _endStroke, so a freehand press in progress is DROPPED,
            // not committed, on disarm. Esc is _active-gated, so a draft surviving disarm would be
            // unreachable and would resurrect on re-arm. Also covers the z-eviction path (_onRemove).
            if (!next) { _dropDraft(); }
            if (!next && _painting) { _endStroke(); }   // true->false mid-stroke must end the stroke
            if (!next && _panning) { _endPan(); }        // ...and mid-pan must end the pan
            _active = next;
            _view.style.pointerEvents = _active ? 'auto' : 'none';
            return undefined;
        }

        function isActive() { return !!_active; }

        function isEmpty() { return !_maskDirty; }

        function clear() {
            if (_destroyed || !_mask) {
                if (!_mask && !_destroyed) { _warnOnce('no-image', 'OSDAnnotator: no image loaded yet'); }
                return;
            }
            _mctx.clearRect(0, 0, _maskW, _maskH);
            for (var id in _silh) {
                if (_silh.hasOwnProperty(id) && _silh[id] && _silh[id].ctx) {
                    _silh[id].ctx.clearRect(0, 0, _maskW, _maskH);
                }
            }
            _maskDirty = false;
            _undoStack.length = 0; _redoStack.length = 0; _fireHistory();   // Inc-B: clear FLUSHES history (not undoable)
            _dropDraft();   // Inc-7: the rubber band must not dangle after Clear
            _polys.length = 0;   // Inc-8: the pixels are gone, so the edit records must go too
            _render();
        }

        function exportPNG(opts2) {
            return new Promise(function (resolve) {
                if (_destroyed) { resolve(null); return; }
                if (!_mask) { _warnOnce('export-no-image', 'OSDAnnotator: no image loaded yet'); resolve(null); return; }
                if (isEmpty()) { _warnOnce('export-empty', 'OSDAnnotator: mask is empty — nothing to export'); resolve(null); return; }
                var target = (opts2 && opts2.downscaleShorterSide) || null;
                var scale = target ? Math.min(1, target / Math.min(_maskW, _maskH)) : 1;
                var outW = Math.round(_maskW * scale);
                var outH = Math.round(_maskH * scale);
                var c = document.createElement('canvas');
                c.width = outW; c.height = outH;
                var ectx = c.getContext('2d');
                ectx.imageSmoothingEnabled = false;
                // Inc-14 (A8/A9/A15): ALWAYS a per-class BINARY mask — the two-branch shape is
                // gone. `opts2.classId` (integer, default = the active class) selects the plane so
                // Save's per-class loop needs no setActiveClass round trip. `opts2.activeOnly` is
                // ACCEPTED AND IGNORED: the binary is the only behaviour, so pre-Inc-14 callers
                // passing activeOnly:true get exactly what they got. The indexed Save branch is
                // deleted outright — after Inc-14 the module NEVER derives an indexed view.
                var cid = (opts2 && typeof opts2.classId === 'number' && isFinite(opts2.classId))
                    ? Math.floor(opts2.classId) : _activeClassId;
                var se = _silh[cid];
                if (se && se.canvas) { ectx.drawImage(se.canvas, 0, 0, outW, outH); }
                var ed = ectx.getImageData(0, 0, outW, outH);
                var d = ed.data;
                for (var i = 0; i < d.length; i += 4) {
                    // white where the class covers (plane alpha > 0), black elsewhere; fully opaque.
                    // An unallocated or empty plane yields an all-black PNG.
                    var on = (d[i + 3] > 0) ? 255 : 0;
                    d[i] = on; d[i + 1] = on; d[i + 2] = on; d[i + 3] = 255;
                }
                ectx.putImageData(ed, 0, 0);
                c.toBlob(function (blob) {
                    if (_destroyed) { resolve(null); return; }
                    if (!blob) { _warnOnce('toBlob-null', 'OSDAnnotator: toBlob returned null — export failed'); resolve(null); return; }
                    resolve(blob);
                }, 'image/png');
            });
        }

        // ---- construct ----
        _mount();
        _registerHandler(_viewer, 'update-viewport', _onView);
        _registerHandler(_viewer, 'animation', _onView);
        _registerHandler(_viewer, 'resize', _onView);
        _registerHandler(_viewer.world, 'remove-item', _onRemove);
        _registerHandler(_viewer.world, 'add-item', _onAddPersist);
        _tryInit();
        _addDomHandler(_view, 'pointerdown',        _onPointerDown);
        _addDomHandler(_view, 'pointermove',        _onPointerMove);
        _addDomHandler(_view, 'pointerup',          _onPointerUp);
        _addDomHandler(_view, 'pointercancel',      _onPointerCancel);
        _addDomHandler(_view, 'lostpointercapture', _onLostCapture);
        _addDomHandler(_view, 'pointerleave',       _onPointerLeave);
        _addDomHandler(_view, 'click',    _onClickSwallow);
        _addDomHandler(_view, 'dblclick', _onClickSwallow);
        _addDomHandler(_view, 'contextmenu', _onContextMenu);
        _addDomHandler(_view, 'pointermove',  _onHoverMove);
        _addDomHandler(_view, 'pointerleave', _onHoverLeave);
        _addDomHandler(_view, 'wheel', _onWheel, { passive: false });   // Shift+wheel brush resize; {passive:false} so preventDefault works
        _addDomHandler(document, 'keydown', _onKeyDown);   // Inc-B: Ctrl+Z/Ctrl+Shift+Z/Ctrl+Y (active-gated), OUTSIDE the frozen _addDomHandler_calls span
        _addDomHandler(document, 'keyup',  _onKeyUp);      // Inc-9: Alt-hold release — OUTSIDE the frozen _addDomHandler_calls span
        _addDomHandler(window,   'blur',   _onWindowBlur); // Inc-9: Alt+Tab / focus-loss mid-hold — restore path (§6.3)

        var instance = {
            destroy: destroy,
            setTool: setTool,
            setBrushSize: setBrushSize,
            setActive: setActive,
            isActive: isActive,
            isEmpty: isEmpty,
            clear: clear,
            exportPNG: exportPNG,
            setColor: setColor,
            setFillAlpha: setFillAlpha,
            // ---- Inc-A: multi-class registry API ----
            addClass: addClass,
            removeClass: removeClass,
            renameClass: renameClass,
            setClassColor: setClassColor,
            setActiveClass: setActiveClass,
            getClasses: getClasses,
            getActiveClass: getActiveClass,
            // ---- Inc-B: persistence + stroke-level undo/redo API ----
            loadLabel: loadLabel,
            loadMasks: loadMasks,        // Inc-14 (§8.5): the per-class v3 session load
            undo: undo,
            redo: redo,
            canUndo: canUndo,
            canRedo: canRedo,
            // ---- Inc-7: polygon INPUT tool API ----
            finishPolygon: finishPolygon,
            cancelPolygon: cancelPolygon,
            setPolygonOptions: setPolygonOptions,
            // ---- Inc-8: polygon RESHAPE persistence + the empty-class Export guard ----
            getPolygons: getPolygons,
            loadPolygons: loadPolygons,
            getClassPixelCount: getClassPixelCount,
            // ---- Inc-15: raster → vector tracing; Inc-17: the json-less Load auto-trace ----
            traceActiveClass: traceActiveClass,
            traceLoadedClasses: traceLoadedClasses,
            // private — exposed for Inc1 tests/harness (instance._render(), __seedMask, __readView)
            _render: _render
        };

        // Expose internal state via accessor properties so harness hooks
        // (__seedMask / __readView) can reach _mask/_mctx/_ctx/_view/_refTI/_maskScale
        // without the module leaking them as a public API.
        Object.defineProperties(instance, {
            _viewer: { get: function () { return _viewer; } },
            _canvas: { get: function () { return _canvas; } },
            _view: { get: function () { return _view; } },
            _ctx: { get: function () { return _ctx; } },
            _mask: { get: function () { return _mask; } },
            _mctx: { get: function () { return _mctx; } },
            _refTI: { get: function () { return _refTI; }, set: function (v) { _refTI = v; } },
            _maskScale: { get: function () { return _maskScale; } },
            _imgW: { get: function () { return _imgW; } },
            _imgH: { get: function () { return _imgH; } },
            _maskW: { get: function () { return _maskW; } },
            _maskH: { get: function () { return _maskH; } },
            _maskDirty: { get: function () { return _maskDirty; }, set: function (v) { _maskDirty = v; } },
            _frozen: { get: function () { return _frozen; } },
            _destroyed: { get: function () { return _destroyed; } },
            _handlers: { get: function () { return _handlers; } },
            _firedWarnKeys: { get: function () { return _firedWarnKeys; } },
            _tool:            { get: function () { return _tool; } },
            _brushImg:        { get: function () { return _brushImg; } },
            _painting:        { get: function () { return _painting; } },
            _lastPt:          { get: function () { return _lastPt; } },
            _activePointerId: { get: function () { return _activePointerId; } },
            _rafId:           { get: function () { return _rafId; } },
            __mapFn:          { get: function () { return _pointerToMask; } },  // (= __viewToImageMask in harness Part F)
            _color:       { get: function () { return _activeColor(); } },   // Inc-A: active class color
            _fillAlpha:   { get: function () { return _fillAlpha; } },
            _outlineW:    { get: function () { return _outlineW; } },
            _lastSPerImg: { get: function () { return _lastSPerImg; } },
            _hoverCss:    { get: function () { return _hoverCss; } },
            // ---- Inc-A: multi-class registry (read-only, for tests/harness) ----
            _classes:        { get: function () { return _classes; } },
            _activeClassId:  { get: function () { return _activeClassId; } },
            _silh:           { get: function () { return _silh; } },
            // ---- Inc-B: undo/redo state (read-only, for tests/harness) ----
            _undoStack:      { get: function () { return _undoStack; } },
            _redoStack:      { get: function () { return _redoStack; } },
            _undoDepth:      { get: function () { return opts.undoDepth || DEFAULTS.undoDepth; } },
            // ---- Inc-7: polygon draft state (read-only except the harness-only _polyDraft setter,
            // precedent: _refTI / _maskDirty) ----
            _polyDraft:      { get: function () { return _polyDraft; }, set: function (v) { _polyDraft = v; } },
            _polyDrawing:    { get: function () { return _polyDraft !== null; } },
            _polyLastCommit: { get: function () { return _polyLastCommit; } },
            _panning:        { get: function () { return _panning; } },
            _panPointerId:   { get: function () { return _panPointerId; } },
            _renderCount:    { get: function () { return _renderCount; } },
            __polyRuns:      { get: function () { return _polyRuns; } },
            // ---- Inc-8: edit records / selection / drag (READ-ONLY — every selection in tests is
            // made by a real click, so there is deliberately NO _selPoly setter) ----
            _polys:          { get: function () { return _polys; } },
            _selPoly:        { get: function () { return _selPoly; } },
            _dragVert:       { get: function () { return _dragVert; } },
            _selVert:        { get: function () { return _selVert; } },   // Inc-13: the OBJECT {polyId, idx} | null (GETTER ONLY)
            __runsDiff:      { get: function () { return _runsDiff; } },
            __ptInPoly:      { get: function () { return _ptInPoly; } },
            // ---- Inc-9: Alt-hold latch (GETTER ONLY — every latch in tests is made by a dispatched
            // Alt keydown; there is deliberately no setter) ----
            _altSavedTool:   { get: function () { return _altSavedTool; } },
            // ---- Inc-11: pure-geometry oracles (GETTERS ONLY; state-free, callable
            // pre-init — nothing here reads the mask or the records) ----
            __runsSubtract:     { get: function () { return function (dab, prot) { return _runsSubtract(dab, _mergeRunsToRows(prot)); }; } },
            __projectPointToSeg:{ get: function () { return _projectPointToSeg; } },
            __nearestEdge:      { get: function () { return _nearestEdge; } },
            // ---- Inc-15 (§11.1): tracer hooks (GETTERS ONLY). __trace and __recRuns are pure and
            // callable pre-init; __lastTraceStats is the diagnostics record, not state (I-28) ----
            __trace:            { get: function () { return _trace; } },
            __recRuns:          { get: function () { return _recRuns; } },
            __lastTraceStats:   { get: function () { return _lastTraceStats; } },
            __retraceStats:     { get: function () { return _retraceStats; } }   // Inc-16 (§5.7): {runs, ms, crop, grown} — diagnostics, not state
        });

        return instance;
    }

    var OSDAnnotator = { attach: attach, DEFAULTS: DEFAULTS };

    if (typeof module !== 'undefined' && module.exports) { module.exports = OSDAnnotator; }
    global.OSDAnnotator = OSDAnnotator;

})(typeof window !== 'undefined' ? window : this);
