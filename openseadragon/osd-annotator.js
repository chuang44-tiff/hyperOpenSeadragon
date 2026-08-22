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
        var _silh = {};                // id -> {canvas, ctx}  LAZY render cache; _silh[id] ≡ {_mask px == id}
        var _scratch = null, _sctx = null;   // ONE reusable _view-sized per-class composite scratch

        // ---- Inc-B: stroke-level undo/redo state (non-frozen) ----
        var _undoStack = [];           // [{x,y,w,h, before:ImageData, after:ImageData}]  bbox in MASK px
        var _redoStack = [];
        var _undoPre   = null;         // single reusable full-mask <canvas> holding pre-stroke pixels
        var _upctx     = null;         // its 2d ctx (imageSmoothingEnabled=false)
        var _strokeBBox = null;        // {x0,y0,x1,y1} inclusive, accumulated during the current stroke

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
        var _polys = [];               // [{id,classId,pts:[{x,y}]}] — array order = z-order (last = topmost)
        var _polySeq = 0;              // monotonic id source (++_polySeq per append); reset only by destroy
        var _selPoly = null;           // null | record reference into _polys — the selected polygon
        var _dragVert = null;          // null | {poly, idx, pts} — pts is a WORKING COPY (no mask write)
        var _polyScratch = null, _psctx = null;   // reusable full-mask binary scratch (silhouette-clipped erase)
        // ---- Inc-9 (R3): Alt-hold temporary-eraser latch. Non-null ⇔ an Alt hold is latched and the
        // saved tool must be restored on keyup / window blur. Under the Inc-9 mode split the latch can
        // only engage from Brush mode, so the ONLY value ever stored is 'brush'; it is kept as a
        // SAVED-TOOL variable (not a boolean) so the restore reads setTool(_altSavedTool) and the
        // mechanism survives any future mode being added.
        var _altSavedTool = null;                 // null | 'brush'

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
                    var cls = _classes[ci];
                    var se = _silh[cls.id];
                    if (!se || !se.ctx) { continue; }            // no pixels / no ctx → skip
                    _sctx.setTransform(1, 0, 0, 1, 0, 0);
                    _sctx.clearRect(0, 0, _scratch.width, _scratch.height);
                    _sctx.imageSmoothingEnabled = false;
                    _sctx.globalAlpha = 1;
                    _sctx.globalCompositeOperation = 'source-over';
                    // (1) 12-dir dilation: build the outward rim in DEVICE space
                    for (var di = 0; di < NDIR; di++) {
                        var ang = (Math.PI * 2 * di) / NDIR;
                        _sctx.setTransform(A, B, C, D, E0 + Math.cos(ang) * rimDev, F0 + Math.sin(ang) * rimDev);
                        _sctx.drawImage(se.canvas, 0, 0);        // dilate silhouette in DEVICE space
                    }
                    // (2) recolor dilated silhouette → cls.color
                    _sctx.setTransform(1, 0, 0, 1, 0, 0);        // identity: tint whole canvas
                    _sctx.globalCompositeOperation = 'source-atop';
                    _sctx.fillStyle = cls.color;
                    _sctx.fillRect(0, 0, _scratch.width, _scratch.height);
                    // (3) punch out interior → outward rim only
                    _sctx.setTransform(A, B, C, D, E0, F0);      // base (un-offset) transform
                    _sctx.globalCompositeOperation = 'destination-out';
                    _sctx.drawImage(se.canvas, 0, 0);
                    // (4) translucent fill BEHIND the rim via destination-over
                    _sctx.globalCompositeOperation = 'destination-over';
                    _sctx.globalAlpha = _fillAlpha;
                    _sctx.drawImage(se.canvas, 0, 0);            // shape @ _fillAlpha, behind rim (base transform)
                    _sctx.globalAlpha = 1;
                    // (5) tint fill → cls.color
                    _sctx.setTransform(1, 0, 0, 1, 0, 0);        // identity: tint whole canvas
                    _sctx.globalCompositeOperation = 'source-atop';
                    _sctx.fillStyle = cls.color;
                    _sctx.fillRect(0, 0, _scratch.width, _scratch.height);
                    // (6) restore scratch defaults
                    _sctx.globalCompositeOperation = 'source-over';
                    _sctx.globalAlpha = 1;
                    // blit this class onto _view (classes are disjoint in mask space → disjoint here)
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
                // (b) handles: one filled square per vertex (the DRAGGED one in POLY_SEL_COLOR)
                var hSide = POLY_HANDLE_PX * DPRs;
                for (var qhi = 0; qhi < sdev.length; qhi++) {
                    var hx0 = sdev[qhi].x - hSide / 2, hy0 = sdev[qhi].y - hSide / 2;
                    _ctx.fillStyle = (_dragVert && qhi === _dragVert.idx) ? POLY_SEL_COLOR : '#fff';
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
            // ---- Inc-7: drop the polygon draft INLINE (not via _dropDraft — that would schedule
            // an rAF this teardown has already cancelled) ----
            _polyDraft = null; _polyDownCss = null; _polyLastCss = null;
            _polyCurMask = null; _polyLastClick = null; _polyLastCommit = false;
            // ---- Inc-8: drop the edit records, selection, drag and the reshape scratch INLINE
            // (not via _dropDraft — same rAF reason as above) ----
            _polys = []; _polySeq = 0; _selPoly = null; _dragVert = null;
            _polyScratch = null; _psctx = null;
            _altSavedTool = null;                            // ---- Inc-9: drop any latched Alt hold ----
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
            var k;
            if (_tool === 'eraser') {
                // erase the mask AND every allocated silhouette over the runs
                _fillRuns(_mctx, runs, 'destination-out', '#fff');
                for (k in _silh) {
                    if (_silh.hasOwnProperty(k) && _silh[k] && _silh[k].ctx) {
                        _fillRuns(_silh[k].ctx, runs, 'destination-out', '#fff');
                    }
                }
            } else {
                // (a) write the class id into the indexed mask (alpha 1 overwrites any prior class)
                _fillRuns(_mctx, runs, 'source-over', 'rgba(' + id + ',0,0,1)');
                // (b) add these pixels to the active class silhouette (null-guarded)
                var se = _ensureSilh(id);
                if (se) { _fillRuns(se.ctx, runs, 'source-over', '#fff'); }
                // (c) remove these pixels from EVERY OTHER allocated silhouette (overwrite semantics)
                for (k in _silh) {
                    if (_silh.hasOwnProperty(k) && _silh[k] && _silh[k].ctx && String(k) !== String(id)) {
                        _fillRuns(_silh[k].ctx, runs, 'destination-out', '#fff');
                    }
                }
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
        function _polyRuns(pts) {
            var out = [];
            if (!pts) { return out; }
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
            var j1 = Math.min(_maskH - 1, Math.ceil(maxY));   // rows outside the mask are never visited
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
                    var i1 = Math.min(_maskW - 1, Math.ceil(cr[c + 1].x - 0.5) - 1);
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

        // ---- Inc-7: commit a closed polygon into the indexed mask (mirrors _stampDisc's three writes) ----
        // _stampDisc stays byte-identical (frozen); this reuses the same non-frozen _fillRuns helper.
        function _polyCommit(runs, cid) {
            if (!runs || runs.length === 0) { return; }       // makes the _maskDirty gate explicit
            // Inc-8: `cid` is OPTIONAL — every Inc-7 caller passes one argument and behaves as
            // before (class taken at CLOSE time); _reshapeCommit passes the RECORD's classId so a
            // reshape always writes the polygon's own class, never the active one.
            var id = (typeof cid === 'number') ? cid : _activeClassId;
            var k;
            // (a) write the class id into the indexed mask (alpha 1 overwrites any prior class)
            _fillRuns(_mctx, runs, 'source-over', 'rgba(' + id + ',0,0,1)');
            // (b) add these pixels to the active class silhouette (null-guarded)
            var se = _ensureSilh(id);
            if (se) { _fillRuns(se.ctx, runs, 'source-over', '#fff'); }
            // (c) remove these pixels from EVERY OTHER allocated silhouette (overwrite semantics)
            for (k in _silh) {
                if (_silh.hasOwnProperty(k) && _silh[k] && _silh[k].ctx && String(k) !== String(id)) {
                    _fillRuns(_silh[k].ctx, runs, 'destination-out', '#fff');
                }
            }
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
            if (_polyDraft !== null) { _polyDraft = null; _scheduleRender(); }
            // ---- Inc-8: this is the LOAD-BEARING abort for a hijacked vertex drag. Every teardown
            // path (setTool on change, setActive(false), clear, cancelPolygon/Esc) already calls
            // _dropDraft BEFORE _endStroke, so nulling _dragVert here means the _endStroke prologue
            // sees nothing to commit — no mask write, no history entry. _polys is NOT touched here:
            // Esc must deselect, never delete committed polygons.
            if (_selPoly !== null || _dragVert !== null) { _selPoly = null; _dragVert = null; _scheduleRender(); }
            _polyLastClick = null; _polyDownCss = null; _polyLastCss = null; _polyCurMask = null;
        }

        // Closes the open draft: rasterize + commit as ONE undo entry. Returns true iff pixels
        // were written. Degeneracy is simply "zero runs" — there is no min-area epsilon.
        function _closeDraft() {
            var d = _polyDraft;
            _dropDraft();                                    // draft nulled FIRST (re-entrancy safety)
            if (!d || d.pts.length < 3) { _polyLastCommit = false; return false; }
            var runs = _polyRuns(d.pts);
            if (runs.length === 0) { _polyLastCommit = false; return false; }   // collinear/degenerate/off-mask
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
            var O = _polyRuns(rec.pts);
            var N = deleting ? [] : _polyRuns(newPts);        // may be [] if the drag made the ring degenerate
            var d = _runsDiff(O, N);
            // (2) O △ N empty at the RUNS level → vertices follow the drop point, NO history entry,
            // _maskDirty untouched (a history entry is impossible with a null bbox anyway).
            // Inc-9: a ZERO-FOOTPRINT delete lands here — the record is removed with NO undo entry
            // (an entry is impossible with a null bbox, and the action wrote nothing).
            if (d.fill.length === 0 && d.erase.length === 0) {
                if (deleting) { _removePolyRec(rec); return; }
                // Inc-10 (§5.9): the vertices moved but no pixel did. This still MUTATES the record,
                // and any record mutation invalidates redo — otherwise a stale redo entry (e.g. a
                // deletion the user just undid) stays live and Ctrl+Shift+Z resurrects it.
                // The _ptsEqual guard is load-bearing: a zero-movement handle grab-release reaches
                // this branch on EVERY click, and must leave the redo chain untouched.
                if (!_ptsEqual(rec.pts, newPts)) {
                    rec.pts = _copyPts(newPts);
                    _redoStack.length = 0;
                    _fireHistory();                            // host redo button greys immediately
                }
                return;
            }
            _histBegin();
            // (4) erase O \ N, CLIPPED to pixels this class still owns (readback-free, via the
            // silhouette). Other classes' silhouettes need no touch: a pixel owned by another class
            // never enters the scratch, and a pixel owned by this one is already absent from theirs.
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
                    _mctx.globalCompositeOperation = 'destination-out';
                    _mctx.drawImage(_polyScratch, 0, 0);
                    _mctx.globalCompositeOperation = 'source-over';
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
            _histCommit({ polyId: rec.id, classId: rec.classId, ptsBefore: _copyPts(rec.pts), ptsAfter: deleting ? null : _copyPts(newPts) });
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
            if (_selPoly === rec) { _selPoly = null; _dragVert = null; }
            _scheduleRender();
        }

        // ---- Inc-B: stroke-level undo/redo helpers (non-frozen) ----
        // Class-op undo (add/remove/rename/recolor) is OUT OF SCOPE for Inc-B; undo scope is strokes only.
        function _histBegin() {
            if (!_mask) { return; }
            if (!_undoPre || _undoPre.width !== _maskW || _undoPre.height !== _maskH) {   // [AUDIT 5] re-alloc if mask size changed
                _undoPre = document.createElement('canvas');
                _undoPre.width = _maskW; _undoPre.height = _maskH;
                _upctx = _undoPre.getContext('2d');
                if (_upctx) { _upctx.imageSmoothingEnabled = false; }
            }
            if (_upctx) {
                _upctx.clearRect(0, 0, _maskW, _maskH);
                _upctx.drawImage(_mask, 0, 0);   // GPU blit, no readback
            }
            _strokeBBox = null;
        }

        // Inc-8: `tag` is OPTIONAL — {polyId, classId, ptsBefore, ptsAfter}. Entries WITHOUT it are
        // byte-for-byte the Inc-B shape and behave exactly as before. There is no `kind`
        // discriminator and no second restore path: every entry still carries a raster bbox pair;
        // the vertex fields are a FIXUP applied after the raster restore (see _applyPolyFixup).
        function _histCommit(tag) {
            if (!_strokeBBox || !_upctx || !_mctx) { return; }   // no paint this stroke (pan/no-op)
            var x = _strokeBBox.x0, y = _strokeBBox.y0;
            var w = _strokeBBox.x1 - _strokeBBox.x0 + 1;
            var h = _strokeBBox.y1 - _strokeBBox.y0 + 1;
            _strokeBBox = null;                                  // [AUDIT 13] null FIRST — makes re-entry a no-op
            if (w <= 0 || h <= 0) { return; }
            var before = _upctx.getImageData(x, y, w, h);        // pre-stroke (blit shadow)
            var after  = _mctx.getImageData(x, y, w, h);         // post-stroke
            var ent = { x: x, y: y, w: w, h: h, before: before, after: after };
            _undoStack.push(ent);
            if (tag) { ent.polyId = tag.polyId; ent.classId = tag.classId; ent.ptsBefore = tag.ptsBefore; ent.ptsAfter = tag.ptsAfter; }
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
        // schedules its own render (undo/redo call _render() themselves). ----
        function _applyPolyFixup(polyId, classId, pts) {
            var rec = null, at = -1, i;
            for (i = 0; i < _polys.length; i++) {
                if (_polys[i].id === polyId) { rec = _polys[i]; at = i; break; }
            }
            if (pts === null || pts === undefined) {              // undo of a CREATION → drop the record
                if (rec) {
                    _polys.splice(at, 1);
                    if (_selPoly === rec) { _selPoly = null; _dragVert = null; }
                }
                return;
            }
            if (rec) { rec.pts = _copyPts(pts); return; }
            // not found + pts present: redo of a creation (or a record dropped by loadPolygons).
            // Re-added at the TOP of the z-order; skipped silently if its class no longer exists
            // (the pixels are still restored — only the edit affordance is missing).
            if (polyId > _polySeq) { _polySeq = polyId; }
            if (_classById(classId)) { _polys.push({ id: polyId, classId: classId, pts: _copyPts(pts) }); }
        }

        function undo() {
            if (_destroyed || !_mask) { if (!_mask && !_destroyed) { _warnOnce('no-image', 'OSDAnnotator: no image loaded yet'); } return; }
            if (_painting) { return; }                            // never undo mid-stroke
            if (_polyDraft || _dragVert) { return; }              // Inc-7/8: no-op mid-draft or mid-reshape-drag
            if (!_undoStack.length) { return; }
            var e = _undoStack.pop();
            _mctx.putImageData(e.before, e.x, e.y);
            _rebuildSilh(e.x, e.y, e.w, e.h);
            if (e.polyId !== undefined) { _applyPolyFixup(e.polyId, e.classId, e.ptsBefore); }
            _redoStack.push(e);
            _maskDirty = true;                                    // isEmpty stays coarse-true latch [AUDIT 10]
            _fireHistory();
            _render();
        }

        function redo() {
            if (_destroyed || !_mask) { if (!_mask && !_destroyed) { _warnOnce('no-image', 'OSDAnnotator: no image loaded yet'); } return; }
            if (_painting) { return; }
            if (_polyDraft || _dragVert) { return; }              // Inc-7/8: no-op mid-draft or mid-reshape-drag
            if (!_redoStack.length) { return; }
            var e = _redoStack.pop();
            _mctx.putImageData(e.after, e.x, e.y);
            _rebuildSilh(e.x, e.y, e.w, e.h);
            if (e.polyId !== undefined) { _applyPolyFixup(e.polyId, e.classId, e.ptsAfter); }
            _undoStack.push(e);
            _maskDirty = true;
            _fireHistory();
            _render();
        }

        function canUndo() { return _undoStack.length > 0; }
        function canRedo() { return _redoStack.length > 0; }
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
                        _view.setPointerCapture(e.pointerId);     // the EXISTING latch lines, reused
                        _activePointerId = e.pointerId;
                        _painting = true;
                        _dragVert = { poly: _selPoly, idx: bestK, pts: _copyPts(_selPoly.pts) };
                        _scheduleRender();
                        return;                                   // no vertex placed, no draft
                    }
                }
                // (2) a DRAFT always wins -> fall through to (iii)(iv) below unchanged.
                if (!_polyDraft) {
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
                    if (hitp) { _selPoly = hitp; _scheduleRender(); return; }
                    // (4) otherwise -> opening a draft DESELECTS (invariant I1), then (iii)(iv).
                    _selPoly = null;
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
            if (key === 'Escape') { if (_polyDraft || _selPoly) { e.preventDefault(); cancelPolygon(); } return; }
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
            if (key === 'Backspace' || key === 'Delete') {         // R4 (§7)
                if (_selPoly && !_painting) {
                    _reshapeCommit(_selPoly, null);                // erase footprint + remove record + one undo entry
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
            // zero this class's pixels from the indexed mask via its silhouette (readback-free).
            // NOTE: does NOT touch _maskDirty — isEmpty stays the coarse first-paint latch (CODEX-2).
            if (_silh[id] && _silh[id].ctx) {
                _mctx.globalCompositeOperation = 'destination-out';
                _mctx.drawImage(_silh[id].canvas, 0, 0);
                _mctx.globalCompositeOperation = 'source-over';
            }
            if (_silh[id]) { delete _silh[id]; }         // free the silhouette cache
            // ---- Inc-8: this class's edit records go with its pixels. Nothing holds the _polys
            // ARRAY object (only records, and the getter reads the variable), so a filter is safe.
            // The unconditional nulling costs nothing — a drag cannot be live during a host click
            // (the pointer is captured on _view) — and needs no reasoning about which record went.
            _polys = _polys.filter(function (p) { return p.classId !== id; });
            _selPoly = null; _dragVert = null;
            for (var i = 0; i < _classes.length; i++) {
                if (_classes[i].id === id) { _classes.splice(i, 1); break; }
            }
            if (_activeClassId === id) { _activeClassId = _classes[0].id; }
            // Inc-B [AUDIT 3]: removeClass mutates mask pixels (destination-out above) →
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
            _polys.length = 0; _selPoly = null; _dragVert = null;
            // (9) mark dirty + FLUSH history
            _maskDirty = true;
            _undoStack.length = 0; _redoStack.length = 0; _fireHistory();
            // (10) render
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
                out.push({ classId: p.classId, pts: pts });
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
            _polys.length = 0; _selPoly = null; _dragVert = null;   // ALWAYS first
            if (!Array.isArray(arr)) { _scheduleRender(); return false; }
            var i, k, r, ent;
            for (i = 0; i < arr.length; i++) {                      // pass 1: validate EVERYTHING
                r = arr[i];
                if (!r || typeof r !== 'object') { _scheduleRender(); return false; }
                if (typeof r.classId !== 'number' || !isFinite(r.classId)
                    || Math.floor(r.classId) !== r.classId
                    || r.classId < 1 || r.classId > 255) { _scheduleRender(); return false; }
                if (!Array.isArray(r.pts) || r.pts.length < 3) { _scheduleRender(); return false; }
                for (k = 0; k < r.pts.length; k++) {
                    ent = r.pts[k];
                    if (!Array.isArray(ent) || ent.length !== 2
                        || typeof ent[0] !== 'number' || !isFinite(ent[0])
                        || typeof ent[1] !== 'number' || !isFinite(ent[1])) { _scheduleRender(); return false; }
                }
            }
            for (i = 0; i < arr.length; i++) {                      // pass 2: adopt
                r = arr[i];
                if (!_classById(r.classId)) { continue; }           // unreachable vectors — skipped silently
                var pts = [];
                for (k = 0; k < r.pts.length; k++) { pts.push({ x: r.pts[k][0], y: r.pts[k][1] }); }
                _polys.push({ id: ++_polySeq, classId: r.classId, pts: pts });
            }
            _scheduleRender();
            return true;
        }

        // GETTER (no guard/warn). ONE full-mask getImageData per call — Export-click only; never
        // called from a render or status path.
        function getClassPixelCount(id) {
            if (_destroyed || !_mask || !_mctx) { return 0; }
            if (typeof id !== 'number' || !isFinite(id) || Math.floor(id) !== id) { return 0; }
            var d = _mctx.getImageData(0, 0, _maskW, _maskH).data;
            var n = 0;
            for (var i = 0; i < d.length; i += 4) { if (d[i] === id && d[i + 3] > 0) { n++; } }
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
                ectx.drawImage(_mask, 0, 0, outW, outH);
                var ed = ectx.getImageData(0, 0, outW, outH);
                var d = ed.data;
                var activeOnly = !!(opts2 && opts2.activeOnly);
                var activeId = _activeClassId;   // captured once; nearest-scaled ids are exact
                for (var i = 0; i < d.length; i += 4) {
                    if (activeOnly) {
                        // per-class BINARY mask (Export): white where the ACTIVE class is
                        // painted (R === active id), black everywhere else. Fully opaque.
                        var on = (d[i] === activeId) ? 255 : 0;
                        d[i] = on; d[i + 1] = on; d[i + 2] = on; d[i + 3] = 255;
                    } else {
                        // indexed label mask (Save): R = class id (0=bg,1..N), OPAQUE, G=B=0.
                        d[i + 1] = 0; d[i + 2] = 0; d[i + 3] = 255;   // R (d[i]) LEFT AS-IS = class id
                    }
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
            __runsDiff:      { get: function () { return _runsDiff; } },
            __ptInPoly:      { get: function () { return _ptInPoly; } },
            // ---- Inc-9: Alt-hold latch (GETTER ONLY — every latch in tests is made by a dispatched
            // Alt keydown; there is deliberately no setter) ----
            _altSavedTool:   { get: function () { return _altSavedTool; } }
        });

        return instance;
    }

    var OSDAnnotator = { attach: attach, DEFAULTS: DEFAULTS };

    if (typeof module !== 'undefined' && module.exports) { module.exports = OSDAnnotator; }
    global.OSDAnnotator = OSDAnnotator;

})(typeof window !== 'undefined' ? window : this);
