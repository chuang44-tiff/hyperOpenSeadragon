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

    var DEFAULTS = { maxMaskEdge: 4096, brushSize: 25, tool: 'brush', active: false, maxClasses: 16, undoDepth: 20 };

    // ---- Inc-A: multi-class registry constants ----
    var MAX_CLASSES = 16;
    var PALETTE = ['#4a7c8a', '#c0504d', '#9bbb59', '#8064a2', '#4bacc6', '#f79646',
                   '#2c4d75', '#a5a5a5', '#d99694', '#7f6084', '#4f81bd', '#c3d69b'];

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
            // ---- R3: brush ring cursor (identity transform, screen px) ----
            // SCREEN-SPACE brush: ring radius is CONSTANT on screen (independent of zoom).
            // _brushImg is a CSS-px diameter; the painted image footprint scales with zoom
            // inside _getRPx (screen px ÷ _lastSPerImg), NOT here. Rendered as a bold,
            // high-contrast cursor (dark halo under a bright ring) so it reads on any
            // background and makes brush-size changes obvious.
            if (_active && _hoverCss && !_frozen) {
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

        // ---- E3: pointer FSM helpers ----
        function _ptFromEvent(e) {
            var r = _view.getBoundingClientRect();   // LIVE per event
            return _pointerToMask(e.clientX - r.left, e.clientY - r.top);
        }

        function _endStroke() {
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

        function _histCommit() {
            if (!_strokeBBox || !_upctx || !_mctx) { return; }   // no paint this stroke (pan/no-op)
            var x = _strokeBBox.x0, y = _strokeBBox.y0;
            var w = _strokeBBox.x1 - _strokeBBox.x0 + 1;
            var h = _strokeBBox.y1 - _strokeBBox.y0 + 1;
            _strokeBBox = null;                                  // [AUDIT 13] null FIRST — makes re-entry a no-op
            if (w <= 0 || h <= 0) { return; }
            var before = _upctx.getImageData(x, y, w, h);        // pre-stroke (blit shadow)
            var after  = _mctx.getImageData(x, y, w, h);         // post-stroke
            _undoStack.push({ x: x, y: y, w: w, h: h, before: before, after: after });
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

        function undo() {
            if (_destroyed || !_mask) { if (!_mask && !_destroyed) { _warnOnce('no-image', 'OSDAnnotator: no image loaded yet'); } return; }
            if (_painting) { return; }                            // never undo mid-stroke
            if (!_undoStack.length) { return; }
            var e = _undoStack.pop();
            _mctx.putImageData(e.before, e.x, e.y);
            _rebuildSilh(e.x, e.y, e.w, e.h);
            _redoStack.push(e);
            _maskDirty = true;                                    // isEmpty stays coarse-true latch [AUDIT 10]
            _fireHistory();
            _render();
        }

        function redo() {
            if (_destroyed || !_mask) { if (!_mask && !_destroyed) { _warnOnce('no-image', 'OSDAnnotator: no image loaded yet'); } return; }
            if (_painting) { return; }
            if (!_redoStack.length) { return; }
            var e = _redoStack.pop();
            _mctx.putImageData(e.after, e.x, e.y);
            _rebuildSilh(e.x, e.y, e.w, e.h);
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

        // ---- Inc-B: keyboard undo/redo (annotator-scoped = _active-gated) ----
        function _onKeyDown(e) {
            if (_destroyed || !_active) { return; }               // [AUDIT 7] a disarmed annotator never hijacks host Ctrl+Z
            var t = e.target;                                      // typing guard: let native text-undo win in inputs
            if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) { return; }
            if (!(e.ctrlKey || e.metaKey)) { return; }
            var k = (e.key || '').toLowerCase();
            if (k === 'z' && !e.shiftKey) { e.preventDefault(); undo(); }
            else if ((k === 'z' && e.shiftKey) || k === 'y') { e.preventDefault(); redo(); }
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
            if (t === 'brush' || t === 'eraser') { _tool = t; }
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
            _undoDepth:      { get: function () { return opts.undoDepth || DEFAULTS.undoDepth; } }
        });

        return instance;
    }

    var OSDAnnotator = { attach: attach, DEFAULTS: DEFAULTS };

    if (typeof module !== 'undefined' && module.exports) { module.exports = OSDAnnotator; }
    global.OSDAnnotator = OSDAnnotator;

})(typeof window !== 'undefined' ? window : this);
