// hyperpicasso.js — in-browser mosaic-PICASSO unmixing-matrix generator.
//
// Clean-room JS port of hyperpicasso/hyperpicasso/optimizer.py (NON-CLIPPING form).
// PURE module: no DOM / OpenSeadragon / network — so it drops into a Web Worker
// unchanged. The browser already has the FOV as typed arrays (readChannelRegion /
// __hpExtract); this replaces the Flask /picasso/generate round-trip.
//
// Parity is A/B-validated against the Python reference (tests/test_hyperpicasso_parity.mjs
// vs tests/fixtures/hyperpicasso_parity_A.json). The Python optimizer is retained as
// the parity oracle while this is hardened.
//
// Contract:
//   generateUnmixMatrix(fov, opts) -> { M: Float64Array(C*C) row-major, iters,
//                                       converged, changeHistory: number[] }
//     fov  : { data: Float64Array|number[]  (channels-last flat, idx=(y*W+x)*C+c),
//              H, W, C }
//     opts : { gstep, iterLimit, bins, tile, tileSelectRatio, alphaMin, alphaMax,
//              acceptMargin, reselectTiles, tol, onProgress(it,total) }
//
// NOTE (this commit): the CORE (normalize -> MI -> bounded-alpha search -> gradient
// loop -> compose) is ported and A/B-grounded on a single-tile fixture. Multi-tile
// SSIM tile-selection parity (skimage.structural_similarity windowing) is the next
// validation step — `_ssim`/`_selectLowSsimPixels` below mirror the Python but their
// multi-tile ranking is not yet diffed against skimage.

// ---- numpy-compatible auto-ranged bin index ----
function _binIndex(v, lo, hi, n) {
    if (hi <= lo) return n >> 1;            // constant axis: numpy expands range -> middle bin
    let i = Math.floor(((v - lo) / (hi - lo)) * n);
    if (i < 0) return 0;
    if (i >= n) return n - 1;               // value == max goes in the last (closed) bin
    return i;
}

// ---- normalized mutual information I(a;b)/max(H(a),H(b)) from a 2-D histogram ----
// Port of _normalized_mi (optimizer.py:64). Auto-ranged per call (a,b each).
function _normalizedMI(a, b, bins) {
    const N = a.length;
    if (N === 0) return 0.0;
    let aMin = Infinity, aMax = -Infinity, bMin = Infinity, bMax = -Infinity;
    for (let k = 0; k < N; k++) {
        const av = a[k], bv = b[k];
        if (av < aMin) aMin = av; if (av > aMax) aMax = av;
        if (bv < bMin) bMin = bv; if (bv > bMax) bMax = bv;
    }
    const H = new Float64Array(bins * bins);
    for (let k = 0; k < N; k++) {
        const ia = _binIndex(a[k], aMin, aMax, bins);
        const ib = _binIndex(b[k], bMin, bMax, bins);
        H[ia * bins + ib] += 1;
    }
    const total = N;                         // all points are in [min,max] -> sum(H)==N
    const pa = new Float64Array(bins);
    const pb = new Float64Array(bins);
    let hJoint = 0.0;
    for (let i = 0; i < bins; i++) {
        const row = i * bins;
        for (let j = 0; j < bins; j++) {
            const cnt = H[row + j];
            if (cnt > 0) {
                const p = cnt / total;
                pa[i] += p;
                pb[j] += p;
                hJoint -= p * Math.log(p);
            }
        }
    }
    let hA = 0.0, hB = 0.0;
    for (let i = 0; i < bins; i++) {
        if (pa[i] > 0) hA -= pa[i] * Math.log(pa[i]);
        if (pb[i] > 0) hB -= pb[i] * Math.log(pb[i]);
    }
    const denom = Math.max(hA, hB);
    if (denom <= 1e-12) return 0.0;
    return (hA + hB - hJoint) / denom;
}

// ---- bounded 1-D minimizer (golden-section) ----
// Faithful drop-in for scipy Powell on a single bounded scalar (Powell degenerates
// to a bracketed line search here). Returns argmin over [lo, hi].
function _minimizeScalar(f, lo, hi, tol, maxIter) {
    const invphi = (Math.sqrt(5) - 1) / 2;           // 0.618...
    let a = lo, b = hi;
    let c = b - invphi * (b - a);
    let d = a + invphi * (b - a);
    let fc = f(c), fd = f(d);
    let it = 0;
    while ((b - a) > tol && it < maxIter) {
        if (fc < fd) { b = d; d = c; fd = fc; c = b - invphi * (b - a); fc = f(c); }
        else { a = c; c = d; fc = fd; d = a + invphi * (b - a); fd = f(d); }
        it++;
    }
    // Best evaluated point in the final bracket.
    return fc < fd ? c : d;
}

// ---- SSIM (skimage.metrics.structural_similarity, uniform window) ----
// Port of utils.calculate_ssim usage (optimizer.py:111). Uniform (box) filter,
// odd win_size, per-patch data_range, boundary cropped by win//2, mean of map.
// (Used only for MULTI-tile selection; single-tile fixtures never rank on it.)
function _uniformFilter(src, h, w, win) {
    // Separable box filter, 'reflect' boundary (scipy.ndimage default).
    const rad = (win - 1) >> 1;
    const tmp = new Float64Array(h * w);
    const out = new Float64Array(h * w);
    const refl = (i, n) => { // scipy 'reflect' (d c b a | a b c d | d c b a)
        if (n === 1) return 0;
        let p = i;
        const period = 2 * n;
        p = ((p % period) + period) % period;
        return p < n ? p : period - 1 - p;
    };
    for (let y = 0; y < h; y++) {            // horizontal pass
        for (let x = 0; x < w; x++) {
            let s = 0;
            for (let k = -rad; k <= rad; k++) s += src[y * w + refl(x + k, w)];
            tmp[y * w + x] = s / win;
        }
    }
    for (let y = 0; y < h; y++) {            // vertical pass
        for (let x = 0; x < w; x++) {
            let s = 0;
            for (let k = -rad; k <= rad; k++) s += tmp[refl(y + k, h) * w + x];
            out[y * w + x] = s / win;
        }
    }
    return out;
}

function _ssim(aPatch, bPatch, h, w, dataRange, win) {
    const NP = win * win;
    const covNorm = NP / (NP - 1);                   // skimage unbiased covariance
    const ux = _uniformFilter(aPatch, h, w, win);
    const uy = _uniformFilter(bPatch, h, w, win);
    const axx = new Float64Array(h * w), ayy = new Float64Array(h * w), axy = new Float64Array(h * w);
    for (let i = 0; i < h * w; i++) { axx[i] = aPatch[i] * aPatch[i]; ayy[i] = bPatch[i] * bPatch[i]; axy[i] = aPatch[i] * bPatch[i]; }
    const uxx = _uniformFilter(axx, h, w, win);
    const uyy = _uniformFilter(ayy, h, w, win);
    const uxy = _uniformFilter(axy, h, w, win);
    const C1 = (0.01 * dataRange) ** 2, C2 = (0.03 * dataRange) ** 2;
    const rad = (win - 1) >> 1;
    let sum = 0; let cnt = 0;
    for (let y = rad; y < h - rad; y++) {            // crop border by win//2
        for (let x = rad; x < w - rad; x++) {
            const i = y * w + x;
            const vx = covNorm * (uxx[i] - ux[i] * ux[i]);
            const vy = covNorm * (uyy[i] - uy[i] * uy[i]);
            const vxy = covNorm * (uxy[i] - ux[i] * uy[i]);
            const A1 = 2 * ux[i] * uy[i] + C1, A2 = 2 * vxy + C2;
            const B1 = ux[i] * ux[i] + uy[i] * uy[i] + C1, B2 = vx + vy + C2;
            sum += (A1 * A2) / (B1 * B2);
            cnt++;
        }
    }
    return cnt > 0 ? sum / cnt : 1.0;
}

// ---- tile selection (low-SSIM pixels) — port of _select_low_ssim_pixels ----
function _percentile(sorted, q) {
    // numpy default 'linear' interpolation; sorted ascending.
    if (sorted.length === 1) return sorted[0];
    const pos = (q / 100) * (sorted.length - 1);
    const lo = Math.floor(pos), hi = Math.ceil(pos);
    if (lo === hi) return sorted[lo];
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

export function _selectLowSsimPixels(chA, chB, H, W, tile, ratio) {
    const n1 = Math.floor(H / tile), n2 = Math.floor(W / tile);
    if (n1 < 1 || n2 < 1) { const all = new Int32Array(H * W); for (let i = 0; i < all.length; i++) all[i] = i; return all; }
    // NOTE: no `n1*n2===1` short-circuit. Python's _select_low_ssim_pixels (optimizer.py:87)
    // scores/returns ONLY complete tiles — for a non-tile-multiple FOV (e.g. H=41,W=42,tile=40)
    // it returns the single 40×40 tile's 1600 pixels, NOT all H*W. The general path below
    // already assembles exactly the complete-tile ranges (r in ti*tile..(ti+1)*tile), so for the
    // single-tile case it returns precisely that one tile, matching Python. (For 40×40 tile=40
    // this is all 1600 pixels, unchanged.)
    const win = Math.min(7, tile % 2 === 1 ? tile : tile - 1);
    const scores = new Float64Array(n1 * n2);
    const pa = new Float64Array(tile * tile), pb = new Float64Array(tile * tile);
    for (let ti = 0; ti < n1; ti++) {
        for (let tj = 0; tj < n2; tj++) {
            let mn = Infinity, mx = -Infinity;
            for (let r = 0; r < tile; r++) {
                for (let cc = 0; cc < tile; cc++) {
                    const av = chA[(ti * tile + r) * W + (tj * tile + cc)];
                    const bv = chB[(ti * tile + r) * W + (tj * tile + cc)];
                    pa[r * tile + cc] = av; pb[r * tile + cc] = bv;
                    if (av < mn) mn = av; if (av > mx) mx = av;
                }
            }
            let dr = mx - mn; if (dr <= 0) dr = 1.0;
            scores[ti * n2 + tj] = _ssim(pa, pb, tile, tile, dr, win);
        }
    }
    const th = _percentile(Float64Array.from(scores).sort(), 100.0 * ratio);
    const chosen = [];
    for (let t = 0; t < scores.length; t++) if (scores[t] <= th) chosen.push(t);
    if (chosen.length === 0) { let am = 0; for (let t = 1; t < scores.length; t++) if (scores[t] < scores[am]) am = t; chosen.push(am); }
    const idx = [];
    for (const t of chosen) {
        const ti = Math.floor(t / n2), tj = t % n2;
        for (let r = ti * tile; r < (ti + 1) * tile; r++)
            for (let cc = tj * tile; cc < (tj + 1) * tile; cc++) idx.push(r * W + cc);
    }
    return Int32Array.from(idx);
}

function _std(arr) {
    const n = arr.length; if (n === 0) return 0;
    let m = 0; for (let i = 0; i < n; i++) m += arr[i]; m /= n;
    let v = 0; for (let i = 0; i < n; i++) { const d = arr[i] - m; v += d * d; } v /= n;   // ddof=0
    return Math.sqrt(v);
}

// ---- main optimizer — port of generate_unmix_matrix ----
export function generateUnmixMatrix(fov, opts = {}) {
    const H = fov.H, W = fov.W, C = fov.C;
    const src = fov.data;
    if (C < 2) throw new Error('need >= 2 channels; got C=' + C);
    const gstep = opts.gstep ?? 0.2;
    const iterLimit = opts.iterLimit ?? 10;
    const bins = opts.bins ?? 256;
    const tile = opts.tile ?? 40;
    const ratio = opts.tileSelectRatio ?? 0.01;
    const alphaMin = opts.alphaMin ?? -0.01;
    const alphaMax = opts.alphaMax ?? 0.6;
    const acceptMargin = opts.acceptMargin ?? 0.0;
    const reselectTiles = opts.reselectTiles ?? true;
    const onProgress = opts.onProgress || null;
    if (tile < 3) throw new Error('tile must be >= 3; got ' + tile);

    const N = H * W;
    // normalize: nan_to_num -> divide by single global peak (scale-invariant)
    const cur = new Float64Array(N * C);
    let peak = 0;
    for (let i = 0; i < N * C; i++) { let v = src[i]; if (!isFinite(v)) v = 0; cur[i] = v; if (v > peak) peak = v; }
    if (peak > 0) for (let i = 0; i < N * C; i++) cur[i] /= peak;
    const tol = (opts.tol !== undefined && opts.tol !== null) ? opts.tol : 0.002 / C;

    // M = identity (row-major)
    let M = new Float64Array(C * C); for (let i = 0; i < C; i++) M[i * C + i] = 1;
    let prevP = new Float64Array(C * C); for (let i = 0; i < C; i++) prevP[i * C + i] = 1;
    const changeHistory = [];
    let converged = false, iters = 0;
    let firstP = null;                       // iteration-1 P (A/B iter-1 gate); always returned

    const pairIdxCache = new Map();
    function indicesFor(i, j) {
        if (reselectTiles) return _selectLowSsimPixels(_channel(cur, i, N, C), _channel(cur, j, N, C), H, W, tile, ratio);
        const key = i < j ? i + ',' + j : j + ',' + i;
        if (pairIdxCache.has(key)) return pairIdxCache.get(key);
        const idx = _selectLowSsimPixels(_channel(cur, i, N, C), _channel(cur, j, N, C), H, W, tile, ratio);
        pairIdxCache.set(key, idx);
        return idx;
    }

    for (let it = 1; it <= iterLimit; it++) {
        iters = it;
        if (onProgress) onProgress(it, iterLimit);
        const P = new Float64Array(C * C); for (let i = 0; i < C; i++) P[i * C + i] = 1;

        for (let i = 0; i < C; i++) {
            for (let j = 0; j < C; j++) {
                if (i === j) continue;
                const idx = indicesFor(i, j);
                const m = idx.length;
                const xi = new Float64Array(m), xj = new Float64Array(m);
                for (let t = 0; t < m; t++) { const k = idx[t]; xi[t] = cur[k * C + i]; xj[t] = cur[k * C + j]; }
                if (_std(xi) < 1e-9 || _std(xj) < 1e-9) continue;   // degenerate pair -> no subtraction
                const resid = new Float64Array(m);
                const objective = (a) => { for (let t = 0; t < m; t++) resid[t] = xi[t] - a * xj[t]; return _normalizedMI(xj, resid, bins); };
                let alpha = _minimizeScalar(objective, alphaMin, alphaMax, 1e-5, 200);
                if (acceptMargin > 0.0) {
                    const mi0 = objective(0.0);
                    if (!(alpha > 1e-6 && objective(alpha) < mi0 * (1.0 - acceptMargin))) alpha = 0.0;
                }
                P[i * C + j] = -gstep * alpha;
            }
        }
        if (it === 1) firstP = Float64Array.from(P);   // capture before composition (iter-1 gate)

        // cur = cur_flat @ P^T  (NO clamp): new[k,i] = sum_j cur[k,j]*P[i,j]
        const next = new Float64Array(N * C);
        for (let k = 0; k < N; k++) {
            const base = k * C;
            for (let i = 0; i < C; i++) {
                let s = 0; const prow = i * C;
                for (let jj = 0; jj < C; jj++) s += cur[base + jj] * P[prow + jj];
                next[base + i] = s;
            }
        }
        cur.set(next);
        // M = P @ M
        const Mn = new Float64Array(C * C);
        for (let i = 0; i < C; i++) for (let j = 0; j < C; j++) { let s = 0; for (let k = 0; k < C; k++) s += P[i * C + k] * M[k * C + j]; Mn[i * C + j] = s; }
        M = Mn;

        let change = 0; for (let q = 0; q < C * C; q++) change += Math.abs(P[q] - prevP[q]);
        change /= (C * C - C);
        changeHistory.push(change);
        prevP = P;
        if (it > 1 && change < tol) { converged = true; break; }
    }
    return { M, iters, converged, changeHistory, firstP };
}

// view of channel c as an (H*W) Float64Array (row-major), gathered from channels-last
function _channel(flat, c, N, C) {
    const out = new Float64Array(N);
    for (let k = 0; k < N; k++) out[k] = flat[k * C + c];
    return out;
}

// ---- decode/route helpers (port of _core.py _run; shared by tests + the worker) ----
// Banker's rounding (round-half-to-even) — matches numpy np.round, NOT Math.round
// (which is round-half-up). Only differs at exact half-quantum; inputs here are >= 0.
export function _roundHalfEven(x) {
    const f = Math.floor(x), d = x - f;
    if (d < 0.5) return f;
    if (d > 0.5) return f + 1;
    return (f % 2 === 0) ? f : f + 1;
}

// Route 1 ('picasso'): channels-last stack of the ENABLED raw planes (0..255 floats).
// planes: array of 16 typed arrays (H*W uint8 each, row-major). N = enabled count.
export function buildRoute1Stack(planes, W, H, enabledMask) {
    const enabled = [];
    for (let i = 0; i < 16; i++) if (enabledMask[i]) enabled.push(i);
    const C = enabled.length, N = W * H;
    const data = new Float64Array(N * C);
    for (let k = 0; k < N; k++) for (let ci = 0; ci < C; ci++) data[k * C + ci] = planes[enabled[ci]][k];
    return { data, H, W, C };
}

// Route 2 ('hyperpicasso'): apply the linear-unmix matrix then the uint8-FBO contract.
// lin = (raw/255) @ M_unmix[:, :numOutputs] (M_unmix is 16x8 ROW-MAJOR); clip [0,1];
// quantize round(lin*255)/255 (banker's). N = numOutputs. (np.flipud is dropped — the
// optimizer is flip-invariant; documented in _core.py.)
export function buildRoute2Stack(planes, W, H, unmixMatrix, numOutputs) {
    const N = W * H, C = numOutputs;
    const data = new Float64Array(N * C);
    for (let k = 0; k < N; k++) {
        for (let o = 0; o < C; o++) {
            let s = 0;
            for (let i = 0; i < 16; i++) s += (planes[i][k] / 255) * unmixMatrix[i * 8 + o];
            if (s < 0) s = 0; else if (s > 1) s = 1;
            data[k * C + o] = _roundHalfEven(s * 255) / 255;
        }
    }
    return { data, H, W, C };
}

// flat row-major Float64Array(N*N) -> nested number[][] (handleResult consumes matrix[r][c])
export function reshapeFlatToNested(M, N) {
    const out = [];
    for (let i = 0; i < N; i++) { const row = []; for (let j = 0; j < N; j++) row.push(M[i * N + j]); out.push(row); }
    return out;
}

export default { generateUnmixMatrix, buildRoute1Stack, buildRoute2Stack, reshapeFlatToNested, _roundHalfEven, _selectLowSsimPixels };
