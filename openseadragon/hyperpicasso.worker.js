// hyperpicasso.worker.js — ES-module Web Worker: no-server replacement for POST
// /picasso/generate. Runs the PICASSO/HYPERPICASSO unmix-matrix optimizer off the
// main thread. Browser-only (uses self/postMessage); NOT node-runnable. The pure
// helpers it calls live in hyperpicasso.js and are unit-tested separately.
//
// Message IN (from main thread):
//   { route: 'picasso' | 'hyperpicasso',
//     width, height,
//     planeBuffers: ArrayBuffer[16]  // one de-interleaved uint8 plane each (H*W bytes)
//     enabledMask: number[16],       // route 'picasso' only
//     unmixMatrix: number[] | null,  // route 'hyperpicasso' only (16x8 row-major)
//     numOutputs:  number  | null }  // route 'hyperpicasso' only
//   planeBuffers SHOULD be transferred (postMessage(msg, planeBuffers)) — we view
//   them zero-copy, so the main thread must not reuse them after sending.
//
// Message OUT:
//   { type:'progress', it, total }                  // per optimizer iteration
//   { type:'result',   N, matrix:number[][] }       // BYTE-IDENTICAL to the Flask
//                                                   //   {N, matrix} handleResult reads
//   { type:'error',    error:string }

import {
    generateUnmixMatrix,
    buildRoute1Stack,
    buildRoute2Stack,
    reshapeFlatToNested,
} from './hyperpicasso.js';

self.onmessage = (e) => {
    try {
        const { route, width, height, planeBuffers, enabledMask, unmixMatrix, numOutputs } = e.data;

        // ---- Validate the message up front. Bad inputs used to silently produce a
        //      plausible-but-wrong matrix; fail loudly with a single terminal {type:'error'}.
        const isPosInt = (v) => Number.isInteger(v) && v > 0;
        if (!isPosInt(width) || !isPosInt(height)) {
            self.postMessage({ type: 'error', error: 'width and height must be positive integers' });
            return;
        }
        if (!Array.isArray(planeBuffers) || planeBuffers.length !== 16) {
            self.postMessage({ type: 'error', error: 'expected 16 planes of width*height bytes' });
            return;
        }
        const planeBytes = width * height;
        for (let i = 0; i < 16; i++) {
            const b = planeBuffers[i];
            if (!(b instanceof ArrayBuffer) || b.byteLength !== planeBytes) {
                self.postMessage({ type: 'error', error: 'expected 16 planes of width*height bytes' });
                return;
            }
        }
        if (route === 'hyperpicasso') {
            if (!Number.isInteger(numOutputs) || numOutputs < 2 || numOutputs > 8) {
                self.postMessage({ type: 'error', error: 'numOutputs must be an integer in [2,8]' });
                return;
            }
            if (!Array.isArray(unmixMatrix) || unmixMatrix.length !== 128 || !unmixMatrix.every((v) => Number.isFinite(v))) {
                self.postMessage({ type: 'error', error: 'unmixMatrix must be an array of exactly 128 finite numbers' });
                return;
            }
        } else if (route === 'picasso') {
            if (!Array.isArray(enabledMask) || enabledMask.length !== 16) {
                self.postMessage({ type: 'error', error: 'enabledMask must be a 16-length array' });
                return;
            }
            let enabledCount = 0;
            for (let i = 0; i < 16; i++) if (enabledMask[i]) enabledCount++;
            if (enabledCount < 2) {
                self.postMessage({ type: 'error', error: 'enabledMask must enable at least 2 channels' });
                return;
            }
        } else {
            self.postMessage({ type: 'error', error: "route must be 'picasso' or 'hyperpicasso'" });
            return;
        }

        // Zero-copy views over the (transferred) plane buffers — 16 uint8 planes.
        const planes = planeBuffers.map((b) => new Uint8Array(b));

        // Decode the raw planes into the channels-last stack the optimizer expects.
        const stack = (route === 'hyperpicasso')
            ? buildRoute2Stack(planes, width, height, unmixMatrix, numOutputs)
            : buildRoute1Stack(planes, width, height, enabledMask);

        // Run the optimizer, streaming progress back to the main thread.
        const res = generateUnmixMatrix(stack, {
            onProgress: (it, total) => self.postMessage({ type: 'progress', it, total }),
        });

        // N = channel count of the decoded stack == matrix dimension.
        const N = stack.C;

        // Finite-check every matrix entry — a non-finite cell would silently corrupt
        // the apply path, so fail loudly instead.
        for (let q = 0; q < res.M.length; q++) {
            if (!isFinite(res.M[q])) {
                self.postMessage({ type: 'error', error: 'optimizer produced non-finite matrix' });
                return;
            }
        }

        const matrix = reshapeFlatToNested(res.M, N);
        self.postMessage({ type: 'result', N, matrix });
    } catch (err) {
        self.postMessage({ type: 'error', error: String((err && err.message) || err) });
    }
};
