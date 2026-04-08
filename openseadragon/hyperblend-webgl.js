// =========================================================================
//  HYPERBLEND WEBGL DRAWER — MODULE OVERVIEW
//  Custom OpenSeadragon DrawerBase subclass that performs 16-channel
//  fluorescence microscopy HSV-to-RGB blending entirely on the GPU.
//  Slider/checkbox changes push uniforms and trigger immediate re-render
//  without per-tile CPU reprocessing.  Registered as drawer type
//  'hyperblend-webgl' so OSD can instantiate it via { drawer: '...' }.
// =========================================================================
/**
 * HyperBlend WebGL Drawer for OpenSeadragon 6.0.2
 *
 * A custom DrawerBase subclass that performs 16-channel HSV-to-RGB blending
 * entirely on the GPU via a fragment shader.  Slider changes push new uniforms
 * and trigger an immediate re-render — no per-tile CPU reprocessing.
 *
 * Usage:
 *   <script src="openseadragon/hyperblend-webgl.js"></script>
 *   new OpenSeadragon({ drawer: 'hyperblend-webgl', ... });
 */
(function ($) {

    'use strict';

    // =========================================================================
    //  GLSL SHADER SOURCES (ES 1.0)
    //  Vertex shader: pass-through for a full-screen quad.
    //  Fragment shader: samples 4 layer textures, processes 16 channels with
    //  pre-computed RGB colors + gain (4 color channels: RGBA), and additively blends.
    // =========================================================================

    // -----------------------------------------------------------------------
    // Shader sources (GLSL ES 1.0 for maximum compatibility)
    // -----------------------------------------------------------------------

    var VERTEX_SHADER_SRC = [
        'attribute vec2 aPosition;',
        'attribute vec2 aTexCoord;',
        'varying vec2 vTexCoord;',
        'void main() {',
        '    vTexCoord = aTexCoord;',
        '    gl_Position = vec4(aPosition, 0.0, 1.0);',
        '}'
    ].join('\n');

    var FRAGMENT_SHADER_SRC = [
        'precision mediump float;',
        '',
        'varying vec2 vTexCoord;',
        '',
        'uniform sampler2D uLayer0;',
        'uniform sampler2D uLayer1;',
        'uniform sampler2D uLayer2;',
        'uniform sampler2D uLayer3;',
        '',
        'uniform vec3  uChannelColor[16];',
        'uniform float uChannelGain[16];',
        'uniform float uChannelEnabled[16];',
        'uniform float uToneMode;',
        'uniform float uUnmixEnabled;',
        'uniform float uUnmixMatrix[128];',
        '',
        'void main() {',
        '    // Pre-sample all 4 layer textures once (4 fetches total, not 16)',
        '    // FBO compositing with UNPACK_PREMULTIPLY_ALPHA_WEBGL=false preserves',
        '    // all 4 RGBA channels — no unpremultiply needed.',
        '    vec4 L0 = texture2D(uLayer0, vTexCoord);',
        '    vec4 L1 = texture2D(uLayer1, vTexCoord);',
        '    vec4 L2 = texture2D(uLayer2, vTexCoord);',
        '    vec4 L3 = texture2D(uLayer3, vTexCoord);',
        '',
        '    // Extract 16 raw channel values',
        '    float r0=L0.r, r1=L0.g, r2=L0.b, r3=L0.a;',
        '    float r4=L1.r, r5=L1.g, r6=L1.b, r7=L1.a;',
        '    float r8=L2.r, r9=L2.g, r10=L2.b, r11=L2.a;',
        '    float r12=L3.r, r13=L3.g, r14=L3.b, r15=L3.a;',
        '',
        '    // Compute blend-input channels: unmixed outputs or raw passthrough',
        '    float ch0, ch1, ch2, ch3, ch4, ch5, ch6, ch7;',
        '    float ch8, ch9, ch10, ch11, ch12, ch13, ch14, ch15;',
        '',
        '    if (uUnmixEnabled > 0.5) {',
        '        // 16x8 matrix multiply: out[j] = sum_i(raw[i] * M[i*8+j])',
        '        ch0 = r0*uUnmixMatrix[0] + r1*uUnmixMatrix[8] + r2*uUnmixMatrix[16] + r3*uUnmixMatrix[24] + r4*uUnmixMatrix[32] + r5*uUnmixMatrix[40] + r6*uUnmixMatrix[48] + r7*uUnmixMatrix[56] + r8*uUnmixMatrix[64] + r9*uUnmixMatrix[72] + r10*uUnmixMatrix[80] + r11*uUnmixMatrix[88] + r12*uUnmixMatrix[96] + r13*uUnmixMatrix[104] + r14*uUnmixMatrix[112] + r15*uUnmixMatrix[120];',
        '        ch1 = r0*uUnmixMatrix[1] + r1*uUnmixMatrix[9] + r2*uUnmixMatrix[17] + r3*uUnmixMatrix[25] + r4*uUnmixMatrix[33] + r5*uUnmixMatrix[41] + r6*uUnmixMatrix[49] + r7*uUnmixMatrix[57] + r8*uUnmixMatrix[65] + r9*uUnmixMatrix[73] + r10*uUnmixMatrix[81] + r11*uUnmixMatrix[89] + r12*uUnmixMatrix[97] + r13*uUnmixMatrix[105] + r14*uUnmixMatrix[113] + r15*uUnmixMatrix[121];',
        '        ch2 = r0*uUnmixMatrix[2] + r1*uUnmixMatrix[10] + r2*uUnmixMatrix[18] + r3*uUnmixMatrix[26] + r4*uUnmixMatrix[34] + r5*uUnmixMatrix[42] + r6*uUnmixMatrix[50] + r7*uUnmixMatrix[58] + r8*uUnmixMatrix[66] + r9*uUnmixMatrix[74] + r10*uUnmixMatrix[82] + r11*uUnmixMatrix[90] + r12*uUnmixMatrix[98] + r13*uUnmixMatrix[106] + r14*uUnmixMatrix[114] + r15*uUnmixMatrix[122];',
        '        ch3 = r0*uUnmixMatrix[3] + r1*uUnmixMatrix[11] + r2*uUnmixMatrix[19] + r3*uUnmixMatrix[27] + r4*uUnmixMatrix[35] + r5*uUnmixMatrix[43] + r6*uUnmixMatrix[51] + r7*uUnmixMatrix[59] + r8*uUnmixMatrix[67] + r9*uUnmixMatrix[75] + r10*uUnmixMatrix[83] + r11*uUnmixMatrix[91] + r12*uUnmixMatrix[99] + r13*uUnmixMatrix[107] + r14*uUnmixMatrix[115] + r15*uUnmixMatrix[123];',
        '        ch4 = r0*uUnmixMatrix[4] + r1*uUnmixMatrix[12] + r2*uUnmixMatrix[20] + r3*uUnmixMatrix[28] + r4*uUnmixMatrix[36] + r5*uUnmixMatrix[44] + r6*uUnmixMatrix[52] + r7*uUnmixMatrix[60] + r8*uUnmixMatrix[68] + r9*uUnmixMatrix[76] + r10*uUnmixMatrix[84] + r11*uUnmixMatrix[92] + r12*uUnmixMatrix[100] + r13*uUnmixMatrix[108] + r14*uUnmixMatrix[116] + r15*uUnmixMatrix[124];',
        '        ch5 = r0*uUnmixMatrix[5] + r1*uUnmixMatrix[13] + r2*uUnmixMatrix[21] + r3*uUnmixMatrix[29] + r4*uUnmixMatrix[37] + r5*uUnmixMatrix[45] + r6*uUnmixMatrix[53] + r7*uUnmixMatrix[61] + r8*uUnmixMatrix[69] + r9*uUnmixMatrix[77] + r10*uUnmixMatrix[85] + r11*uUnmixMatrix[93] + r12*uUnmixMatrix[101] + r13*uUnmixMatrix[109] + r14*uUnmixMatrix[117] + r15*uUnmixMatrix[125];',
        '        ch6 = r0*uUnmixMatrix[6] + r1*uUnmixMatrix[14] + r2*uUnmixMatrix[22] + r3*uUnmixMatrix[30] + r4*uUnmixMatrix[38] + r5*uUnmixMatrix[46] + r6*uUnmixMatrix[54] + r7*uUnmixMatrix[62] + r8*uUnmixMatrix[70] + r9*uUnmixMatrix[78] + r10*uUnmixMatrix[86] + r11*uUnmixMatrix[94] + r12*uUnmixMatrix[102] + r13*uUnmixMatrix[110] + r14*uUnmixMatrix[118] + r15*uUnmixMatrix[126];',
        '        ch7 = r0*uUnmixMatrix[7] + r1*uUnmixMatrix[15] + r2*uUnmixMatrix[23] + r3*uUnmixMatrix[31] + r4*uUnmixMatrix[39] + r5*uUnmixMatrix[47] + r6*uUnmixMatrix[55] + r7*uUnmixMatrix[63] + r8*uUnmixMatrix[71] + r9*uUnmixMatrix[79] + r10*uUnmixMatrix[87] + r11*uUnmixMatrix[95] + r12*uUnmixMatrix[103] + r13*uUnmixMatrix[111] + r14*uUnmixMatrix[119] + r15*uUnmixMatrix[127];',
        '        // Clamp unmixed outputs to [0,1]',
        '        ch0 = clamp(ch0, 0.0, 1.0); ch1 = clamp(ch1, 0.0, 1.0);',
        '        ch2 = clamp(ch2, 0.0, 1.0); ch3 = clamp(ch3, 0.0, 1.0);',
        '        ch4 = clamp(ch4, 0.0, 1.0); ch5 = clamp(ch5, 0.0, 1.0);',
        '        ch6 = clamp(ch6, 0.0, 1.0); ch7 = clamp(ch7, 0.0, 1.0);',
        '        // Unused output slots are zero',
        '        ch8 = 0.0; ch9 = 0.0; ch10 = 0.0; ch11 = 0.0;',
        '        ch12 = 0.0; ch13 = 0.0; ch14 = 0.0; ch15 = 0.0;',
        '    } else {',
        '        // Bypass: raw channels pass through unchanged',
        '        ch0=r0; ch1=r1; ch2=r2; ch3=r3;',
        '        ch4=r4; ch5=r5; ch6=r6; ch7=r7;',
        '        ch8=r8; ch9=r9; ch10=r10; ch11=r11;',
        '        ch12=r12; ch13=r13; ch14=r14; ch15=r15;',
        '    }',
        '',
        '    // Additive blend with pre-computed RGB colors and tone mapping.',
        '    vec3 sum = vec3(0.0);',
        '',
        '    // Blend channels via ch0-ch15 (raw passthrough or unmixed outputs)',
        '    if (uChannelEnabled[0]  > 0.5) { sum += uChannelColor[0]  * ch0  * uChannelGain[0];  }',
        '    if (uChannelEnabled[1]  > 0.5) { sum += uChannelColor[1]  * ch1  * uChannelGain[1];  }',
        '    if (uChannelEnabled[2]  > 0.5) { sum += uChannelColor[2]  * ch2  * uChannelGain[2];  }',
        '    if (uChannelEnabled[3]  > 0.5) { sum += uChannelColor[3]  * ch3  * uChannelGain[3];  }',
        '    if (uChannelEnabled[4]  > 0.5) { sum += uChannelColor[4]  * ch4  * uChannelGain[4];  }',
        '    if (uChannelEnabled[5]  > 0.5) { sum += uChannelColor[5]  * ch5  * uChannelGain[5];  }',
        '    if (uChannelEnabled[6]  > 0.5) { sum += uChannelColor[6]  * ch6  * uChannelGain[6];  }',
        '    if (uChannelEnabled[7]  > 0.5) { sum += uChannelColor[7]  * ch7  * uChannelGain[7];  }',
        '    if (uChannelEnabled[8]  > 0.5) { sum += uChannelColor[8]  * ch8  * uChannelGain[8];  }',
        '    if (uChannelEnabled[9]  > 0.5) { sum += uChannelColor[9]  * ch9  * uChannelGain[9];  }',
        '    if (uChannelEnabled[10] > 0.5) { sum += uChannelColor[10] * ch10 * uChannelGain[10]; }',
        '    if (uChannelEnabled[11] > 0.5) { sum += uChannelColor[11] * ch11 * uChannelGain[11]; }',
        '    if (uChannelEnabled[12] > 0.5) { sum += uChannelColor[12] * ch12 * uChannelGain[12]; }',
        '    if (uChannelEnabled[13] > 0.5) { sum += uChannelColor[13] * ch13 * uChannelGain[13]; }',
        '    if (uChannelEnabled[14] > 0.5) { sum += uChannelColor[14] * ch14 * uChannelGain[14]; }',
        '    if (uChannelEnabled[15] > 0.5) { sum += uChannelColor[15] * ch15 * uChannelGain[15]; }',
        '',
        '    // Tone mapping: uToneMode selects the algorithm.',
        '    // 0 = Knee curve (default): linear below 0.8, soft shoulder above.',
        '    //     Preserves exact intensity ratios below knee for scientific accuracy.',
        '    // 1 = Reinhard: smooth compression everywhere (presentation/printing).',
        '    float lum = max(sum.r, max(sum.g, sum.b));',
        '    if (uToneMode < 0.5) {',
        '        // Knee curve: linear below 0.8, exponential shoulder above',
        '        if (lum > 0.8) {',
        '            float over = lum - 0.8;',
        '            float compressed = 0.8 + 0.2 * (1.0 - exp(-over / 0.2));',
        '            sum *= compressed / lum;',
        '        }',
        '    } else {',
        '        // Reinhard: luminance-preserving compression (all RGB scaled equally)',
        '        if (lum > 0.0) {',
        '            float compressed = lum / (lum + 1.0);',
        '            sum *= compressed / lum;',
        '        }',
        '    }',
        '    gl_FragColor = vec4(clamp(sum, 0.0, 1.0), 1.0);',
        '}'
    ].join('\n');

    // -----------------------------------------------------------------------
    // Beer's law post-process fragment shader (GLSL ES 1.0)
    // Single shader for both H&E (2-stain) and Trichrome (3-stain).
    // u_mode controls channel mapping:
    //   H&E      (u_mode=0): nuclei=B, stroma=G
    //   Trichrome (u_mode=1): nuclei=G, stroma=R, collagen=B
    // -----------------------------------------------------------------------

    var BEERS_LAW_FRAGMENT_SRC = [
        'precision mediump float;',
        'varying vec2 vTexCoord;',
        '',
        'uniform sampler2D uHyperBlendOutput;',
        'uniform float u_k;',
        'uniform vec3 u_nucRGB;',
        'uniform vec3 u_strRGB;',
        'uniform vec3 u_colRGB;',
        'uniform float u_unmix;',
        'uniform float u_nucGain;',
        'uniform float u_strGain;',
        'uniform float u_colGain;',
        'uniform float u_mode;',   // 0.0 = H&E, 1.0 = Trichrome
        '',
        'void main() {',
        '    vec4 px = texture2D(uHyperBlendOutput, vTexCoord);',
        '',
        '    float nuc, str, col;',
        '    if (u_mode < 0.5) {',
        '        // H&E: nuclei from Blue, stroma from Green',
        '        nuc = px.b * 255.0 * u_nucGain;',
        '        str = px.g * 255.0 * u_strGain;',
        '        col = 0.0;',
        '    } else {',
        '        // Trichrome: nuclei from Green, stroma from Red, collagen from Blue',
        '        nuc = px.g * 255.0 * u_nucGain;',
        '        str = px.r * 255.0 * u_strGain;',
        '        col = px.b * 255.0 * u_colGain;',
        '    }',
        '',
        '    // Spectral unmixing (H&E only; for Trichrome u_unmix = 0.0)',
        '    str = max(0.0, str - u_unmix * nuc);',
        '',
        '    float k1 = exp(-u_k);',
        '    float k2 = 1.0 / (1.0 - k1);',
        '',
        '    // Cascade 1: stroma (eosin)',
        '    vec3 result = (exp(-u_k * (str / 255.0) * u_strRGB) - k1) * k2;',
        '',
        '    // Cascade 2: nuclei (hematoxylin)',
        '    result *= (exp(-u_k * (nuc / 255.0) * u_nucRGB) - k1) * k2;',
        '',
        '    // Cascade 3: collagen (Trichrome only — skipped when u_colGain = 0.0)',
        '    if (u_mode > 0.5) {',
        '        result *= (exp(-u_k * (col / 255.0) * u_colRGB) - k1) * k2;',
        '    }',
        '',
        '    gl_FragColor = vec4(result, 1.0);',
        '}'
    ].join('\n');

    var BLIT_FRAGMENT_SRC = [
        'precision mediump float;',
        'varying vec2 vTexCoord;',
        'uniform sampler2D uTile;',
        'void main() {',
        '    gl_FragColor = texture2D(uTile, vTexCoord);',
        '}'
    ].join('\n');

    // =========================================================================
    //  CLASS DEFINITION / CONSTRUCTOR
    //  HyperBlendWebGLDrawer extends OpenSeadragon.DrawerBase.
    //  Constructor initialises channel config, off-screen compositing canvases,
    //  WebGL context/program, and fast-path optimisation state.
    // =========================================================================

    // -----------------------------------------------------------------------
    // HyperBlendWebGLDrawer class
    // -----------------------------------------------------------------------

    var HyperBlendWebGLDrawer = class HyperBlendWebGLDrawer extends $.DrawerBase {

        constructor(options) {
            super(options);

            // Channel mode: 'rgba' (16ch) or 'rgb' (12ch — alpha channels locked)
            var channelMode = (options && options.channelMode) ||
                              (typeof window.__hyperBlendChannelMode !== 'undefined' ? window.__hyperBlendChannelMode : 'rgba');
            this._channelsPerImage = (channelMode === 'rgb') ? 3 : 4;

            // Locked channels: in RGB mode, channels 3/7/11/15 (.a slots) are
            // permanently disabled because RGB tiles have alpha=1.0 (not data).
            this._lockedChannels = new Set();
            if (this._channelsPerImage === 3) {
                for (var lc = 0; lc < 4; lc++) this._lockedChannels.add(lc * 4 + 3);
            }

            // Tone mapping mode: 0 = knee curve (default), 1 = Reinhard
            this._toneMode = 0;

            // Channel configuration: 16 channels, each { h, gain, enabled }
            // S is hardcoded 1.0 in _precomputeChannelColors(); V removed (redundant with Gain)
            this._channelConfig = [];
            for (var i = 0; i < 16; i++) {
                this._channelConfig.push({
                    h: Math.round((0.66 - (i / 15) * 0.66) * 100) / 100,
                    gain: 1.0,
                    enabled: false
                });
            }
            // Per-layer FBOs for compositing tiles (bypasses canvas premultiplied alpha)
            this._layerFBOs = [];
            this._layerFBOTextures = [];
            this._layerFBOWidth = 0;
            this._layerFBOHeight = 0;

            // Blit shader resources (for drawing tiles into layer FBOs)
            this._blitProgram = null;
            this._blitUniforms = {};
            this._blitAttribs = {};
            // Tile texture cache: avoids redundant CPU→GPU uploads during pan/zoom
            this._tileCache = new Map();
            this._frameCount = 0;
            // Cache cap sized for 1024×1024 tiles (~4MB VRAM each)
            // 300 entries ≈ ~1.2GB VRAM max — reasonable for modern GPUs
            this._tileCacheCap = 300;
            this._evictionInterval = 60;    // check eviction every N frames
            this._evictionMaxAge = 240;     // evict tiles unused for N frames (~4s at 60fps)
            this._inactiveSettleInterval = 30; // settle inactive layers every N frames (not every frame)
            this._blitPosBuffer = null;
            this._clipData = new Float32Array(8);  // pre-allocated for tile blit (avoids per-tile GC)
            this._corners = new Float32Array(8);   // pre-allocated for tile corner computation
            this._lastLayerTileHashes = [];        // per-layer tile hashes for selective re-composite

            // WebGL setup
            this._gl = null;
            this._program = null;
            this._uniformLocations = {};
            this._attribLocations = {};  // cached attribute locations
            this._posBuffer = null;

            // Optimization: skip tile compositing when only uniforms changed
            this._texturesValid = false;   // true when layer textures are up-to-date
            this._lastVpX = NaN;
            this._lastVpY = NaN;
            this._lastVpW = NaN;
            this._lastVpRot = NaN;
            this._lastFlip = -1;
            this._lastZIndex = -1;         // detect z-level changes
            this._lastCanvasW = 0;
            this._lastCanvasH = 0;
            this._lastTileCounts = [];     // per-layer tile count for lightweight change detection
            this._lastTileChecksums = [];  // per-layer XOR checksum of cacheKey hashes
            this._lastTileSumChecksums = [];  // per-layer additive checksum (collision resistance)
            this._tileDropRejectedAge = 0;    // frames since last tile-drop rejection; escape hatch at 60

            // Post-process (Beer's law) state
            this._postProcessConfig = {
                active: false,
                filterType: 'none',  // 'he', 'trichrome', or 'none'
                mode: 0.0,           // 0.0 = H&E (nuc=B, str=G), 1.0 = Trichrome (nuc=G, str=R, col=B)
                k: 2.5,
                unmix: 0.1,
                nucGain: 1.0,
                strGain: 1.0,
                colGain: 0.0,
                nucRGB: [0.86, 1.0, 0.30],
                strRGB: [0.05, 1.0, 0.544],
                colRGB: [0.2, 0.05, 0.3]
            };
            this._postProcessProgram = null;
            this._postProcessUniforms = {};
            this._postProcessAttribs = {};
            this._fbo = null;
            this._fboTexture = null;
            this._fboWidth = 0;
            this._fboHeight = 0;

            // Pre-allocated uniform arrays (avoid per-frame GC)
            this._uColor = new Float32Array(48);  // 16 channels × 3 RGB (pre-computed from HSV)
            this._uGain = new Float32Array(16);
            this._uEnabled = new Float32Array(16);
            this._precomputeChannelColors();  // initialize _uColor from default config

            // Context-loss banner
            this._contextBanner = null;

            // Linear unmixing state
            this._unmixEnabled = false;
            this._unmixMatrix = new Float32Array(128);  // 16 inputs x 8 outputs, zero-padded
            this._unmixMatrixDirty = true;
            this._numOutputs = 0;

            try {
                this._initWebGL();
            } catch (e) {
                $.console.error('[HyperBlendWebGLDrawer] _initWebGL threw:', e);
                this._gl = null;
                this._program = null;
            }

            // WebGL context loss/restore handling
            this._contextLost = false;
            var self = this;
            this.canvas.addEventListener('webglcontextlost', function(e) {
                e.preventDefault();  // required to allow restoration
                self._contextLost = true;
                $.console.warn('[HyperBlendWebGLDrawer] WebGL context lost');
                if (self.viewer && self.viewer.element) {
                    if (!self._contextBanner) {
                        self._contextBanner = document.createElement('div');
                        self._contextBanner.style.cssText = 'position:absolute;top:0;left:0;right:0;padding:12px 16px;z-index:10000;background:#d32f2f;color:#fff;font:bold 14px/1.4 sans-serif;text-align:center;pointer-events:none;';
                        self.viewer.element.style.position = self.viewer.element.style.position || 'relative';
                        self.viewer.element.appendChild(self._contextBanner);
                    }
                    self._contextBanner.textContent = 'WebGL context lost \u2014 recovering GPU resources\u2026';
                    self._contextBanner.style.display = 'block';
                }
            });
            this.canvas.addEventListener('webglcontextrestored', function() {
                $.console.log('[HyperBlendWebGLDrawer] WebGL context restored, reinitializing');
                self._contextLost = false;
                if (self._contextBanner) self._contextBanner.style.display = 'none';
                self._tileCache.clear();
                self._layerFBOs = [];
                self._layerFBOTextures = [];
                self._layerFBOWidth = 0;
                self._layerFBOHeight = 0;
                self._fboWidth = 0;
                self._fboHeight = 0;
                self._lastTileCounts = [];
                self._lastTileChecksums = [];
                self._lastTileSumChecksums = [];
                self._lastVpX = NaN;
                self._lastVpY = NaN;
                self._lastVpW = NaN;
                self._lastVpRot = NaN;
                self._lastFlip = -1;
                self._lastLayerTileHashes = [];
                self._texturesValid = false;
                self._unmixMatrixDirty = true;
                try {
                    self._initWebGL();
                    if (self.viewer) self.viewer.forceRedraw();
                } catch (e) {
                    $.console.error('[HyperBlendWebGLDrawer] Failed to reinit after context restore:', e);
                }
            });

            // Allow tile-drawn events (for filter bundle compatibility)
            try {
                this.viewer.allowEventHandler("tile-drawn");
                this.viewer.allowEventHandler("tile-drawing");
            } catch (e) {
                $.console.warn('[HyperBlendWebGLDrawer] allowEventHandler failed:', e);
            }

        }

        // =====================================================================
        //  DRAWERBASE ABSTRACT METHOD IMPLEMENTATIONS
        //  Required overrides: isSupported, getType, getSupportedDataFormats,
        //  _createDrawingElement, canRotate, setImageSmoothingEnabled, destroy.
        // =====================================================================

        // ---- Abstract method implementations ----

        static isSupported() {
            if (typeof document === 'undefined') { return false; }
            var c = document.createElement('canvas');
            var gl = c.getContext('webgl2') || c.getContext('webgl') || c.getContext('experimental-webgl');
            if (!gl) { return false; }
            // Validate that all shaders compile and FBO works on this GPU/driver
            try {
                // Helper to compile and check a shader
                function testShader(type, src) {
                    var s = gl.createShader(type);
                    gl.shaderSource(s, src);
                    gl.compileShader(s);
                    var ok = gl.getShaderParameter(s, gl.COMPILE_STATUS);
                    return { shader: s, ok: ok };
                }

                // Test main vertex + fragment shaders
                var vs = testShader(gl.VERTEX_SHADER, VERTEX_SHADER_SRC);
                var fs = testShader(gl.FRAGMENT_SHADER, FRAGMENT_SHADER_SRC);
                // Test blit fragment shader
                var blitFs = testShader(gl.FRAGMENT_SHADER, BLIT_FRAGMENT_SRC);
                // Test Beer's law fragment shader
                var beersFs = testShader(gl.FRAGMENT_SHADER, BEERS_LAW_FRAGMENT_SRC);

                var allOk = vs.ok && fs.ok && blitFs.ok && beersFs.ok;

                // Test FBO creation
                var fboOk = false;
                if (allOk) {
                    var tex = gl.createTexture();
                    gl.bindTexture(gl.TEXTURE_2D, tex);
                    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
                    var fbo = gl.createFramebuffer();
                    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
                    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
                    fboOk = (gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE);
                    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
                    gl.deleteFramebuffer(fbo);
                    gl.deleteTexture(tex);
                }

                // Cleanup all test shaders
                gl.deleteShader(vs.shader);
                gl.deleteShader(fs.shader);
                gl.deleteShader(blitFs.shader);
                gl.deleteShader(beersFs.shader);
                c.width = 1;
                c.height = 1;

                if (!allOk) {
                    console.warn('[HyperBlendWebGLDrawer] isSupported: shader compilation failed');
                    return false;
                }
                if (!fboOk) {
                    console.warn('[HyperBlendWebGLDrawer] isSupported: FBO creation failed');
                    return false;
                }
            } catch (e) {
                c.width = 1;
                c.height = 1;
                console.warn('[HyperBlendWebGLDrawer] isSupported: exception during capability test:', e);
                return false;
            }
            return true;
        }

        getType() {
            return 'hyperblend-webgl';
        }

        getSupportedDataFormats() {
            return ['context2d', 'image'];
        }

        _createDrawingElement() {
            var canvas = $.makeNeutralElement('canvas');
            var viewportSize = this._calculateCanvasSize();
            canvas.width = viewportSize.x;
            canvas.height = viewportSize.y;
            return canvas;
        }

        canRotate() {
            return true;
        }

        setImageSmoothingEnabled(enabled) {
            // No-op: FBO compositing does not use 2D canvas smoothing
        }

        destroy() {
            if (this._contextBanner && this._contextBanner.parentNode) {
                this._contextBanner.parentNode.removeChild(this._contextBanner);
                this._contextBanner = null;
            }
            var gl = this._gl;
            if (gl) {
                // Layer FBOs
                for (var i = 0; i < this._layerFBOTextures.length; i++) {
                    if (this._layerFBOTextures[i]) gl.deleteTexture(this._layerFBOTextures[i]);
                    if (this._layerFBOs[i]) gl.deleteFramebuffer(this._layerFBOs[i]);
                }
                // Tile texture cache
                if (this._tileCache) {
                    var self = this;
                    this._tileCache.forEach(function(entry) {
                        gl.deleteTexture(entry.texture);
                    });
                    this._tileCache.clear();
                }
                // Blit resources
                if (this._blitPosBuffer) gl.deleteBuffer(this._blitPosBuffer);
                if (this._blitProgram) gl.deleteProgram(this._blitProgram);
                // Main resources
                if (this._posBuffer) gl.deleteBuffer(this._posBuffer);
                if (this._texBufferFBO) gl.deleteBuffer(this._texBufferFBO);
                if (this._program) gl.deleteProgram(this._program);
                // Post-process resources
                if (this._postProcessProgram) gl.deleteProgram(this._postProcessProgram);
                if (this._fboTexture) gl.deleteTexture(this._fboTexture);
                if (this._fbo) gl.deleteFramebuffer(this._fbo);
            }
            this._layerFBOs = [];
            this._layerFBOTextures = [];
            this.canvas.width = 1;
            this.canvas.height = 1;
            this.container.removeChild(this.canvas);
            super.destroy();
        }

        // =====================================================================
        //  MAIN DRAW METHOD
        //  Called by OSD each frame.  Two code paths:
        //    FULL PATH — viewport, z-level, or tiles changed: re-composite
        //      tiles onto off-screen canvases, upload as GL textures, then
        //      set uniforms and draw a full-screen quad.
        //    FAST PATH — only channel config (uniforms) changed: skip tile
        //      compositing, rebind cached textures, update uniforms, draw quad.
        // =====================================================================

        // ---- Main draw method ----

        draw(tiledImages) {
            if (this._contextLost) return;  // suppress draws during context loss
            var gl = this._gl;
            if (!gl || !this._program) {
                if (tiledImages && tiledImages.length > 0) {
                    this._raiseDrawerErrorEvent(tiledImages[0],
                        'HyperBlendWebGLDrawer: WebGL context or shader program unavailable.');
                }
                return;
            }

            // Resize the canvas if the viewport changed
            var viewportSize = this._calculateCanvasSize();
            if (this.canvas.width !== viewportSize.x || this.canvas.height !== viewportSize.y) {
                this.canvas.width = viewportSize.x;
                this.canvas.height = viewportSize.y;
                this._texturesValid = false;
            }

            var canvasW = this.canvas.width;
            var canvasH = this.canvas.height;

            // Detect whether viewport/z-level changed (requires full re-composite)
            this._frameCount++;
            var zIdx = (typeof window.currentZIndex !== 'undefined') ? window.currentZIndex : 0;
            var vpBounds = this.viewport.getBounds(true);
            var vpX = Math.round(vpBounds.x * 1e6);
            var vpY = Math.round(vpBounds.y * 1e6);
            var vpW = Math.round(vpBounds.width * 1e6);
            var vpRot = Math.round(this.viewport.getRotation(true) * 100);
            var vpFlip = this.viewport.getFlip() ? 1 : 0;

            // Check if tiles changed (new tiles loaded, tiles unloaded, etc.)
            // NOTE: Do NOT include _needsDraw in the hash — forceRedraw() sets it
            // on every slider change, which would defeat the fast path entirely.
            var activeLayers = this._getActiveLayers(tiledImages);
            var viewportChanged = (vpX !== this._lastVpX || vpY !== this._lastVpY ||
                                    vpW !== this._lastVpW || vpRot !== this._lastVpRot ||
                                    vpFlip !== this._lastFlip);
            var zChanged = (zIdx !== this._lastZIndex);
            var sizeChanged = (canvasW !== this._lastCanvasW || canvasH !== this._lastCanvasH);

            // Skip expensive string hashing when viewport already forces recomposite.
            // For static viewports, use count + checksum (detects both tile additions
            // and same-count replacements without GC-heavy string concatenation).
            //
            // IMPORTANT: Only trigger re-composite when tiles IMPROVE (count increases).
            // OSD cache eviction can cause tile counts to DROP (fine tiles evicted,
            // coarse fallbacks returned by getTilesToDraw). Re-compositing with fewer
            // tiles would overwrite our good FBO content with blurry coarse tiles.
            // Our GPU _tileCache retains the fine textures independently of OSD's cache.
            var tilesChanged = false;
            var tileDropRejected = false;
            if (!viewportChanged && !zChanged && !sizeChanged) {
                var totalNow = 0, totalPrev = 0;
                var newCounts = [], newChecksums = [], newSumChecksums = [];
                for (var thi = 0; thi < activeLayers.length; thi++) {
                    var ti = activeLayers[thi];
                    var count = 0;
                    var checksum = 0;
                    var sumChecksum = 0;
                    if (ti && ti._tilesToDraw) {
                        for (var thj = 0; thj < ti._tilesToDraw.length; thj++) {
                            var levelTiles = ti._tilesToDraw[thj];
                            if (Array.isArray(levelTiles)) {
                                for (var thk = 0; thk < levelTiles.length; thk++) {
                                    if (levelTiles[thk] && levelTiles[thk].tile) {
                                        count++;
                                        var h = this._hashCacheKey(levelTiles[thk].tile.cacheKey);
                                        checksum = (checksum ^ h) | 0;
                                        sumChecksum = (sumChecksum + h) | 0;
                                    }
                                }
                            }
                        }
                    }
                    totalNow += count;
                    totalPrev += (this._lastTileCounts[thi] || 0);
                    newCounts.push(count);
                    newChecksums.push(checksum);
                    newSumChecksums.push(sumChecksum);
                    if (count !== (this._lastTileCounts[thi] || 0) || checksum !== (this._lastTileChecksums[thi] || 0) || sumChecksum !== (this._lastTileSumChecksums[thi] || 0)) {
                        tilesChanged = true;
                    }
                }
                // Reject tile-change composites when total tile count dropped —
                // OSD cache eviction replaced fine tiles with coarse fallbacks.
                // Keep existing FBO content (which has the fine tiles from GPU cache).
                // Do NOT update tracking on rejection so future frames compare
                // against the last good composite, not the degraded state.
                // Escape hatch: after 60 frames (~1s) of continuous rejection,
                // force-accept to prevent indefinitely stale FBO content.
                if (tilesChanged && totalNow < totalPrev && this._tileDropRejectedAge < 60) {
                    tilesChanged = false;
                    tileDropRejected = true;
                    this._tileDropRejectedAge++;
                    // Protect tiles from OSD cache eviction during rejection.
                    // Without this, rejected tiles lose beingDrawn protection
                    // (reset by OSD's _updateLevelsForViewport), OSD evicts them,
                    // causing cascading tile drops (vicious cycle).
                    for (var bdi = 0; bdi < activeLayers.length; bdi++) {
                        var bdTi = activeLayers[bdi];
                        if (bdTi && bdTi._tilesToDraw) {
                            for (var bdj = 0; bdj < bdTi._tilesToDraw.length; bdj++) {
                                var bdLevel = bdTi._tilesToDraw[bdj];
                                if (Array.isArray(bdLevel)) {
                                    for (var bdk = 0; bdk < bdLevel.length; bdk++) {
                                        if (bdLevel[bdk] && bdLevel[bdk].tile && bdLevel[bdk].tile.loaded) {
                                            bdLevel[bdk].tile.beingDrawn = true;
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else if (tilesChanged) {
                    this._lastTileCounts = newCounts;
                    this._lastTileChecksums = newChecksums;
                    this._lastTileSumChecksums = newSumChecksums;
                    this._tileDropRejectedAge = 0;
                }
            }

            // Fast path: skip composite only when NOTHING changed except uniforms.
            // Viewport/tile/z changes always force a full composite — a slider
            // event coinciding with zoom must not suppress tile uploads.
            var needsComposite = !this._texturesValid || viewportChanged || zChanged || sizeChanged || tilesChanged;

            if (needsComposite) {
                // ---- FULL PATH: composite tiles into per-layer FBOs ----
                // Unbind layer textures from units 0-3 to prevent feedback loop
                // (they may still be bound from previous frame's Pass 1)
                for (var ub = 0; ub < this._layerFBOTextures.length; ub++) {
                    gl.activeTexture(gl.TEXTURE0 + ub);
                    gl.bindTexture(gl.TEXTURE_2D, null);
                }
                this._resizeLayerFBOs(canvasW, canvasH);

                // getTilesToDraw() ONCE per TiledImage (OSD rule: once per frame)
                var layerTileInfos = [];
                for (var li = 0; li < activeLayers.length; li++) {
                    var tiledImage = activeLayers[li];
                    if (!tiledImage) {
                        layerTileInfos.push([]);
                        continue;
                    }
                    layerTileInfos.push(tiledImage.getTilesToDraw());
                }

                // Composite tiles into layer FBOs (bypasses 2D canvas premultiplied alpha)
                // Per-layer optimization: skip layers whose tiles haven't changed
                var forceAll = !this._texturesValid || viewportChanged || zChanged || sizeChanged;
                gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
                gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
                for (var li = 0; li < activeLayers.length; li++) {
                    if (!activeLayers[li] || layerTileInfos[li].length === 0) {
                        if (li < this._layerFBOs.length && (forceAll || !this._lastLayerTileHashes[li])) {
                            gl.bindFramebuffer(gl.FRAMEBUFFER, this._layerFBOs[li]);
                            gl.viewport(0, 0, canvasW, canvasH);
                            gl.clearColor(0, 0, 0, 0);
                            gl.clear(gl.COLOR_BUFFER_BIT);
                        }
                        this._lastLayerTileHashes[li] = null;
                        continue;
                    }
                    // Always composite all layers — even those with no enabled channels.
                    // The shader handles enabled/disabled via uniforms; FBOs must contain
                    // raw tile data for both the spectrum inspector (readPixelChannels reads
                    // ALL channels) and unmixing (matrix multiply needs all 16 inputs).
                    // Check if this layer's tiles actually changed (integer count+checksum, no GC)
                    var lhCount = layerTileInfos[li].length;
                    var lhChecksum = 0;
                    var lhSumChecksum = 0;
                    for (var lhi = 0; lhi < lhCount; lhi++) {
                        var lhHash = this._hashCacheKey(layerTileInfos[li][lhi].tile.cacheKey);
                        lhChecksum = (lhChecksum ^ lhHash) | 0;
                        lhSumChecksum = (lhSumChecksum + lhHash) | 0;
                    }
                    var prev = this._lastLayerTileHashes[li];
                    if (!forceAll && prev && prev.count === lhCount && prev.checksum === lhChecksum && prev.sumChecksum === lhSumChecksum) {
                        continue;  // skip — this layer's FBO is still valid
                    }
                    this._compositeTilesToFBO(activeLayers[li], layerTileInfos[li], li, canvasW, canvasH, forceAll);
                    this._lastLayerTileHashes[li] = { count: lhCount, checksum: lhChecksum, sumChecksum: lhSumChecksum };
                }
                gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
                gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);

                // Ensure no layer FBO is bound before sampling layer textures
                // (skip-disabled and empty-layer paths clear FBOs without unbinding)
                gl.bindFramebuffer(gl.FRAMEBUFFER, null);

                // Bind layer FBO textures to units 0-3
                for (var ti = 0; ti < this._layerFBOTextures.length; ti++) {
                    gl.activeTexture(gl.TEXTURE0 + ti);
                    gl.bindTexture(gl.TEXTURE_2D, this._layerFBOTextures[ti]);
                }

                // Update state tracking
                this._lastVpX = vpX;
                this._lastVpY = vpY;
                this._lastVpW = vpW;
                this._lastVpRot = vpRot;
                this._lastFlip = vpFlip;
                this._lastZIndex = zIdx;
                this._lastCanvasW = canvasW;
                this._lastCanvasH = canvasH;

                var totalTilesComposited = 0;
                for (var tc = 0; tc < layerTileInfos.length; tc++) {
                    totalTilesComposited += layerTileInfos[tc].length;
                }
                // After a forced composite (viewport/z/size), reset tile tracking
                // to match the actually-composited state so future tile-change checks
                // compare against this baseline (not a stale "best" from before the change).
                if (forceAll) {
                    this._tileDropRejectedAge = 0;
                    this._lastTileCounts = [];
                    this._lastTileChecksums = [];
                    this._lastTileSumChecksums = [];
                    // Note: _lastLayerTileHashes are already set per-layer in the
                    // compositing loop above — do NOT reset them here or the next
                    // non-forceAll frame would re-composite all layers unnecessarily.
                    for (var rti = 0; rti < layerTileInfos.length; rti++) {
                        // Reuse per-layer hashes already computed in compositing loop
                        var cached = this._lastLayerTileHashes[rti];
                        this._lastTileCounts.push(cached ? cached.count : 0);
                        this._lastTileChecksums.push(cached ? cached.checksum : 0);
                        this._lastTileSumChecksums.push(cached ? cached.sumChecksum : 0);
                    }
                }
                // An empty composite is valid — FBOs are in correct state (cleared).
                // Setting false when totalTilesComposited=0 caused needsComposite=true
                // every frame, running the full path endlessly doing nothing.
                this._texturesValid = true;

                // Raise events for compatibility
                for (var ei = 0; ei < activeLayers.length; ei++) {
                    if (activeLayers[ei] && layerTileInfos[ei].length > 0) {
                        var lastDrawn = layerTileInfos[ei].map(function(info) { return info.tile; });
                        this._raiseTiledImageDrawnEvent(activeLayers[ei], lastDrawn);
                    }
                }
            } else {
                // ---- FAST PATH: FBO textures still valid, just re-bind them ----
                for (var fti = 0; fti < this._layerFBOTextures.length; fti++) {
                    gl.activeTexture(gl.TEXTURE0 + fti);
                    gl.bindTexture(gl.TEXTURE_2D, this._layerFBOTextures[fti]);
                }
            }

            // Settle inactive TiledImages so OSD's setDrawn() sees non-empty
            // _lastDrawn and stops re-triggering the render loop.
            // Throttled: only runs every _inactiveSettleInterval frames to
            // avoid per-frame CPU cost that scales with total z-level count.
            // Once settled, _lastDrawn persists and setDrawn() returns false
            // until OSD invalidates the layer (z-switch, tile load, etc.).
            if (this._frameCount % this._inactiveSettleInterval === 0) {
                for (var iAll = 0; iAll < tiledImages.length; iAll++) {
                    if (activeLayers.indexOf(tiledImages[iAll]) === -1) {
                        tiledImages[iAll].getTilesToDraw();
                    }
                }
            }

            // Periodic cache eviction (runs on all frames, not just composite frames)
            if (this._frameCount % this._evictionInterval === 0) {
                this._evictStaleTiles(viewportChanged);
            }

            // ---- Determine if post-processing is active ----
            var postActive = this._postProcessConfig.active && this._postProcessProgram;

            // ---- PASS 1: HyperBlend shader ----
            if (postActive) {
                // Render to FBO for post-processing
                this._resizeFBO(canvasW, canvasH);
                gl.bindFramebuffer(gl.FRAMEBUFFER, this._fbo);
            }

            gl.clearColor(0, 0, 0, 1);
            gl.clear(gl.COLOR_BUFFER_BIT);
            gl.viewport(0, 0, canvasW, canvasH);

            gl.useProgram(this._program);
            var loc = this._uniformLocations;

            // Fill pre-allocated uniform arrays
            var gArr = this._uGain, eArr = this._uEnabled;
            for (var ci = 0; ci < 16; ci++) {
                var cfg = this._channelConfig[ci];
                gArr[ci] = cfg.gain;
                eArr[ci] = cfg.enabled ? 1.0 : 0.0;
            }
            gl.uniform3fv(loc.uChannelColor, this._uColor);
            gl.uniform1fv(loc.uChannelGain, gArr);
            gl.uniform1fv(loc.uChannelEnabled, eArr);
            gl.uniform1f(loc.uToneMode, this._toneMode);

            // Unmixing uniforms
            gl.uniform1f(loc.uUnmixEnabled, this._unmixEnabled ? 1.0 : 0.0);
            if (this._unmixEnabled && this._unmixMatrixDirty) {
                gl.uniform1fv(loc.uUnmixMatrix, this._unmixMatrix);
                this._unmixMatrixDirty = false;
            }

            // Draw full-screen quad (Pass 1)
            var aPos = this._attribLocations.aPosition;
            var aTex = this._attribLocations.aTexCoord;
            gl.bindBuffer(gl.ARRAY_BUFFER, this._posBuffer);
            gl.enableVertexAttribArray(aPos);
            gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);
            gl.bindBuffer(gl.ARRAY_BUFFER, this._texBufferFBO);
            gl.enableVertexAttribArray(aTex);
            gl.vertexAttribPointer(aTex, 2, gl.FLOAT, false, 0, 0);
            gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

            // ---- PASS 2: Beer's law post-processing ----
            if (postActive) {
                gl.bindFramebuffer(gl.FRAMEBUFFER, null);
                gl.clearColor(0, 0, 0, 1);
                gl.clear(gl.COLOR_BUFFER_BIT);
                gl.viewport(0, 0, canvasW, canvasH);

                gl.useProgram(this._postProcessProgram);

                // Bind FBO texture to texture unit 4
                gl.activeTexture(gl.TEXTURE4);
                gl.bindTexture(gl.TEXTURE_2D, this._fboTexture);

                var pp = this._postProcessUniforms;
                var ppc = this._postProcessConfig;
                gl.uniform1f(pp.u_k, ppc.k);
                gl.uniform3f(pp.u_nucRGB, ppc.nucRGB[0], ppc.nucRGB[1], ppc.nucRGB[2]);
                gl.uniform3f(pp.u_strRGB, ppc.strRGB[0], ppc.strRGB[1], ppc.strRGB[2]);
                gl.uniform3f(pp.u_colRGB, ppc.colRGB[0], ppc.colRGB[1], ppc.colRGB[2]);
                gl.uniform1f(pp.u_unmix, ppc.unmix);
                gl.uniform1f(pp.u_nucGain, ppc.nucGain);
                gl.uniform1f(pp.u_strGain, ppc.strGain);
                gl.uniform1f(pp.u_colGain, ppc.colGain);
                gl.uniform1f(pp.u_mode, ppc.mode);

                // Draw full-screen quad (Pass 2) — use FBO tex coords (no Y-flip)
                var ppPos = this._postProcessAttribs.aPosition;
                var ppTex = this._postProcessAttribs.aTexCoord;
                gl.bindBuffer(gl.ARRAY_BUFFER, this._posBuffer);
                gl.enableVertexAttribArray(ppPos);
                gl.vertexAttribPointer(ppPos, 2, gl.FLOAT, false, 0, 0);
                gl.bindBuffer(gl.ARRAY_BUFFER, this._texBufferFBO);
                gl.enableVertexAttribArray(ppTex);
                gl.vertexAttribPointer(ppTex, 2, gl.FLOAT, false, 0, 0);
                gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
            }

        }

        // =====================================================================
        //  PUBLIC API
        //  updateChannelConfig — push new H/gain/enabled values (fast path)
        //  invalidateTextures  — force full re-composite on next draw
        //  getChannelConfig    — read current channel state
        // =====================================================================

        // ---- Public API ----

        /**
         * Update channel configuration and request a re-draw.
         * @param {Array} config - Array of 16 {h, gain, enabled} objects
         */
        updateChannelConfig(config) {
            if (!config || config.length !== 16) {
                $.console.warn('[HyperBlendWebGLDrawer] updateChannelConfig expects an array of 16 channel objects.');
                return;
            }
            // Check if any channel's enabled state changed — if so, a previously
            // skipped layer's FBO may be empty and needs a full re-composite
            var enabledChanged = false;
            for (var i = 0; i < 16; i++) {
                if (this._channelConfig[i].enabled !== !!config[i].enabled) {
                    enabledChanged = true;
                }
                this._channelConfig[i].h = config[i].h;
                // S/V no longer stored: S is hardcoded 1.0, V was redundant with Gain
                this._channelConfig[i].gain = config[i].gain;
                this._channelConfig[i].enabled = !!config[i].enabled;
                // Enforce locked channels (RGB mode: alpha slots always disabled)
                if (this._lockedChannels.has(i)) {
                    this._channelConfig[i].enabled = false;
                }
            }
            // enabledChanged: no FBO re-composite needed — all layers are always
            // composited regardless of enabled state (layer-skip optimization was
            // removed). The enabled flag is a shader uniform; FBO content is unaffected.
            // Pre-compute RGB colors from HSV (matches shader's old hsvToRgb, no 0.5 offset)
            this._precomputeChannelColors();
            if (this.viewer) {
                this.viewer.forceRedraw();
            }
        }

        /**
         * Invalidate cached textures (call on z-level switch, viewport change, etc.)
         */
        invalidateTextures() {
            this._texturesValid = false;
        }

        /**
         * Get the current channel configuration.
         * @returns {Array} Array of 16 {h, gain, enabled} objects
         */
        getChannelConfig() {
            return this._channelConfig.slice();
        }

        /**
         * Read raw channel intensities at a given CSS pixel position.
         * Reads 1 pixel from each layer FBO (up to 4 layers × RGBA = 16 channels).
         * @param {number} cssX - X coordinate relative to OSD container (CSS pixels)
         * @param {number} cssY - Y coordinate relative to OSD container (CSS pixels)
         * @returns {Uint8Array|null} 16-element array of channel values (0-255), or null if unavailable
         */
        readPixelChannels(cssX, cssY) {
            var gl = this._gl;
            if (!gl || this._contextLost || this._layerFBOs.length === 0) return null;

            // Ensure FBOs contain current data before reading
            if (!this._texturesValid && this.viewer && this.viewer.world) {
                var tiledImages = [];
                for (var wi = 0; wi < this.viewer.world.getItemCount(); wi++) {
                    tiledImages.push(this.viewer.world.getItemAt(wi));
                }
                this.draw(tiledImages);
            }
            if (!this._texturesValid) return null;

            var dpr = OpenSeadragon.pixelDensityRatio;
            var glX = Math.round(cssX * dpr);
            var glY = this.canvas.height - Math.round(cssY * dpr) - 1;

            if (glX < 0 || glX >= this.canvas.width || glY < 0 || glY >= this.canvas.height) return null;

            var result = new Uint8Array(16);
            var buf = new Uint8Array(4);

            for (var i = 0; i < this._layerFBOs.length; i++) {
                gl.bindFramebuffer(gl.FRAMEBUFFER, this._layerFBOs[i]);
                gl.readPixels(glX, glY, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, buf);
                result[i * 4]     = buf[0];
                result[i * 4 + 1] = buf[1];
                result[i * 4 + 2] = buf[2];
                result[i * 4 + 3] = buf[3];
            }

            gl.bindFramebuffer(gl.FRAMEBUFFER, null);
            return result;
        }

        /**
         * Set tone mapping mode.
         * @param {string} mode - 'knee' (default) or 'reinhard'
         */
        setToneMappingMode(mode) {
            this._toneMode = (mode === 'reinhard') ? 1 : 0;
            if (this.viewer) {
                this.viewer.forceRedraw();
            }
        }

        /**
         * Capture the current WebGL canvas content via readPixels.
         * Works without preserveDrawingBuffer — forces a synchronous
         * draw + readPixels within the same GPU frame.
         * Returns a 2D canvas with the captured image.
         */
        captureCanvas() {
            var gl = this._gl;
            if (!gl) return null;
            var w = this.canvas.width;
            var h = this.canvas.height;
            // Force a synchronous draw so the framebuffer has current content
            if (this.viewer && this.viewer.world) {
                var tiledImages = [];
                for (var i = 0; i < this.viewer.world.getItemCount(); i++) {
                    tiledImages.push(this.viewer.world.getItemAt(i));
                }
                this.draw(tiledImages);
            }
            // Read pixels from the default framebuffer (immediately after draw,
            // before the browser can clear it)
            gl.bindFramebuffer(gl.FRAMEBUFFER, null);
            var pixels = new Uint8Array(w * h * 4);
            gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
            // Create a 2D canvas and flip vertically (WebGL is bottom-up)
            var out = document.createElement('canvas');
            out.width = w;
            out.height = h;
            var ctx = out.getContext('2d');
            var imgData = ctx.createImageData(w, h);
            for (var row = 0; row < h; row++) {
                var srcOff = (h - 1 - row) * w * 4;
                var dstOff = row * w * 4;
                imgData.data.set(pixels.subarray(srcOff, srcOff + w * 4), dstOff);
            }
            ctx.putImageData(imgData, 0, 0);
            return out;
        }

        /**
         * Pre-compute RGB colors from channel H into _uColor Float32Array(48).
         * S is hardcoded to 1.0 (full saturation). V was redundant with Gain and is removed.
         * Matches hsvToRgb(h, 1.0, 1.0) — NO 0.5 hue offset.
         * The 0.5 offset in _hsvToRgb (HTML) is for Beer's Law only.
         */
        // ---- Internal: hash a cacheKey string (djb2) ----
        _hashCacheKey(key) {
            var h = 0;
            for (var i = 0; i < key.length; i++) {
                h = ((h << 5) - h + key.charCodeAt(i)) | 0;
            }
            return h;
        }

        // ---- Internal: compute count + XOR checksum for a tileInfo array ----
        _computeLayerTileStats(tileInfos) {
            var count = tileInfos.length;
            var checksum = 0;
            var sumChecksum = 0;
            for (var i = 0; i < tileInfos.length; i++) {
                if (tileInfos[i] && tileInfos[i].tile) {
                    var h = this._hashCacheKey(tileInfos[i].tile.cacheKey);
                    checksum = (checksum ^ h) | 0;
                    sumChecksum = (sumChecksum + h) | 0;
                }
            }
            return { count: count, checksum: checksum, sumChecksum: sumChecksum };
        }

        _precomputeChannelColors() {
            for (var i = 0; i < 16; i++) {
                var cfg = this._channelConfig[i];
                var h = cfg.h;
                var fi = Math.floor(h * 6.0);
                var f = h * 6.0 - fi;
                var r, g, b;
                var m = ((fi % 6) + 6) % 6;
                switch (m) {
                    case 0: r = 1.0; g = f;       b = 0.0; break;
                    case 1: r = 1.0 - f; g = 1.0; b = 0.0; break;
                    case 2: r = 0.0; g = 1.0;     b = f;   break;
                    case 3: r = 0.0; g = 1.0 - f; b = 1.0; break;
                    case 4: r = f;   g = 0.0;     b = 1.0; break;
                    default:r = 1.0; g = 0.0;     b = 1.0 - f; break;
                }
                this._uColor[i * 3]     = r;
                this._uColor[i * 3 + 1] = g;
                this._uColor[i * 3 + 2] = b;
            }
        }

        /**
         * Update post-process (Beer's law) configuration and request a re-draw.
         * @param {Object} config - { active, filterType, k, unmix, nucGain, strGain, colGain, nucRGB, strRGB, colRGB }
         */
        updatePostProcessConfig(config) {
            if (!config) return;
            var pp = this._postProcessConfig;
            if (typeof config.active !== 'undefined') pp.active = !!config.active;
            if (typeof config.filterType !== 'undefined') pp.filterType = config.filterType;
            if (typeof config.mode !== 'undefined') pp.mode = config.mode;
            if (typeof config.k !== 'undefined') pp.k = config.k;
            if (typeof config.unmix !== 'undefined') pp.unmix = config.unmix;
            if (typeof config.nucGain !== 'undefined') pp.nucGain = config.nucGain;
            if (typeof config.strGain !== 'undefined') pp.strGain = config.strGain;
            if (typeof config.colGain !== 'undefined') pp.colGain = config.colGain;
            if (config.nucRGB) pp.nucRGB = config.nucRGB.slice();
            if (config.strRGB) pp.strRGB = config.strRGB.slice();
            if (config.colRGB) pp.colRGB = config.colRGB.slice();
            if (this.viewer) {
                this.viewer.forceRedraw();
            }
        }

        /**
         * Update linear unmixing configuration.
         * @param {Object} config - { active: bool, matrix: Float32Array(128), numOutputs: int }
         */
        updateUnmixConfig(config) {
            if (!config) return;
            if (typeof config.active !== 'undefined') {
                var wasEnabled = this._unmixEnabled;
                this._unmixEnabled = !!config.active;
                // Unmixing toggle requires full FBO re-composite: when enabling,
                // layers 1-3 may have been cleared (their channels were "disabled");
                // when disabling, we need to restore raw channel FBO state.
                if (wasEnabled !== this._unmixEnabled) {
                    this._texturesValid = false;
                }
            }
            if (config.matrix) {
                this._unmixMatrix.set(config.matrix);
                this._unmixMatrixDirty = true;
            }
            if (typeof config.numOutputs !== 'undefined') {
                this._numOutputs = config.numOutputs;
            }
            if (this.viewer) {
                this.viewer.forceRedraw();
            }
        }

        // =====================================================================
        //  WEBGL INITIALIZATION (_initWebGL)
        //  Acquires a WebGL context on the main canvas, compiles and links the
        //  shader program, caches uniform/attribute locations, creates the
        //  full-screen quad geometry buffers, and initialises 4 layer textures.
        // =====================================================================

        // ---- Internal: WebGL init ----

        _initWebGL() {
            // Get a WebGL context on a hidden off-screen canvas.
            // We render to this, then draw the result to the visible canvas.
            // Actually, we can get WebGL directly on the main canvas.
            var gl = this.canvas.getContext('webgl2', { premultipliedAlpha: false });
            if (!gl) {
                gl = this.canvas.getContext('webgl', { premultipliedAlpha: false });
            }
            if (!gl) {
                gl = this.canvas.getContext('experimental-webgl', { premultipliedAlpha: false });
            }
            if (!gl) {
                $.console.error('[HyperBlendWebGLDrawer] Could not create WebGL context.');
                return;
            }
            this._gl = gl;

            // Compile shaders
            var vertShader = this._compileShader(gl, gl.VERTEX_SHADER, VERTEX_SHADER_SRC);
            var fragShader = this._compileShader(gl, gl.FRAGMENT_SHADER, FRAGMENT_SHADER_SRC);
            if (!vertShader || !fragShader) {
                $.console.error('[HyperBlendWebGLDrawer] Shader compilation failed.');
                this._gl = null;
                return;
            }

            // Link program
            var program = gl.createProgram();
            gl.attachShader(program, vertShader);
            gl.attachShader(program, fragShader);
            gl.linkProgram(program);
            if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
                $.console.error('[HyperBlendWebGLDrawer] Program link failed:', gl.getProgramInfoLog(program));
                this._gl = null;
                return;
            }
            this._program = program;

            // Get uniform locations
            this._uniformLocations = {
                uLayer0: gl.getUniformLocation(program, 'uLayer0'),
                uLayer1: gl.getUniformLocation(program, 'uLayer1'),
                uLayer2: gl.getUniformLocation(program, 'uLayer2'),
                uLayer3: gl.getUniformLocation(program, 'uLayer3'),
                uChannelColor: gl.getUniformLocation(program, 'uChannelColor[0]'),
                uChannelGain: gl.getUniformLocation(program, 'uChannelGain[0]'),
                uChannelEnabled: gl.getUniformLocation(program, 'uChannelEnabled[0]'),
                uToneMode: gl.getUniformLocation(program, 'uToneMode'),
                uUnmixEnabled: gl.getUniformLocation(program, 'uUnmixEnabled'),
                uUnmixMatrix: gl.getUniformLocation(program, 'uUnmixMatrix[0]')
            };

            // Validate uniform locations
            var uloc = this._uniformLocations;
            var nullUniforms = Object.keys(uloc).filter(function(k) { return uloc[k] === null; });
            if (nullUniforms.length > 0) {
                $.console.warn('[HyperBlendWebGLDrawer] Null uniform locations:', nullUniforms.join(', '));
            }

            // Static sampler bindings (never change)
            gl.useProgram(program);
            gl.uniform1i(this._uniformLocations.uLayer0, 0);
            gl.uniform1i(this._uniformLocations.uLayer1, 1);
            gl.uniform1i(this._uniformLocations.uLayer2, 2);
            gl.uniform1i(this._uniformLocations.uLayer3, 3);

            // Cache attribute locations (avoids per-frame getAttribLocation calls)
            this._attribLocations = {
                aPosition: gl.getAttribLocation(program, 'aPosition'),
                aTexCoord: gl.getAttribLocation(program, 'aTexCoord')
            };
            if (this._attribLocations.aPosition < 0 || this._attribLocations.aTexCoord < 0) {
                $.console.warn('[HyperBlendWebGLDrawer] Attribute location not found: aPosition=' +
                    this._attribLocations.aPosition + ' aTexCoord=' + this._attribLocations.aTexCoord);
            }

            // Full-screen quad geometry
            // Positions: clip space (-1,-1) to (1,1)
            this._posBuffer = gl.createBuffer();
            gl.bindBuffer(gl.ARRAY_BUFFER, this._posBuffer);
            gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
                -1, -1,
                 1, -1,
                -1,  1,
                 1,  1
            ]), gl.STATIC_DRAW);

            // Tex coords for Pass 2 (FBO read): standard GL coords, no Y-flip
            // FBO textures are already in GL coordinate space (Y-up)
            this._texBufferFBO = gl.createBuffer();
            gl.bindBuffer(gl.ARRAY_BUFFER, this._texBufferFBO);
            gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
                0, 0,
                1, 0,
                0, 1,
                1, 1
            ]), gl.STATIC_DRAW);

            // ---- Per-layer FBOs for tile compositing ----
            for (var i = 0; i < 4; i++) {
                var fboTex = gl.createTexture();
                gl.activeTexture(gl.TEXTURE0 + i);
                gl.bindTexture(gl.TEXTURE_2D, fboTex);
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
                gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE,
                    new Uint8Array([0, 0, 0, 0]));

                var fbo = gl.createFramebuffer();
                gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
                gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, fboTex, 0);
                var status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
                if (status !== gl.FRAMEBUFFER_COMPLETE) {
                    $.console.error('[HyperBlendWebGLDrawer] Layer FBO ' + i + ' incomplete: 0x' + status.toString(16));
                }
                gl.bindFramebuffer(gl.FRAMEBUFFER, null);

                this._layerFBOs.push(fbo);
                this._layerFBOTextures.push(fboTex);
            }

            // ---- Blit shader for drawing tiles into layer FBOs ----
            var blitVertShader = this._compileShader(gl, gl.VERTEX_SHADER, VERTEX_SHADER_SRC);
            var blitFragShader = this._compileShader(gl, gl.FRAGMENT_SHADER, BLIT_FRAGMENT_SRC);
            if (blitVertShader && blitFragShader) {
                var blitProgram = gl.createProgram();
                gl.attachShader(blitProgram, blitVertShader);
                gl.attachShader(blitProgram, blitFragShader);
                gl.linkProgram(blitProgram);
                if (gl.getProgramParameter(blitProgram, gl.LINK_STATUS)) {
                    this._blitProgram = blitProgram;
                    this._blitUniforms = {
                        uTile: gl.getUniformLocation(blitProgram, 'uTile')
                    };
                    this._blitAttribs = {
                        aPosition: gl.getAttribLocation(blitProgram, 'aPosition'),
                        aTexCoord: gl.getAttribLocation(blitProgram, 'aTexCoord')
                    };
                }
                gl.useProgram(blitProgram);
                gl.uniform1i(this._blitUniforms.uTile, 5);
                gl.deleteShader(blitVertShader);
                gl.deleteShader(blitFragShader);
            }

            // Tile textures are now cached per-tile in this._tileCache (created on demand)

            // Dynamic position buffer for tile quads (updated per tile)
            this._blitPosBuffer = gl.createBuffer();
            gl.bindBuffer(gl.ARRAY_BUFFER, this._blitPosBuffer);
            gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(8), gl.DYNAMIC_DRAW);

            // Cleanup shader objects (they are attached to program)
            gl.deleteShader(vertShader);
            gl.deleteShader(fragShader);

            // ---- Beer's law post-process shader program ----
            var beersVertShader = this._compileShader(gl, gl.VERTEX_SHADER, VERTEX_SHADER_SRC);
            var beersFragShader = this._compileShader(gl, gl.FRAGMENT_SHADER, BEERS_LAW_FRAGMENT_SRC);
            if (beersVertShader && beersFragShader) {
                var beersProgram = gl.createProgram();
                gl.attachShader(beersProgram, beersVertShader);
                gl.attachShader(beersProgram, beersFragShader);
                gl.linkProgram(beersProgram);
                if (gl.getProgramParameter(beersProgram, gl.LINK_STATUS)) {
                    this._postProcessProgram = beersProgram;
                    this._postProcessUniforms = {
                        uHyperBlendOutput: gl.getUniformLocation(beersProgram, 'uHyperBlendOutput'),
                        u_k: gl.getUniformLocation(beersProgram, 'u_k'),
                        u_nucRGB: gl.getUniformLocation(beersProgram, 'u_nucRGB'),
                        u_strRGB: gl.getUniformLocation(beersProgram, 'u_strRGB'),
                        u_colRGB: gl.getUniformLocation(beersProgram, 'u_colRGB'),
                        u_unmix: gl.getUniformLocation(beersProgram, 'u_unmix'),
                        u_nucGain: gl.getUniformLocation(beersProgram, 'u_nucGain'),
                        u_strGain: gl.getUniformLocation(beersProgram, 'u_strGain'),
                        u_colGain: gl.getUniformLocation(beersProgram, 'u_colGain'),
                        u_mode: gl.getUniformLocation(beersProgram, 'u_mode')
                    };
                    this._postProcessAttribs = {
                        aPosition: gl.getAttribLocation(beersProgram, 'aPosition'),
                        aTexCoord: gl.getAttribLocation(beersProgram, 'aTexCoord')
                    };
                    gl.useProgram(beersProgram);
                    gl.uniform1i(this._postProcessUniforms.uHyperBlendOutput, 4);
                } else {
                    $.console.error('[HyperBlendWebGLDrawer] Beer\'s law program link failed:',
                        gl.getProgramInfoLog(beersProgram));
                }
                gl.deleteShader(beersVertShader);
                gl.deleteShader(beersFragShader);
            } else {
                $.console.warn('[HyperBlendWebGLDrawer] Beer\'s law shader compilation failed');
                if (beersVertShader) gl.deleteShader(beersVertShader);
                if (beersFragShader) gl.deleteShader(beersFragShader);
            }

            // ---- FBO for post-process intermediate render target ----
            this._fbo = gl.createFramebuffer();
            this._fboTexture = gl.createTexture();
            gl.bindTexture(gl.TEXTURE_2D, this._fboTexture);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
            gl.bindFramebuffer(gl.FRAMEBUFFER, this._fbo);
            gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this._fboTexture, 0);
            var ppFboStatus = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
            if (ppFboStatus !== gl.FRAMEBUFFER_COMPLETE) {
                $.console.error('[HyperBlendWebGLDrawer] Post-process FBO incomplete: 0x' + ppFboStatus.toString(16));
            }
            gl.bindFramebuffer(gl.FRAMEBUFFER, null);
            gl.bindTexture(gl.TEXTURE_2D, null);
        }

        // =====================================================================
        //  INTERNAL HELPERS
        //  _compileShader        — compile a single GLSL shader, return handle or null
        //  _getActiveLayers      — resolve the 4 TiledImage layers for the current
        //                          z-level from window.allLayers or the draw call
        //  _resizeLayerFBOs      — resize per-layer FBO textures to match canvas
        //  _compositeTilesToFBO  — blit tiles for one TiledImage into its FBO
        //                          (bypasses 2D canvas premultiplied alpha)
        // =====================================================================

        _compileShader(gl, type, source) {
            var shader = gl.createShader(type);
            gl.shaderSource(shader, source);
            gl.compileShader(shader);
            if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
                $.console.error('[HyperBlendWebGLDrawer] Shader compile error:',
                    gl.getShaderInfoLog(shader), '\nSource:\n', source);
                gl.deleteShader(shader);
                return null;
            }
            return shader;
        }

        // ---- Internal: resize FBO texture to match canvas ----

        _resizeFBO(width, height) {
            if (width === this._fboWidth && height === this._fboHeight) return;
            var gl = this._gl;
            if (!gl || !this._fboTexture) return;
            gl.bindTexture(gl.TEXTURE_2D, this._fboTexture);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
            gl.bindTexture(gl.TEXTURE_2D, null);
            this._fboWidth = width;
            this._fboHeight = height;
        }

        // ---- Internal: determine active layers ----

        _getActiveLayers(tiledImages) {
            // Try to use the global allLayers array from zstackHyper.html
            var zIdx = (typeof window.currentZIndex !== 'undefined') ? window.currentZIndex : 0;
            var layerCount = (typeof window.imagesPerZ !== 'undefined') ? window.imagesPerZ : 4;
            if (typeof window.allLayers !== 'undefined' && window.allLayers[zIdx]) {
                var layers = [];
                for (var i = 0; i < layerCount; i++) {
                    layers.push(window.allLayers[zIdx][i] || null);
                }
                return layers;
            }
            // Fallback: use first layerCount tiledImages from the draw call
            var result = [];
            for (var j = 0; j < layerCount; j++) {
                result.push(j < tiledImages.length ? tiledImages[j] : null);
            }
            return result;
        }

        // ---- Internal: resize all layer FBO textures to match canvas ----
        _resizeLayerFBOs(width, height) {
            if (width === this._layerFBOWidth && height === this._layerFBOHeight) return;
            var gl = this._gl;
            for (var i = 0; i < this._layerFBOTextures.length; i++) {
                gl.bindTexture(gl.TEXTURE_2D, this._layerFBOTextures[i]);
                gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
            }
            gl.bindTexture(gl.TEXTURE_2D, null);
            this._layerFBOWidth = width;
            this._layerFBOHeight = height;
        }

        // ---- Internal: composite tiles for one layer into its FBO ----
        // Uploads each tile directly to WebGL with UNPACK_PREMULTIPLY_ALPHA_WEBGL=false,
        // preserving all 4 RGBA channels perfectly.  Replaces the old canvas-based path.
        _compositeTilesToFBO(tiledImage, tilesToDraw, layerIndex, canvasW, canvasH, clearFBO) {
            var gl = this._gl;
            var viewport = this.viewport;

            // Bind this layer's FBO
            gl.bindFramebuffer(gl.FRAMEBUFFER, this._layerFBOs[layerIndex]);
            gl.viewport(0, 0, canvasW, canvasH);
            // Only clear when forced (viewport/z/size change). During progressive tile
            // loading, skip clear so old tile content remains visible in gaps where
            // new tiles haven't loaded yet — prevents black flicker during transitions.
            if (clearFBO !== false) {
                gl.clearColor(0, 0, 0, 0);
                gl.clear(gl.COLOR_BUFFER_BIT);
            }

            gl.useProgram(this._blitProgram);
            gl.activeTexture(gl.TEXTURE5);  // temp unit (0-3 = layers, 4 = post-process FBO)

            // Blit tex coords: full tile (static buffer)
            // Reuse _texBufferFBO (0,0 → 1,1) for tile texture sampling
            gl.bindBuffer(gl.ARRAY_BUFFER, this._texBufferFBO);
            gl.enableVertexAttribArray(this._blitAttribs.aTexCoord);
            gl.vertexAttribPointer(this._blitAttribs.aTexCoord, 2, gl.FLOAT, false, 0, 0);

            // Position buffer (dynamic, updated per tile)
            gl.bindBuffer(gl.ARRAY_BUFFER, this._blitPosBuffer);
            gl.enableVertexAttribArray(this._blitAttribs.aPosition);
            gl.vertexAttribPointer(this._blitAttribs.aPosition, 2, gl.FLOAT, false, 0, 0);

            // Pre-compute rotation
            var rotation = viewport.getRotation(true);
            var flipped = viewport.getFlip();
            var cos_r = 1, sin_r = 0;
            if (rotation !== 0) {
                var rad = rotation * Math.PI / 180;
                cos_r = Math.cos(rad);
                sin_r = Math.sin(rad);
            }
            var tiRotation = tiledImage.getRotation(true);
            var cos_ti = 1, sin_ti = 0;
            if (tiRotation !== 0) {
                var tiRad = tiRotation * Math.PI / 180;
                cos_ti = Math.cos(tiRad);
                sin_ti = Math.sin(tiRad);
            }

            var pdr = $.pixelDensityRatio;

            // Pre-compute TiledImage rotation point (same for all tiles in this layer)
            var rpx = 0, rpy = 0;
            if (tiRotation !== 0) {
                var rotPoint = tiledImage._getRotationPoint(true);
                var rpPixel = viewport.pixelFromPointNoRotate(rotPoint, true);
                rpx = rpPixel.x * pdr;
                rpy = rpPixel.y * pdr;
            }

            for (var i = 0; i < tilesToDraw.length; i++) {
                var tile = tilesToDraw[i].tile;
                if (!tile.loaded) continue;

                var rendered = this.getDataToDraw(tile);
                if (!rendered) continue;

                var sourceEl = rendered.canvas || rendered;
                if (!(sourceEl instanceof HTMLCanvasElement ||
                      sourceEl instanceof HTMLImageElement ||
                      sourceEl instanceof Image)) continue;

                // Cache-aware tile upload: skip CPU→GPU transfer for cached tiles
                var cacheKey = tile.cacheKey;
                var cached = this._tileCache.get(cacheKey);
                if (cached) {
                    // Cache hit — bind existing GPU texture (GPU→GPU, fast)
                    gl.bindTexture(gl.TEXTURE_2D, cached.texture);
                    cached.lastUsed = this._frameCount;
                } else {
                    // Cache miss — upload tile to GPU
                    var tex = gl.createTexture();
                    gl.bindTexture(gl.TEXTURE_2D, tex);
                    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
                    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
                    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
                    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
                    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, sourceEl);
                    this._tileCache.set(cacheKey, { texture: tex, lastUsed: this._frameCount });
                }

                // Tile position in viewport pixels — round to integers to prevent
                // sub-pixel gaps/overlaps between adjacent tiles (jitter fix)
                var px = Math.round(tile.position.x * pdr);
                var py = Math.round(tile.position.y * pdr);
                var pw = Math.round(tile.size.x * pdr);
                var ph = Math.round(tile.size.y * pdr);

                // 4 corners: TL, TR, BL, BR in pixel space (pre-allocated)
                var corners = this._corners;
                corners[0] = px;       corners[1] = py;        // TL
                corners[2] = px + pw;  corners[3] = py;        // TR
                corners[4] = px;       corners[5] = py + ph;   // BL
                corners[6] = px + pw;  corners[7] = py + ph;   // BR

                // Tile flip (mirror X around tile center)
                if (tile.flipped) {
                    var cx = px + pw / 2;
                    corners[0] = 2 * cx - corners[0];  // TL.x
                    corners[2] = 2 * cx - corners[2];  // TR.x
                    corners[4] = 2 * cx - corners[4];  // BL.x
                    corners[6] = 2 * cx - corners[6];  // BR.x
                }

                // TiledImage rotation around rotation point (pre-computed above loop)
                if (tiRotation !== 0) {
                    for (var c = 0; c < 8; c += 2) {
                        var dx = corners[c] - rpx;
                        var dy = corners[c+1] - rpy;
                        corners[c]   = rpx + dx * cos_ti - dy * sin_ti;
                        corners[c+1] = rpy + dx * sin_ti + dy * cos_ti;
                    }
                }

                // Viewport rotation around canvas center
                if (rotation !== 0) {
                    var ccx = canvasW / 2;
                    var ccy = canvasH / 2;
                    for (var c = 0; c < 8; c += 2) {
                        var dx = corners[c] - ccx;
                        var dy = corners[c+1] - ccy;
                        corners[c]   = ccx + dx * cos_r - dy * sin_r;
                        corners[c+1] = ccy + dx * sin_r + dy * cos_r;
                    }
                }

                // Viewport flip (mirror X)
                if (flipped) {
                    for (var c = 0; c < 8; c += 2) {
                        corners[c] = canvasW - corners[c];
                    }
                }

                // Reorder: corners is [TL, TR, BL, BR] → clipData needs [BL, BR, TL, TR]
                // to match _texBufferFBO vertex order (0,0  1,0  0,1  1,1)
                // Convert to clip space: pixel(0,0)=top → clip(-1,1)=top
                var cd = this._clipData;  // pre-allocated Float32Array(8)
                // BL
                cd[0] = (corners[4] / canvasW) * 2.0 - 1.0;
                cd[1] = 1.0 - (corners[5] / canvasH) * 2.0;
                // BR
                cd[2] = (corners[6] / canvasW) * 2.0 - 1.0;
                cd[3] = 1.0 - (corners[7] / canvasH) * 2.0;
                // TL
                cd[4] = (corners[0] / canvasW) * 2.0 - 1.0;
                cd[5] = 1.0 - (corners[1] / canvasH) * 2.0;
                // TR
                cd[6] = (corners[2] / canvasW) * 2.0 - 1.0;
                cd[7] = 1.0 - (corners[3] / canvasH) * 2.0;

                // Upload position data and draw
                gl.bufferSubData(gl.ARRAY_BUFFER, 0, cd);
                gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
            }

            gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        }

        // ---- DrawerBase helpers we inherit ----
        // _calculateCanvasSize, viewportToDrawerRectangle, viewportCoordToDrawerCoord
        // are all defined on DrawerBase and work as-is.

        // ---- Internal: evict stale tile textures from GPU cache ----
        _evictStaleTiles(viewportChanged) {
            // Skip eviction during active viewport changes (zoom/pan) to prevent
            // thrashing tiles that were just uploaded for the new viewport
            if (viewportChanged) {
                return;  // viewport is changing — defer eviction
            }

            var gl = this._gl;
            var maxAge = this._evictionMaxAge;
            var frame = this._frameCount;
            var cap = this._tileCacheCap;

            // Age-based eviction (collect keys first, then delete)
            var staleKeys = [];
            this._tileCache.forEach(function(entry, key) {
                if (frame - entry.lastUsed > maxAge) {
                    staleKeys.push(key);
                }
            });
            for (var si = 0; si < staleKeys.length; si++) {
                var stale = this._tileCache.get(staleKeys[si]);
                if (stale) gl.deleteTexture(stale.texture);
                this._tileCache.delete(staleKeys[si]);
            }

            // Cap-based eviction (if still over limit, evict oldest)
            if (this._tileCache.size > cap) {
                var entries = [];
                this._tileCache.forEach(function(entry, key) {
                    entries.push({ key: key, lastUsed: entry.lastUsed });
                });
                entries.sort(function(a, b) { return a.lastUsed - b.lastUsed; });
                var toRemove = this._tileCache.size - cap;
                for (var i = 0; i < toRemove; i++) {
                    var e = this._tileCache.get(entries[i].key);
                    if (e) gl.deleteTexture(e.texture);
                    this._tileCache.delete(entries[i].key);
                }
            }
        }

        /**
         * Draw a debugging rectangle (optional, for compatibility)
         */
        drawDebuggingRect(rect) {
            // No-op: debugging is not supported in the WebGL blending drawer
        }
    };

    // =========================================================================
    //  OSD REGISTRATION
    //  Attach to the OpenSeadragon namespace so that OSD's determineDrawer()
    //  finds this class when the user specifies { drawer: 'hyperblend-webgl' }.
    // =========================================================================

    // -----------------------------------------------------------------------
    // Register with OpenSeadragon so that drawer: 'hyperblend-webgl' works.
    // determineDrawer() iterates OpenSeadragon.* looking for DrawerBase
    // subclasses whose getType() matches the requested string.
    // -----------------------------------------------------------------------
    $.HyperBlendWebGLDrawer = HyperBlendWebGLDrawer;

    // Also expose globally for convenience
    window.HyperBlendWebGLDrawer = HyperBlendWebGLDrawer;

}(OpenSeadragon));
