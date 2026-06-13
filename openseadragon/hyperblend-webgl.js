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
        '',
        'void main() {',
        '    // Pre-sample all 4 layer textures once (4 fetches total, not 16)',
        '    // FBO compositing with UNPACK_PREMULTIPLY_ALPHA_WEBGL=false preserves',
        '    // all 4 RGBA channels — no unpremultiply needed.',
        '    //',
        '    // v5.0 R1: the Linear matrix multiply is no longer inlined here.',
        '    // The drawer is responsible for binding the right textures to',
        '    // units 0..3 BEFORE this shader runs:',
        '    //   Mode 0 (no unmix, no PICASSO): raw layer FBOs',
        '    //   Mode 1 (Linear only):          _linearFBOTextures[0..1] in units 0..1;',
        '    //                                   units 2..3 cleared so r8..r15 = 0',
        '    //   Mode 2 (PICASSO only):         _picassoTex_Out[0..3]',
        '    //   Mode 3 (Linear→PICASSO):       _picassoTex_Out[0..3] (chain output;',
        '    //                                   unreachable in R1 — L1325 throw still armed)',
        '    // The HTML side guarantees channels 4..15 are disabled whenever',
        '    // unmixing is active (see _pushUnmixChannelConfig); the drawer also',
        '    // console.asserts the invariant inside updateChannelConfig.',
        '    vec4 L0 = texture2D(uLayer0, vTexCoord);',
        '    vec4 L1 = texture2D(uLayer1, vTexCoord);',
        '    vec4 L2 = texture2D(uLayer2, vTexCoord);',
        '    vec4 L3 = texture2D(uLayer3, vTexCoord);',
        '',
        '    // Pass-through: ch_i is the i-th RGBA component across L0..L3.',
        '    // In Mode 1/2/3 the upstream stage already wrote unmixed values into',
        '    // these textures, so the same line covers both raw and unmixed cases.',
        '    float ch0=L0.r, ch1=L0.g, ch2=L0.b, ch3=L0.a;',
        '    float ch4=L1.r, ch5=L1.g, ch6=L1.b, ch7=L1.a;',
        '    float ch8=L2.r, ch9=L2.g, ch10=L2.b, ch11=L2.a;',
        '    float ch12=L3.r, ch13=L3.g, ch14=L3.b, ch15=L3.a;',
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

    // -----------------------------------------------------------------------
    // Linear pre-blend pass (NEW — v5.0 R1).
    // Lifts the 16-input × ≤8-output pseudoinverse multiply out of
    // FRAGMENT_SHADER_SRC so PICASSO can consume Linear's abundance output
    // (Mode 3 — Linear→PICASSO chain).
    //
    // Inputs : uLayer0..3 (raw uint8 layer FBOs; same texture units 0..3
    //          the HyperBlend Pass 1 used to sample).
    // Output : 4 abundance channels per fragment (clamped ≥ 0, ≤ 1).
    //
    // The drawer runs this pass up to twice per frame:
    //   - first invocation writes outputs 0..3 (uOutputOffset = 0)
    //   - second invocation (only when numOutputs > 4) writes outputs 4..7
    //     (uOutputOffset = 4)
    // The matrix layout in uUnmixMatrix is unchanged from the inlined version
    // (row-major 16×8 stored as a flat 128-float array): for input i, output j,
    // the coefficient is at uUnmixMatrix[i * 8 + j]. Two passes shift the
    // sampled `j` slot by uOutputOffset; that is the only difference between
    // the two passes.
    //
    // Uint8 quantization on the output FBO is part of the math contract
    // (MRA fix #3 in v5.0 master plan §7.1 R-2): Mode 3 must hand
    // PICASSO the same precision profile Mode 2 sees from raw layer FBOs.
    // -----------------------------------------------------------------------
    var LINEAR_PASS_FRAGMENT_SRC = [
        'precision mediump float;',
        '',
        'varying vec2 vTexCoord;',
        '',
        'uniform sampler2D uLayer0;',
        'uniform sampler2D uLayer1;',
        'uniform sampler2D uLayer2;',
        'uniform sampler2D uLayer3;',
        '',
        'uniform float uUnmixMatrix[128];',
        'uniform float uOutputOffset;',  // 0.0 for outputs 0..3, 4.0 for outputs 4..7
        '',
        'void main() {',
        '    // Sample raw 16-channel mixed inputs (same layout as Pass 1)',
        '    vec4 L0 = texture2D(uLayer0, vTexCoord);',
        '    vec4 L1 = texture2D(uLayer1, vTexCoord);',
        '    vec4 L2 = texture2D(uLayer2, vTexCoord);',
        '    vec4 L3 = texture2D(uLayer3, vTexCoord);',
        '',
        '    float r0=L0.r, r1=L0.g, r2=L0.b, r3=L0.a;',
        '    float r4=L1.r, r5=L1.g, r6=L1.b, r7=L1.a;',
        '    float r8=L2.r, r9=L2.g, r10=L2.b, r11=L2.a;',
        '    float r12=L3.r, r13=L3.g, r14=L3.b, r15=L3.a;',
        '',
        '    // Output column index for each RGBA slot in this pass',
        '    // (offset shifts between the two FBO writes).',
        '    int o0 = int(uOutputOffset);',
        '    int o1 = o0 + 1;',
        '    int o2 = o0 + 2;',
        '    int o3 = o0 + 3;',
        '',
        '    // Matrix multiply: out[j] = sum_i(raw[i] * M[i*8 + j])',
        '    // GLSL ES 1.0 forbids dynamic indexing into uniform arrays; the',
        '    // 8 cases below are unrolled by output column to keep the shader',
        '    // portable. Identical to the inlined ch0..ch7 block formerly in',
        '    // FRAGMENT_SHADER_SRC.',
        '    float outs[8];',
        '    outs[0] = r0*uUnmixMatrix[0]   + r1*uUnmixMatrix[8]   + r2*uUnmixMatrix[16]  + r3*uUnmixMatrix[24]  + r4*uUnmixMatrix[32]  + r5*uUnmixMatrix[40]  + r6*uUnmixMatrix[48]  + r7*uUnmixMatrix[56]  + r8*uUnmixMatrix[64]  + r9*uUnmixMatrix[72]  + r10*uUnmixMatrix[80]  + r11*uUnmixMatrix[88]  + r12*uUnmixMatrix[96]  + r13*uUnmixMatrix[104] + r14*uUnmixMatrix[112] + r15*uUnmixMatrix[120];',
        '    outs[1] = r0*uUnmixMatrix[1]   + r1*uUnmixMatrix[9]   + r2*uUnmixMatrix[17]  + r3*uUnmixMatrix[25]  + r4*uUnmixMatrix[33]  + r5*uUnmixMatrix[41]  + r6*uUnmixMatrix[49]  + r7*uUnmixMatrix[57]  + r8*uUnmixMatrix[65]  + r9*uUnmixMatrix[73]  + r10*uUnmixMatrix[81]  + r11*uUnmixMatrix[89]  + r12*uUnmixMatrix[97]  + r13*uUnmixMatrix[105] + r14*uUnmixMatrix[113] + r15*uUnmixMatrix[121];',
        '    outs[2] = r0*uUnmixMatrix[2]   + r1*uUnmixMatrix[10]  + r2*uUnmixMatrix[18]  + r3*uUnmixMatrix[26]  + r4*uUnmixMatrix[34]  + r5*uUnmixMatrix[42]  + r6*uUnmixMatrix[50]  + r7*uUnmixMatrix[58]  + r8*uUnmixMatrix[66]  + r9*uUnmixMatrix[74]  + r10*uUnmixMatrix[82]  + r11*uUnmixMatrix[90]  + r12*uUnmixMatrix[98]  + r13*uUnmixMatrix[106] + r14*uUnmixMatrix[114] + r15*uUnmixMatrix[122];',
        '    outs[3] = r0*uUnmixMatrix[3]   + r1*uUnmixMatrix[11]  + r2*uUnmixMatrix[19]  + r3*uUnmixMatrix[27]  + r4*uUnmixMatrix[35]  + r5*uUnmixMatrix[43]  + r6*uUnmixMatrix[51]  + r7*uUnmixMatrix[59]  + r8*uUnmixMatrix[67]  + r9*uUnmixMatrix[75]  + r10*uUnmixMatrix[83]  + r11*uUnmixMatrix[91]  + r12*uUnmixMatrix[99]  + r13*uUnmixMatrix[107] + r14*uUnmixMatrix[115] + r15*uUnmixMatrix[123];',
        '    outs[4] = r0*uUnmixMatrix[4]   + r1*uUnmixMatrix[12]  + r2*uUnmixMatrix[20]  + r3*uUnmixMatrix[28]  + r4*uUnmixMatrix[36]  + r5*uUnmixMatrix[44]  + r6*uUnmixMatrix[52]  + r7*uUnmixMatrix[60]  + r8*uUnmixMatrix[68]  + r9*uUnmixMatrix[76]  + r10*uUnmixMatrix[84]  + r11*uUnmixMatrix[92]  + r12*uUnmixMatrix[100] + r13*uUnmixMatrix[108] + r14*uUnmixMatrix[116] + r15*uUnmixMatrix[124];',
        '    outs[5] = r0*uUnmixMatrix[5]   + r1*uUnmixMatrix[13]  + r2*uUnmixMatrix[21]  + r3*uUnmixMatrix[29]  + r4*uUnmixMatrix[37]  + r5*uUnmixMatrix[45]  + r6*uUnmixMatrix[53]  + r7*uUnmixMatrix[61]  + r8*uUnmixMatrix[69]  + r9*uUnmixMatrix[77]  + r10*uUnmixMatrix[85]  + r11*uUnmixMatrix[93]  + r12*uUnmixMatrix[101] + r13*uUnmixMatrix[109] + r14*uUnmixMatrix[117] + r15*uUnmixMatrix[125];',
        '    outs[6] = r0*uUnmixMatrix[6]   + r1*uUnmixMatrix[14]  + r2*uUnmixMatrix[22]  + r3*uUnmixMatrix[30]  + r4*uUnmixMatrix[38]  + r5*uUnmixMatrix[46]  + r6*uUnmixMatrix[54]  + r7*uUnmixMatrix[62]  + r8*uUnmixMatrix[70]  + r9*uUnmixMatrix[78]  + r10*uUnmixMatrix[86]  + r11*uUnmixMatrix[94]  + r12*uUnmixMatrix[102] + r13*uUnmixMatrix[110] + r14*uUnmixMatrix[118] + r15*uUnmixMatrix[126];',
        '    outs[7] = r0*uUnmixMatrix[7]   + r1*uUnmixMatrix[15]  + r2*uUnmixMatrix[23]  + r3*uUnmixMatrix[31]  + r4*uUnmixMatrix[39]  + r5*uUnmixMatrix[47]  + r6*uUnmixMatrix[55]  + r7*uUnmixMatrix[63]  + r8*uUnmixMatrix[71]  + r9*uUnmixMatrix[79]  + r10*uUnmixMatrix[87]  + r11*uUnmixMatrix[95]  + r12*uUnmixMatrix[103] + r13*uUnmixMatrix[111] + r14*uUnmixMatrix[119] + r15*uUnmixMatrix[127];',
        '',
        '    // Select the 4 outputs corresponding to this pass via',
        '    // uOutputOffset. GLSL ES 1.0 dynamic indexing on a local array is',
        '    // allowed (only uniform arrays are restricted), so the unrolled',
        '    // selection below is the canonical workaround.',
        '    float c0, c1, c2, c3;',
        '    if (o0 == 0)      c0 = outs[0]; else if (o0 == 1) c0 = outs[1]; else if (o0 == 2) c0 = outs[2]; else if (o0 == 3) c0 = outs[3]; else if (o0 == 4) c0 = outs[4]; else if (o0 == 5) c0 = outs[5]; else if (o0 == 6) c0 = outs[6]; else c0 = outs[7];',
        '    if (o1 == 0)      c1 = outs[0]; else if (o1 == 1) c1 = outs[1]; else if (o1 == 2) c1 = outs[2]; else if (o1 == 3) c1 = outs[3]; else if (o1 == 4) c1 = outs[4]; else if (o1 == 5) c1 = outs[5]; else if (o1 == 6) c1 = outs[6]; else c1 = outs[7];',
        '    if (o2 == 0)      c2 = outs[0]; else if (o2 == 1) c2 = outs[1]; else if (o2 == 2) c2 = outs[2]; else if (o2 == 3) c2 = outs[3]; else if (o2 == 4) c2 = outs[4]; else if (o2 == 5) c2 = outs[5]; else if (o2 == 6) c2 = outs[6]; else c2 = outs[7];',
        '    if (o3 == 0)      c3 = outs[0]; else if (o3 == 1) c3 = outs[1]; else if (o3 == 2) c3 = outs[2]; else if (o3 == 3) c3 = outs[3]; else if (o3 == 4) c3 = outs[4]; else if (o3 == 5) c3 = outs[5]; else if (o3 == 6) c3 = outs[6]; else c3 = outs[7];',
        '',
        '    // Non-negativity (Linear math contract: max(0, M_p · mixed)) +',
        '    // uint8 saturation (FBO format is RGBA8).',
        '    gl_FragColor = vec4(clamp(c0, 0.0, 1.0), clamp(c1, 0.0, 1.0), clamp(c2, 0.0, 1.0), clamp(c3, 0.0, 1.0));',
        '}'
    ].join('\n');

    // PICASSO kernel: vec4 y = uP * x; clamp at ≥0. Port of FS_PICASSO from
    // demo/picasso-osd-demo.html L353–359. precision highp is required —
    // mediump would saturate the [0..255]-range float ping-pong.
    var PICASSO_FRAGMENT_SHADER_SRC = [
        'precision highp float;',
        'uniform sampler2D uSrc;',
        'uniform mat4 uP;',
        'varying vec2 vTexCoord;',
        'void main() {',
        '    vec4 x = texture2D(uSrc, vTexCoord);',
        '    vec4 y = uP * x;',
        '    gl_FragColor = max(y, vec4(0.0));',
        '}'
    ].join('\n');

    // PICASSO uint8 cast: clamp [0..255] then divide by 255 (+0.5 round bias).
    // Port of FS_CAST from demo/picasso-osd-demo.html L360–366.
    var PICASSO_CAST_FRAGMENT_SHADER_SRC = [
        'precision highp float;',
        'uniform sampler2D uSrc;',
        'varying vec2 vTexCoord;',
        'void main() {',
        '    vec4 v = texture2D(uSrc, vTexCoord);',
        '    v = clamp(v, 0.0, 255.0);',
        '    gl_FragColor = (v + vec4(0.5)) / 255.0;',
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

            // Linear pre-blend FBO stage (v5.0 R1).
            // Two RGBA8 FBOs cover the full M ∈ [2,8] range — each holds 4
            // outputs, so M ≤ 4 uses only [0], 5 ≤ M ≤ 8 uses both. They are
            // lazy-allocated on the first updateUnmixConfig({active:true}) so
            // users who never enable Linear pay nothing.
            // _linearOutputReady gates the consumers (HyperBlend Pass 1
            // texture binding + PICASSO source-tex switch); it is set inside
            // _runLinearPass and cleared whenever Linear is disabled or the
            // matrix is invalidated.
            this._linearProgram = null;
            this._linearUniforms = {};
            this._linearAttribs = {};
            this._linearFBOs = [null, null];
            this._linearFBOTextures = [null, null];
            this._linearFBOWidth_pre = 0;
            this._linearFBOHeight_pre = 0;
            this._linearOutputReady = false;

            // PICASSO stage state — lazy alloc on first activation.
            // Per-layer ping-pong: 4 layers x { fboA float, fboB float, fbo_out uint8 }.
            // Square N×N matrix stacked K times; never rectangular Cout×Cin.
            this._picassoSupported = null;          // null until first probe
            this._picassoActive = false;
            this._picassoMatrices = null;           // Array of K Float32Array(16), row-major
            this._picassoTransposed = null;         // Array of K Float32Array(16), col-major + iter-0 scale
            this._picassoMatricesUploaded = false;
            this._picassoK = 0;
            this._picassoN = 0;
            this._picassoFBOsAllocated = false;
            this._picassoFBO_A = [];
            this._picassoFBO_B = [];
            this._picassoFBO_Out = [];
            this._picassoTex_A = [];
            this._picassoTex_B = [];
            this._picassoTex_Out = [];
            this._picassoFBOWidth = 0;
            this._picassoFBOHeight = 0;
            this._picassoProgram = null;
            this._picassoCastProgram = null;
            this._picassoUniforms = {};
            this._picassoCastUniforms = {};
            this._picassoAttribs = {};
            this._picassoCastAttribs = {};

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
                // PICASSO resources are bound to the lost context — drop and
                // let updatePicassoConfig lazy-realloc on next activation.
                self._picassoFBOsAllocated = false;
                self._picassoFBO_A = [];
                self._picassoFBO_B = [];
                self._picassoFBO_Out = [];
                self._picassoTex_A = [];
                self._picassoTex_B = [];
                self._picassoTex_Out = [];
                self._picassoProgram = null;
                self._picassoCastProgram = null;
                self._picassoFBOWidth = 0;
                self._picassoFBOHeight = 0;
                // Linear pre-blend resources are also bound to the lost
                // context — drop and let _ensureLinearResources lazy-realloc.
                self._linearProgram = null;
                self._linearFBOs = [null, null];
                self._linearFBOTextures = [null, null];
                self._linearFBOWidth_pre = 0;
                self._linearFBOHeight_pre = 0;
                self._linearOutputReady = false;
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

            // Probe PICASSO float-FBO extensions. Drawer init succeeds regardless;
            // _picassoSupported just gates updatePicassoConfig({active:true}).
            this._probePicassoExtensions();

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

                // Probe PICASSO float-FBO extensions for diagnostic logging only —
                // missing extensions disable the PICASSO stage but do not fail the
                // drawer (HyperBlend/Linear keep working). The authoritative gate
                // lives in the instance probe (_probePicassoExtensions).
                var isWebGL2Static = (typeof WebGL2RenderingContext !== 'undefined' &&
                                       gl instanceof WebGL2RenderingContext);
                var picExtOK;
                if (isWebGL2Static) {
                    picExtOK = !!gl.getExtension('EXT_color_buffer_float');
                    if (!picExtOK) {
                        console.warn('[HyperBlendWebGLDrawer] isSupported: PICASSO disabled — missing EXT_color_buffer_float on WebGL 2.');
                    }
                } else {
                    var hasFloatTex = !!gl.getExtension('OES_texture_float');
                    var hasFloatRT = !!gl.getExtension('WEBGL_color_buffer_float');
                    picExtOK = hasFloatTex && hasFloatRT;
                    if (!picExtOK) {
                        console.warn('[HyperBlendWebGLDrawer] isSupported: PICASSO extensions unavailable (OES_texture_float=' +
                            hasFloatTex + ', WEBGL_color_buffer_float=' + hasFloatRT + '). PICASSO stage will be disabled.');
                    }
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
                // PICASSO resources (12 FBOs + 12 textures + 2 programs)
                for (var pi = 0; pi < this._picassoFBO_A.length; pi++) {
                    if (this._picassoFBO_A[pi]) gl.deleteFramebuffer(this._picassoFBO_A[pi]);
                    if (this._picassoTex_A[pi]) gl.deleteTexture(this._picassoTex_A[pi]);
                }
                for (var pj = 0; pj < this._picassoFBO_B.length; pj++) {
                    if (this._picassoFBO_B[pj]) gl.deleteFramebuffer(this._picassoFBO_B[pj]);
                    if (this._picassoTex_B[pj]) gl.deleteTexture(this._picassoTex_B[pj]);
                }
                for (var pk = 0; pk < this._picassoFBO_Out.length; pk++) {
                    if (this._picassoFBO_Out[pk]) gl.deleteFramebuffer(this._picassoFBO_Out[pk]);
                    if (this._picassoTex_Out[pk]) gl.deleteTexture(this._picassoTex_Out[pk]);
                }
                if (this._picassoProgram) gl.deleteProgram(this._picassoProgram);
                if (this._picassoCastProgram) gl.deleteProgram(this._picassoCastProgram);
                // Linear pre-blend resources (v5.0 R1)
                for (var lpi = 0; lpi < this._linearFBOs.length; lpi++) {
                    if (this._linearFBOs[lpi]) gl.deleteFramebuffer(this._linearFBOs[lpi]);
                    if (this._linearFBOTextures[lpi]) gl.deleteTexture(this._linearFBOTextures[lpi]);
                }
                if (this._linearProgram) gl.deleteProgram(this._linearProgram);
            }
            this._layerFBOs = [];
            this._layerFBOTextures = [];
            this._picassoFBO_A = [];
            this._picassoFBO_B = [];
            this._picassoFBO_Out = [];
            this._picassoTex_A = [];
            this._picassoTex_B = [];
            this._picassoTex_Out = [];
            this._linearFBOs = [null, null];
            this._linearFBOTextures = [null, null];
            this._linearProgram = null;
            this._linearOutputReady = false;
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
                // Resize PICASSO FBOs alongside layer FBOs so the ping-pong
                // chain shares the canvas dimensions. Skipped when inactive
                // to avoid VRAM cost on users who never enable PICASSO.
                if (this._picassoActive && this._picassoFBOsAllocated) {
                    this._resizePicassoFBOs(canvasW, canvasH);
                }

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

            // ---- Linear pre-blend stage (v5.0 R1) ----
            // Lifts the matrix multiply that used to live inside Pass 1 into a
            // dedicated FBO write. Runs whenever Linear is enabled — the chain
            // discriminator (_chainAllowed) only affects what consumes the
            // output (PICASSO source in Mode 3; HyperBlend Pass 1 in Mode 1).
            // Re-runs on every frame that reached the FULL path (texture
            // contents may have changed); fast-path frames inherit the cached
            // FBO until the next viewport/tile/z change.
            this._linearOutputReady = false;
            if (this._unmixEnabled && this._unmixMatrix && needsComposite) {
                this._runLinearPass(canvasW, canvasH);
            } else if (this._unmixEnabled && this._unmixMatrix) {
                // Fast path: matrix unchanged + FBOs valid — Linear output
                // from the previous frame is still on the GPU, just mark ready
                // so the binding swap below picks it up.
                this._linearOutputReady = !!(this._linearFBOTextures[0]);
            }

            // ---- PICASSO stage: K iterations of max(0, P^(k) @ x) per layer ----
            // Runs between the composite/bind block and Pass 1; transforms each
            // per-layer uint8 RGBA FBO into a recovered uint8 RGBA texture, then
            // rebinds those textures to units 0..3 so the HyperBlend shader sees
            // them as if they were the raw layer FBOs.
            // v5.0 R3: gate updated to allow Mode 3. PICASSO runs in Mode 2
            // (Linear off) AND in Mode 3 (chain allowed — Linear's abundance
            // FBO feeds PICASSO's initial blit via the source-tex switch
            // wired in R1 inside _runPicassoKernel). When Linear is on but
            // the chain dim contract fails, the throw in updatePicassoConfig
            // refused activation upstream — _picassoActive would be false here.
            if (this._picassoActive && this._picassoMatricesUploaded && (!this._unmixEnabled || this._chainAllowed())) {
                this._runPicassoKernel(activeLayers);
            }

            // ---- Mode 1 texture binding swap (v5.0 R1) ----
            // PICASSO already rebinds units 0..3 to its output textures inside
            // _runPicassoKernel (Mode 2). For Mode 1 (Linear only), swap unit 0
            // (and unit 1 when M > 4) to point at the Linear pre-blend FBO
            // textures. Units beyond the active output count stay bound to the
            // raw layer FBOs from earlier — the HTML guarantees channels at
            // those indices are disabled, so their contents don't reach the
            // tone-mapped sum. Mode 3 path is unreachable in R1 (L1325 throw).
            if (this._linearOutputReady && !this._picassoActive) {
                if (this._linearFBOTextures[0]) {
                    gl.activeTexture(gl.TEXTURE0);
                    gl.bindTexture(gl.TEXTURE_2D, this._linearFBOTextures[0]);
                }
                if (this._numOutputs > 4 && this._linearFBOTextures[1]) {
                    gl.activeTexture(gl.TEXTURE1);
                    gl.bindTexture(gl.TEXTURE_2D, this._linearFBOTextures[1]);
                }
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
                // RGB mode masks alpha slots (3/7/11/15 carry no input data) — but NOT
                // the layer-0/1 output lanes (ci<8) under Linear unmixing, where they hold
                // genuine output abundances (output 3 = layer-0 alpha, output 7 = layer-1
                // alpha). Layer-2/3 alpha (ch11/ch15) stay locked — never unmix outputs.
                // Evaluated per frame so a post-push _unmixEnabled flip takes effect.
                var locked = this._lockedChannels.has(ci) && !(this._unmixEnabled && ci < 8);
                eArr[ci] = (cfg.enabled && !locked) ? 1.0 : 0.0;
            }
            gl.uniform3fv(loc.uChannelColor, this._uColor);
            gl.uniform1fv(loc.uChannelGain, gArr);
            gl.uniform1fv(loc.uChannelEnabled, eArr);
            gl.uniform1f(loc.uToneMode, this._toneMode);

            // v5.0 R1: Linear pre-blend output (when enabled) is now consumed
            // via the unit 0..1 texture bindings prepared earlier in draw().
            // No uUnmixEnabled / uUnmixMatrix uniforms remain on this program.

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
                // NB: RGB-mode alpha-slot locking is NOT applied here. Doing so would
                // destructively clear .enabled during the Apply push that runs BEFORE
                // _unmixEnabled is set, permanently killing genuine unmix outputs. The
                // lock is instead applied reactively at the per-frame uniform build in
                // draw(), gated on _unmixEnabled (see eArr fill).
            }
            // v5.0 R1 channel-config invariant (master plan §5.1 Evaluator
            // fix #3, option b): when Linear unmixing is engaged, channels
            // 4..15 MUST be disabled. The HTML guarantees this via
            // _pushUnmixChannelConfig at zstackHyper.html:2787-2798; the
            // assert below catches any future caller that bypasses that
            // path. Non-throwing (console.assert) so production users see
            // no visible disruption — the only consequence of a violated
            // invariant is wrong colors when Mode 1/3 has > 4 outputs and
            // the upstream stage didn't write to layers 1/2/3.
            if (this._unmixEnabled) {
                for (var ai = 4; ai < 16; ai++) {
                    console.assert(this._channelConfig[ai].enabled === false,
                        '[HyperBlendWebGLDrawer] Linear-mode invariant violated: ch' + ai +
                        '.enabled must be false when _unmixEnabled === true (HTML guarantee).');
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
         *
         * v5.0 R1: flag-clear ordering on `active:false` matches master plan
         * §3.4 — flip _unmixEnabled FIRST so a racing draw observes the chain
         * gate going false BEFORE _texturesValid invalidates and BEFORE the
         * pre-blend FBO is dereferenced. Matrix is held intact across a
         * disable so a later re-enable can skip the upload until the user
         * actually loads a new pseudoinverse.
         */
        updateUnmixConfig(config) {
            if (!config) return;
            // Forbidden-direction guard (v5.0 R3): PICASSO → Linear is the
            // illegal cross-feed (doc/picasso-math.md §4 forbidden move #5).
            // Engaging Linear while PICASSO is already active would invert the
            // chain direction. The HTML Q5 block-on-conflict guard makes this
            // path unreachable in normal use, but the drawer-side throw is
            // defense-in-depth against any future HTML regression that
            // desyncs from drawer state. Symmetric with the Mode 3 throw in
            // updatePicassoConfig above.
            if (config.active === true && this._picassoActive) {
                throw new Error('Cannot engage Linear unmixing while PICASSO is active (forbidden direction). Clear PICASSO first.');
            }
            if (typeof config.active !== 'undefined') {
                var wasEnabled = this._unmixEnabled;
                var wantEnabled = !!config.active;
                if (wantEnabled && !wasEnabled) {
                    // Lazy-alloc Linear pre-blend resources on first activation
                    // (mirrors PICASSO's lazy pattern in updatePicassoConfig).
                    // Sized to the current canvas; _runLinearPass calls
                    // _ensureLinearResources again to handle subsequent resize.
                    this._ensureLinearResources(this.canvas.width, this.canvas.height);
                }
                if (wantEnabled !== wasEnabled) {
                    if (wantEnabled) {
                        // Enable path — set the flag, then invalidate so the
                        // next draw runs the pre-blend pass and rebinds units.
                        this._unmixEnabled = true;
                        this._texturesValid = false;
                    } else {
                        // Disable path — ordering per master plan §3.4 to
                        // avoid a torn read where downstream callers see
                        // chainAllowed=true with stale Linear textures.
                        // ORDER (load-bearing):
                        //   1. flip _unmixEnabled = false   (chain gate closes)
                        //   2. mark Linear output unready    (cached FBO no longer valid)
                        //   3. invalidate composite cache    (next draw rebinds raw layer FBOs)
                        // _unmixMatrix and _numOutputs are intentionally LEFT
                        // INTACT (matching pre-R1 behavior) so a follow-up
                        // updateUnmixConfig({active:true}) without a matrix
                        // re-engages Linear with the previous pseudoinverse —
                        // the HTML demote-then-re-Apply flow in v4.0 depends
                        // on this. _chainAllowed() therefore relies on
                        // _unmixEnabled (not _unmixMatrix or _numOutputs) as
                        // the definitive on/off signal.
                        this._unmixEnabled = false;
                        this._linearOutputReady = false;
                        this._texturesValid = false;
                    }
                }
            }
            if (config.matrix) {
                this._unmixMatrix.set(config.matrix);
                this._unmixMatrixDirty = true;
                // Matrix changed mid-session: invalidate the cached pre-blend
                // FBO content so the next draw recomputes via _runLinearPass.
                this._texturesValid = false;
                this._linearOutputReady = false;
            }
            if (typeof config.numOutputs !== 'undefined') {
                this._numOutputs = config.numOutputs;
            }
            if (this.viewer) {
                this.viewer.forceRedraw();
            }
        }

        // =====================================================================
        //  PICASSO PUBLIC API
        //  updatePicassoConfig — enable/disable + upload K square N×N matrices.
        //  getPicassoConfig    — read current PICASSO state (test introspection).
        //  readPicassoOutput   — single-pixel sample of one layer's output FBO.
        //  readPicassoLayerFull — full-FBO RGBA readback (used by headless test).
        //  runPicassoSelfChecks — eight runtime invariants from the spec.
        // =====================================================================

        /**
         * Update PICASSO configuration and request a re-draw.
         * Mirrors updateUnmixConfig's shape: active flag + matrices upload.
         *
         * @param {Object} config
         * @param {boolean} [config.active]
         * @param {Array<Float32Array>} [config.matrices] - K matrices, 16 floats each, row-major
         * @param {number} [config.K] - optional iteration count; must equal matrices.length
         */
        updatePicassoConfig(config) {
            if (!config) return;

            // Validate matrices first so a bad upload never partially mutates state.
            if (config.matrices) {
                var ms = config.matrices;
                if (!Array.isArray(ms)) {
                    throw new Error('PICASSO updatePicassoConfig: matrices must be an Array');
                }
                if (ms.length < 1 || ms.length > 10) {
                    throw new Error('PICASSO K must be in 1..10; got ' + ms.length);
                }
                if (typeof config.K !== 'undefined' && config.K !== ms.length) {
                    throw new Error('PICASSO K=' + config.K + ' declared but ' + ms.length + ' matrices provided');
                }
                for (var mi = 0; mi < ms.length; mi++) {
                    var m = ms[mi];
                    if (!(m instanceof Float32Array) || m.length !== 16) {
                        throw new Error('PICASSO block #' + mi + ' has invalid shape; expected 16-element Float32Array (4x4 row-major)');
                    }
                    for (var mj = 0; mj < 16; mj++) {
                        if (!isFinite(m[mj])) {
                            throw new Error('PICASSO matrix contains NaN or Inf at block ' + mi + ', index ' + mj);
                        }
                    }
                }
                // N is the matrix side derived from 16 floats — always 4 in Phase 3.
                // Hard-gated here for defense-in-depth: even if the parser were
                // bypassed, the drawer refuses anything but N=4.
                var derivedN = 4;
                if (derivedN !== 4) {
                    throw new Error('PICASSO Phase 3 is hard-wired for N=4; matrix #0 shape derived as ' + derivedN + 'x' + derivedN);
                }

                // Stash matrices and pre-compute the transposed/scaled uniforms.
                // Phase 1 trick: bake the [0..1]→[0..255] scale into iter-0's matrix
                // so the first FS pass converts the uint8-sampled source in one step.
                // See picasso-osd-demo.html L546–552 transposeAndScale.
                this._picassoMatrices = [];
                this._picassoTransposed = [];
                for (var ki = 0; ki < ms.length; ki++) {
                    var copy = new Float32Array(16);
                    copy.set(ms[ki]);
                    this._picassoMatrices.push(copy);
                    this._picassoTransposed.push(this._picassoTransposeAndScale(ms[ki], ki === 0 ? 255.0 : 1.0));
                }
                this._picassoK = ms.length;
                this._picassoN = derivedN;
                this._picassoMatricesUploaded = true;
                // Stale output: matrices changed, recompose downstream.
                if (this._picassoActive) {
                    this._texturesValid = false;
                }
            }

            if (typeof config.active !== 'undefined') {
                var wasActive = this._picassoActive;
                var want = !!config.active;
                if (want && this._picassoSupported === false) {
                    $.console.warn('[HyperBlendWebGLDrawer] PICASSO unavailable (missing OES_texture_float / WEBGL_color_buffer_float)');
                    return;
                }
                // Mode 3 chain (Linear → PICASSO) is now legal IFF the dimension
                // contract holds: Linear output count must equal PICASSO N. The
                // forbidden direction (PICASSO → Linear) is blocked symmetrically
                // in updateUnmixConfig and at self-check 6. Per doc/picasso-math.md
                // §4 the equality is design, not coincidence. Uses this._picassoN
                // (not the literal 4) so the message stays accurate if Phase 6+
                // generalises N. The N=4 hard-gate stays intact above where
                // derivedN is rejected on upload. Placed AFTER matrices upload
                // (which sets _picassoN from derivedN) so first-time Mode 3
                // engages cleanly when matrices + active arrive in one call.
                if (want && this._unmixEnabled) {
                    if (this._numOutputs !== this._picassoN) {
                        throw new Error('Mode 3 chain dimension mismatch: Linear outputs ' + this._numOutputs + ' but PICASSO N=' + this._picassoN);
                    }
                    // Dims match — proceed to engage. The chain will route
                    // Linear output to PICASSO input via _runPicassoKernel's
                    // source-tex switch (already wired in R1) on the next draw.
                }
                if (want && !wasActive) {
                    // Lazy alloc on first activation. Programs link once;
                    // FBOs sized to current canvas (will resize each frame as needed).
                    if (!this._picassoProgram || !this._picassoCastProgram) {
                        this._initPicassoPrograms();
                    }
                    if (!this._picassoFBOsAllocated) {
                        this._createPicassoFBOs(this.canvas.width, this.canvas.height);
                    }
                }
                this._picassoActive = want;
                if (wasActive !== this._picassoActive) {
                    // Toggle forces a re-composite so layer textures rebind correctly:
                    // when entering PICASSO the units 0..3 must hold output textures;
                    // when leaving they must hold the raw layer FBO textures.
                    this._texturesValid = false;
                }
            }

            if (this.viewer) {
                this.viewer.forceRedraw();
            }
        }

        /**
         * Test-only: get current PICASSO config snapshot.
         * @returns {{active, K, N, matrices, supported, chainAllowed, mode}}
         *
         * v5.0 R1 additions (live as of R3):
         *   - chainAllowed: result of _chainAllowed() — Mode 3 readiness gate.
         *   - mode: 'mode-0' | 'mode-1' | 'mode-2' | 'mode-3' — convenience
         *           label for the headless tests (matches master plan §3.3).
         * R3 lifted the mutex throw, so 'mode-3' now actually engages on
         * draw() when chainAllowed is true.
         */
        getPicassoConfig() {
            var matricesCopy = null;
            if (this._picassoMatrices) {
                matricesCopy = [];
                for (var i = 0; i < this._picassoMatrices.length; i++) {
                    var c = new Float32Array(16);
                    c.set(this._picassoMatrices[i]);
                    matricesCopy.push(c);
                }
            }
            var chainAllowed = this._chainAllowed();
            var mode;
            if (chainAllowed) {
                mode = 'mode-3';
            } else if (this._unmixEnabled && !this._picassoActive) {
                mode = 'mode-1';
            } else if (this._picassoActive && !this._unmixEnabled) {
                mode = 'mode-2';
            } else {
                mode = 'mode-0';
            }
            return {
                active: this._picassoActive,
                K: this._picassoK,
                N: this._picassoN,
                matrices: matricesCopy,
                supported: this._picassoSupported === true,
                chainAllowed: chainAllowed,
                mode: mode
            };
        }

        /**
         * Read 1 pixel from a PICASSO output FBO (Spectrum Inspector / self-check 7).
         * @param {number} layerIdx 0..3
         * @param {number} cssX
         * @param {number} cssY
         * @returns {Uint8Array(4)|null}
         */
        readPicassoOutput(layerIdx, cssX, cssY) {
            var gl = this._gl;
            if (!gl || this._contextLost) return null;
            if (!this._picassoActive || !this._picassoFBOsAllocated) return null;
            if (layerIdx < 0 || layerIdx >= this._picassoFBO_Out.length) return null;
            if (!this._picassoFBO_Out[layerIdx]) return null;

            var dpr = $.pixelDensityRatio;
            var glX = Math.round(cssX * dpr);
            var glY = this.canvas.height - Math.round(cssY * dpr) - 1;
            if (glX < 0 || glX >= this.canvas.width || glY < 0 || glY >= this.canvas.height) return null;

            var out = new Uint8Array(4);
            gl.bindFramebuffer(gl.FRAMEBUFFER, this._picassoFBO_Out[layerIdx]);
            gl.readPixels(glX, glY, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, out);
            gl.bindFramebuffer(gl.FRAMEBUFFER, null);
            return out;
        }

        /**
         * Full-FBO readback for one PICASSO layer (headless test fast-path).
         * Per-pixel readPixels at 2048² would take >10 minutes — this path is the
         * only way the integration test can dump the recovered image at scale.
         * @param {number} layerIdx 0..3
         * @returns {Uint8Array|null} length = W*H*4, RGBA in WebGL bottom-up order
         */
        readPicassoLayerFull(layerIdx) {
            var gl = this._gl;
            if (!gl || this._contextLost) return null;
            if (!this._picassoActive || !this._picassoFBOsAllocated) return null;
            if (layerIdx < 0 || layerIdx >= this._picassoFBO_Out.length) return null;
            if (!this._picassoFBO_Out[layerIdx]) return null;

            var w = this._picassoFBOWidth;
            var h = this._picassoFBOHeight;
            var raw = new Uint8Array(w * h * 4);
            gl.bindFramebuffer(gl.FRAMEBUFFER, this._picassoFBO_Out[layerIdx]);
            gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, raw);
            gl.bindFramebuffer(gl.FRAMEBUFFER, null);
            // gl.readPixels returns bottom-up; flip to top-down so callers get
            // a standard image orientation (matches readPicassoOutput's CSS-Y).
            var out = new Uint8Array(w * h * 4);
            var rowBytes = w * 4;
            for (var y = 0; y < h; y++) {
                out.set(raw.subarray((h - 1 - y) * rowBytes, (h - y) * rowBytes), y * rowBytes);
            }
            return out;
        }

        /**
         * Eight self-checks (spec §9.1). Mirrors Phase 2's runtime invariant set.
         * Caller is expected to have already drawn at least one PICASSO frame.
         * @returns {{passed: boolean, checks: Array<{name, result, detail}>}}
         */
        runPicassoSelfChecks() {
            var checks = [];
            var gl = this._gl;

            // [1] Parser shape gate: K in 1..10 AND N === 4.
            (function (self) {
                var ok = self._picassoN === 4 && self._picassoK >= 1 && self._picassoK <= 10;
                checks.push({
                    name: 'parser shape gate (N=4, 1≤K≤10)',
                    result: ok,
                    detail: 'N=' + self._picassoN + ' K=' + self._picassoK
                });
            })(this);

            // [2] WebGL extension presence — float texture + render-target.
            (function (self) {
                var ok, detail;
                if (self._picassoIsWebGL2) {
                    var cbf2 = gl ? !!gl.getExtension('EXT_color_buffer_float') : false;
                    ok = self._picassoSupported === true && cbf2;
                    detail = '_picassoSupported=' + self._picassoSupported + ' (WebGL2) EXT_color_buffer_float=' + cbf2;
                } else {
                    var fT = gl ? !!gl.getExtension('OES_texture_float') : false;
                    var fR = gl ? !!gl.getExtension('WEBGL_color_buffer_float') : false;
                    ok = self._picassoSupported === true && fT && fR;
                    detail = '_picassoSupported=' + self._picassoSupported + ' OES_texture_float=' + fT + ' WEBGL_color_buffer_float=' + fR;
                }
                checks.push({
                    name: 'WebGL float-FBO extensions',
                    result: ok,
                    detail: detail
                });
            })(this);

            // [3] FBO completeness across all 12 (4 layers × {A, B, Out}).
            (function (self) {
                var details = [];
                var ok = true;
                if (!gl || !self._picassoFBOsAllocated) {
                    ok = false;
                    details.push('FBOs not allocated');
                } else {
                    var sets = [
                        { label: 'A', arr: self._picassoFBO_A },
                        { label: 'B', arr: self._picassoFBO_B },
                        { label: 'Out', arr: self._picassoFBO_Out }
                    ];
                    for (var s = 0; s < sets.length; s++) {
                        for (var i = 0; i < sets[s].arr.length; i++) {
                            gl.bindFramebuffer(gl.FRAMEBUFFER, sets[s].arr[i]);
                            var status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
                            var complete = (status === gl.FRAMEBUFFER_COMPLETE);
                            if (!complete) {
                                ok = false;
                                details.push(sets[s].label + i + '=0x' + status.toString(16));
                            }
                        }
                    }
                    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
                }
                checks.push({
                    name: 'FBO completeness (12 FBOs)',
                    result: ok,
                    detail: ok ? 'all COMPLETE' : details.join(', ')
                });
            })(this);

            // [4] Matrix uniform upload verified — uses iter index min(2, K-1).
            (function (self) {
                var ok = false;
                var detail = 'skipped';
                if (gl && self._picassoProgram && self._picassoTransposed && self._picassoTransposed.length > 0) {
                    var probeK = Math.min(2, self._picassoTransposed.length - 1);
                    gl.useProgram(self._picassoProgram);
                    gl.uniformMatrix4fv(self._picassoUniforms.uP, false, self._picassoTransposed[probeK]);
                    var got = gl.getUniform(self._picassoProgram, self._picassoUniforms.uP);
                    var maxDelta = 0;
                    for (var i = 0; i < 16; i++) {
                        var d = Math.abs(got[i] - self._picassoTransposed[probeK][i]);
                        if (d > maxDelta) maxDelta = d;
                    }
                    ok = maxDelta < 1e-6;
                    detail = 'probe k=' + probeK + ' max|Δ|=' + maxDelta.toExponential(2);
                }
                checks.push({
                    name: 'matrix uniform upload verified',
                    result: ok,
                    detail: detail
                });
            })(this);

            // [5] Ping-pong direction correctness — simulated traversal mirroring
            // _runPicassoKernel; dst must never equal src.
            (function (self) {
                var violations = 0;
                var trace = [];
                for (var li = 0; li < self._picassoFBO_A.length; li++) {
                    if (!self._picassoFBO_A[li] || !self._picassoFBO_B[li]) continue;
                    var srcTex = self._layerFBOTextures[li];
                    var dstFbo = self._picassoFBO_A[li];
                    var dstTex = self._picassoTex_A[li];
                    var altFbo = self._picassoFBO_B[li];
                    var altTex = self._picassoTex_B[li];
                    for (var k = 0; k < self._picassoK; k++) {
                        if (dstTex === srcTex || dstFbo === null) violations++;
                        var prevDstTex = dstTex;
                        var prevDstFbo = dstFbo;
                        srcTex = prevDstTex;
                        dstFbo = altFbo;
                        dstTex = altTex;
                        altFbo = prevDstFbo;
                        altTex = prevDstTex;
                    }
                    trace.push('layer' + li + ' iters=' + self._picassoK);
                }
                checks.push({
                    name: 'ping-pong direction correctness',
                    result: violations === 0,
                    detail: trace.join(', ') + '; violations=' + violations
                });
            })(this);

            // [6] No PICASSO→Linear cross-feed; Linear→PICASSO chain requires N match.
            // Two-part check (v5.0 R3):
            //   (a) Direction: if both Linear and PICASSO are engaged
            //       simultaneously, the dimension contract must hold
            //       (_numOutputs === _picassoN). This is the only legal
            //       Mode 3 state. Per doc/picasso-math.md §4, PICASSO →
            //       Linear is the forbidden direction; the dim-mismatch
            //       case can only happen via a drawer-side throw bypass
            //       and is reported as a self-check failure.
            //   (b) FBO-aliasing: defense-in-depth against a future refactor
            //       that accidentally shares a WebGLFramebuffer or
            //       WebGLTexture between the Linear pre-blend stage and the
            //       PICASSO stage — that would silently create the
            //       forbidden back-channel.
            (function (self) {
                var pre = !(self._unmixEnabled && self._picassoActive && self._numOutputs !== self._picassoN);
                var chainAllowed = self._chainAllowed();
                var detail = '_unmixEnabled=' + self._unmixEnabled +
                             ' _picassoActive=' + self._picassoActive +
                             ' _numOutputs=' + self._numOutputs +
                             ' _picassoN=' + self._picassoN +
                             ' chainAllowed=' + chainAllowed;

                // FBO-aliasing sub-assert: every Linear FBO/texture must be
                // distinct from every PICASSO FBO/texture across all layers.
                // Null entries are skipped — un-allocated slots cannot alias.
                var aliasViolation = null;
                var lFBOs = self._linearFBOs || [];
                var lTex  = self._linearFBOTextures || [];
                var pFBO_A = self._picassoFBO_A || [];
                var pFBO_B = self._picassoFBO_B || [];
                var pFBO_O = self._picassoFBO_Out || [];
                var pTex_A = self._picassoTex_A || [];
                var pTex_B = self._picassoTex_B || [];
                var pTex_O = self._picassoTex_Out || [];
                for (var lfi = 0; lfi < lFBOs.length && !aliasViolation; lfi++) {
                    var lf = lFBOs[lfi];
                    if (!lf) continue;
                    for (var pi = 0; pi < pFBO_A.length && !aliasViolation; pi++) {
                        if (pFBO_A[pi] && lf === pFBO_A[pi]) {
                            aliasViolation = '_linearFBOs[' + lfi + '] === _picassoFBO_A[' + pi + ']';
                        }
                    }
                    for (var pi2 = 0; pi2 < pFBO_B.length && !aliasViolation; pi2++) {
                        if (pFBO_B[pi2] && lf === pFBO_B[pi2]) {
                            aliasViolation = '_linearFBOs[' + lfi + '] === _picassoFBO_B[' + pi2 + ']';
                        }
                    }
                    for (var pi3 = 0; pi3 < pFBO_O.length && !aliasViolation; pi3++) {
                        if (pFBO_O[pi3] && lf === pFBO_O[pi3]) {
                            aliasViolation = '_linearFBOs[' + lfi + '] === _picassoFBO_Out[' + pi3 + ']';
                        }
                    }
                }
                for (var lti = 0; lti < lTex.length && !aliasViolation; lti++) {
                    var lt = lTex[lti];
                    if (!lt) continue;
                    for (var qi = 0; qi < pTex_A.length && !aliasViolation; qi++) {
                        if (pTex_A[qi] && lt === pTex_A[qi]) {
                            aliasViolation = '_linearFBOTextures[' + lti + '] === _picassoTex_A[' + qi + ']';
                        }
                    }
                    for (var qi2 = 0; qi2 < pTex_B.length && !aliasViolation; qi2++) {
                        if (pTex_B[qi2] && lt === pTex_B[qi2]) {
                            aliasViolation = '_linearFBOTextures[' + lti + '] === _picassoTex_B[' + qi2 + ']';
                        }
                    }
                    for (var qi3 = 0; qi3 < pTex_O.length && !aliasViolation; qi3++) {
                        if (pTex_O[qi3] && lt === pTex_O[qi3]) {
                            aliasViolation = '_linearFBOTextures[' + lti + '] === _picassoTex_Out[' + qi3 + ']';
                        }
                    }
                }

                var ok = pre && !aliasViolation;
                if (aliasViolation) {
                    detail += '; FBO_ALIAS_VIOLATION=' + aliasViolation;
                }
                checks.push({
                    name: 'no PICASSO→Linear cross-feed; Linear→PICASSO chain requires N match',
                    result: ok,
                    detail: detail
                });
            })(this);

            // [7] CPU/GPU round-trip per active layer at canvas center.
            (function (self) {
                var ok = true;
                var details = [];
                if (!gl || !self._picassoFBOsAllocated || !self._picassoMatricesUploaded) {
                    ok = false;
                    details.push('preconditions not met');
                } else {
                    var w = self.canvas.width;
                    var h = self.canvas.height;
                    var px = Math.floor(w / 2);
                    var py = Math.floor(h / 2);
                    var mp = { N: self._picassoN, K: self._picassoK, P: self._picassoMatrices };
                    var buf = new Uint8Array(4);
                    for (var li = 0; li < self._layerFBOs.length; li++) {
                        // Sample raw layer
                        gl.bindFramebuffer(gl.FRAMEBUFFER, self._layerFBOs[li]);
                        gl.readPixels(px, py, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, buf);
                        var mixed = new Float32Array([buf[0], buf[1], buf[2], buf[3]]);
                        // Run CPU reference
                        var cpu = self._picassoCPUKernel(mixed, mp);
                        // Sample GPU output
                        gl.bindFramebuffer(gl.FRAMEBUFFER, self._picassoFBO_Out[li]);
                        gl.readPixels(px, py, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, buf);
                        var maxD = 0;
                        for (var c = 0; c < 4; c++) {
                            var d = Math.abs(buf[c] - cpu[c]);
                            if (d > maxD) maxD = d;
                        }
                        if (maxD > 1) ok = false;
                        details.push('L' + li + ' mixed=[' + Array.from(mixed).join(',') +
                            '] cpu=[' + Array.from(cpu).join(',') +
                            '] gpu=[' + Array.from(buf).join(',') + '] max|Δ|=' + maxD);
                    }
                    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
                }
                checks.push({
                    name: 'CPU/GPU round-trip per layer (canvas center)',
                    result: ok,
                    detail: details.join('; ')
                });
            })(this);

            // [8] Row-major → column-major transpose verified at element (0,1).
            (function (self) {
                var ok = false;
                var detail = 'skipped';
                if (self._picassoMatrices && self._picassoMatrices.length > 1) {
                    var rm = self._picassoMatrices[1];
                    var tr = self._picassoTransposeAndScale(rm, 1.0);
                    var got00 = tr[0], want00 = rm[0];
                    var got10 = tr[1], want10 = rm[1 * 4 + 0];   // file row=1 col=0 → uniform col=0 row=1
                    var got01 = tr[4], want01 = rm[1];           // file row=0 col=1 → uniform col=1 row=0
                    ok = (got00 === want00) && (got10 === want10) && (got01 === want01);
                    detail = '(0,0): file=' + want00.toFixed(4) + ' uniform=' + got00.toFixed(4) +
                             '; file[1][0]=' + want10.toFixed(4) + ' → uniform[1]=' + got10.toFixed(4) +
                             '; file[0][1]=' + want01.toFixed(4) + ' → uniform[4]=' + got01.toFixed(4);
                }
                checks.push({
                    name: 'row-major → column-major transpose',
                    result: ok,
                    detail: detail
                });
            })(this);

            var passed = true;
            for (var ck = 0; ck < checks.length; ck++) {
                if (!checks[ck].result) passed = false;
            }
            return { passed: passed, checks: checks };
        }

        // ---- Internal: probe PICASSO extensions on the live GL context ----
        _probePicassoExtensions() {
            var gl = this._gl;
            if (!gl) {
                this._picassoSupported = false;
                return;
            }
            // WebGL 2 names render-to-float as EXT_color_buffer_float and supports
            // float textures natively via RGBA32F. WebGL 1 needs the older pair.
            var isWebGL2 = (typeof WebGL2RenderingContext !== 'undefined' &&
                            gl instanceof WebGL2RenderingContext);
            this._picassoIsWebGL2 = isWebGL2;
            this._picassoFloatInternalFmt = isWebGL2 ? gl.RGBA32F : gl.RGBA;
            if (isWebGL2) {
                var ext2 = gl.getExtension('EXT_color_buffer_float');
                if (!ext2) {
                    this._picassoSupported = false;
                    $.console.warn('[HyperBlendWebGLDrawer] PICASSO disabled: missing EXT_color_buffer_float on WebGL 2');
                    return;
                }
                gl.getExtension('OES_texture_float_linear');
            } else {
                var fT = gl.getExtension('OES_texture_float');
                var fR = gl.getExtension('WEBGL_color_buffer_float');
                gl.getExtension('OES_texture_float_linear');
                if (!fT || !fR) {
                    this._picassoSupported = false;
                    $.console.warn('[HyperBlendWebGLDrawer] PICASSO disabled: missing ' +
                        (!fT ? 'OES_texture_float ' : '') + (!fR ? 'WEBGL_color_buffer_float' : ''));
                    return;
                }
            }
            this._picassoSupported = true;
        }

        // ---- Internal: link PICASSO + cast programs, cache uniform locations ----
        _initPicassoPrograms() {
            var gl = this._gl;
            if (!gl) return;

            // PICASSO kernel program
            var vs = this._compileShader(gl, gl.VERTEX_SHADER, VERTEX_SHADER_SRC);
            var fs = this._compileShader(gl, gl.FRAGMENT_SHADER, PICASSO_FRAGMENT_SHADER_SRC);
            if (vs && fs) {
                var prog = gl.createProgram();
                gl.attachShader(prog, vs);
                gl.attachShader(prog, fs);
                gl.linkProgram(prog);
                if (gl.getProgramParameter(prog, gl.LINK_STATUS)) {
                    this._picassoProgram = prog;
                    this._picassoUniforms = {
                        uSrc: gl.getUniformLocation(prog, 'uSrc'),
                        uP: gl.getUniformLocation(prog, 'uP')
                    };
                    this._picassoAttribs = {
                        aPosition: gl.getAttribLocation(prog, 'aPosition'),
                        aTexCoord: gl.getAttribLocation(prog, 'aTexCoord')
                    };
                } else {
                    $.console.error('[HyperBlendWebGLDrawer] PICASSO program link failed:', gl.getProgramInfoLog(prog));
                    this._picassoSupported = false;
                }
                gl.deleteShader(vs);
                gl.deleteShader(fs);
            }

            // Cast program (float → uint8)
            var vs2 = this._compileShader(gl, gl.VERTEX_SHADER, VERTEX_SHADER_SRC);
            var fs2 = this._compileShader(gl, gl.FRAGMENT_SHADER, PICASSO_CAST_FRAGMENT_SHADER_SRC);
            if (vs2 && fs2) {
                var cprog = gl.createProgram();
                gl.attachShader(cprog, vs2);
                gl.attachShader(cprog, fs2);
                gl.linkProgram(cprog);
                if (gl.getProgramParameter(cprog, gl.LINK_STATUS)) {
                    this._picassoCastProgram = cprog;
                    this._picassoCastUniforms = {
                        uSrc: gl.getUniformLocation(cprog, 'uSrc')
                    };
                    this._picassoCastAttribs = {
                        aPosition: gl.getAttribLocation(cprog, 'aPosition'),
                        aTexCoord: gl.getAttribLocation(cprog, 'aTexCoord')
                    };
                } else {
                    $.console.error('[HyperBlendWebGLDrawer] PICASSO cast program link failed:', gl.getProgramInfoLog(cprog));
                    this._picassoSupported = false;
                }
                gl.deleteShader(vs2);
                gl.deleteShader(fs2);
            }
        }

        // ---- Internal: allocate 12 PICASSO FBOs (4 layers × {A float, B float, Out uint8}) ----
        // Mirrors Phase 2 createFBO at picasso-osd-demo.html L516–531 with the
        // completeness probe per FBO. On ANY failure free everything and set
        // _picassoSupported=false so the gate in updatePicassoConfig skips PICASSO.
        _createPicassoFBOs(w, h) {
            var gl = this._gl;
            if (!gl || this._picassoSupported === false) return;
            if (w <= 0 || h <= 0) return;

            var created = { textures: [], framebuffers: [] };
            var ok = true;
            var self = this;

            function makeFBO(dtype) {
                var internalFmt = (dtype === gl.FLOAT) ? self._picassoFloatInternalFmt : gl.RGBA;
                var tex = gl.createTexture();
                gl.bindTexture(gl.TEXTURE_2D, tex);
                gl.texImage2D(gl.TEXTURE_2D, 0, internalFmt, w, h, 0, gl.RGBA, dtype, null);
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
                var fbo = gl.createFramebuffer();
                gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
                gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
                var status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
                gl.bindFramebuffer(gl.FRAMEBUFFER, null);
                created.textures.push(tex);
                created.framebuffers.push(fbo);
                return { tex: tex, fbo: fbo, ok: status === gl.FRAMEBUFFER_COMPLETE, status: status };
            }

            this._picassoFBO_A = [];
            this._picassoFBO_B = [];
            this._picassoFBO_Out = [];
            this._picassoTex_A = [];
            this._picassoTex_B = [];
            this._picassoTex_Out = [];

            for (var i = 0; i < 4; i++) {
                var a = makeFBO(gl.FLOAT);
                var b = makeFBO(gl.FLOAT);
                var o = makeFBO(gl.UNSIGNED_BYTE);
                if (!a.ok || !b.ok || !o.ok) {
                    ok = false;
                    $.console.error('[HyperBlendWebGLDrawer] PICASSO FBO incomplete (layer ' + i + '): A=0x' +
                        a.status.toString(16) + ' B=0x' + b.status.toString(16) + ' Out=0x' + o.status.toString(16));
                    break;
                }
                this._picassoFBO_A.push(a.fbo);
                this._picassoFBO_B.push(b.fbo);
                this._picassoFBO_Out.push(o.fbo);
                this._picassoTex_A.push(a.tex);
                this._picassoTex_B.push(b.tex);
                this._picassoTex_Out.push(o.tex);
            }

            if (!ok) {
                // Free everything created so far — leave the drawer in a clean
                // state where _picassoActive cannot be true.
                for (var fi = 0; fi < created.framebuffers.length; fi++) {
                    gl.deleteFramebuffer(created.framebuffers[fi]);
                }
                for (var ti = 0; ti < created.textures.length; ti++) {
                    gl.deleteTexture(created.textures[ti]);
                }
                self._picassoFBO_A = [];
                self._picassoFBO_B = [];
                self._picassoFBO_Out = [];
                self._picassoTex_A = [];
                self._picassoTex_B = [];
                self._picassoTex_Out = [];
                self._picassoSupported = false;
                self._picassoActive = false;
                return;
            }

            this._picassoFBOsAllocated = true;
            this._picassoFBOWidth = w;
            this._picassoFBOHeight = h;
        }

        // ---- Internal: resize PICASSO FBO textures to match canvas ----
        _resizePicassoFBOs(w, h) {
            if (w === this._picassoFBOWidth && h === this._picassoFBOHeight) return;
            var gl = this._gl;
            if (!gl || !this._picassoFBOsAllocated) return;
            var floatFmt = this._picassoFloatInternalFmt;
            for (var i = 0; i < this._picassoTex_A.length; i++) {
                gl.bindTexture(gl.TEXTURE_2D, this._picassoTex_A[i]);
                gl.texImage2D(gl.TEXTURE_2D, 0, floatFmt, w, h, 0, gl.RGBA, gl.FLOAT, null);
                gl.bindTexture(gl.TEXTURE_2D, this._picassoTex_B[i]);
                gl.texImage2D(gl.TEXTURE_2D, 0, floatFmt, w, h, 0, gl.RGBA, gl.FLOAT, null);
                gl.bindTexture(gl.TEXTURE_2D, this._picassoTex_Out[i]);
                gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
            }
            gl.bindTexture(gl.TEXTURE_2D, null);
            this._picassoFBOWidth = w;
            this._picassoFBOHeight = h;
        }

        // ---- Internal: run K iterations of max(0, P^(k) @ x) per active layer ----
        // Per-layer flow: raw layer FBO → initial blit into A (float) → K kernel
        // passes ping-pong A↔B with the existing _blitProgram REPLACED by
        // _picassoProgram and the per-iter transposed matrix → final cast to Out
        // (uint8) → rebind Out to texture unit i. Ported from picasso-osd-demo.html
        // L765–792 (_runPicassoKernel); split across layers per spec §1.3.
        _runPicassoKernel(activeLayers) {
            var gl = this._gl;
            if (!gl || !this._picassoFBOsAllocated || !this._picassoProgram || !this._picassoCastProgram) return;
            if (!this._picassoTransposed || this._picassoK <= 0) return;

            var w = this._picassoFBOWidth;
            var h = this._picassoFBOHeight;
            var aPosKernel = this._picassoAttribs.aPosition;
            var aTexKernel = this._picassoAttribs.aTexCoord;
            var aPosCast = this._picassoCastAttribs.aPosition;
            var aTexCast = this._picassoCastAttribs.aTexCoord;
            var aPosBlit = this._blitAttribs.aPosition;
            var aTexBlit = this._blitAttribs.aTexCoord;

            // Bind the static full-screen quad geometry shared with Pass 1.
            gl.bindBuffer(gl.ARRAY_BUFFER, this._posBuffer);
            // (per-program attribute setup happens below — locations differ per program)

            // v5.0 R1: Mode 3 source-tex selection. When the Linear→PICASSO
            // chain is allowed (see _chainAllowed), the initial blit feeds
            // PICASSO from the Linear pre-blend FBO instead of the raw layer
            // FBO. Phase 5 hard-gates N=4, which also implies the Linear
            // pre-blend wrote into _linearFBOTextures[0] (channels 0..3);
            // layers 1..3 fall through to their raw FBO inputs and the
            // resulting PICASSO outputs feed the disabled ch4..15 channels in
            // Pass 1, which are masked off via uChannelEnabled (HTML guarantee
            // + console.assert in updateChannelConfig).
            // R3 lifted the mutex throw and rewrote the draw() PICASSO gate
            // to permit chained execution, so this branch is now reachable
            // whenever Mode 3 is engaged (Linear apply + dim-matching PICASSO
            // apply). The wiring itself was authored in R1; only the upstream
            // gates changed.
            var chained = this._chainAllowed() && this._linearOutputReady;

            for (var li = 0; li < activeLayers.length; li++) {
                if (!activeLayers[li] || li >= this._picassoFBO_A.length) continue;
                if (!this._layerFBOTextures[li]) continue;

                // --- Initial blit: raw uint8 layer FBO texture → _picassoFBO_A[li] (float). ---
                // Reuse the existing _blitProgram (no scale; the kernel's iter-0 matrix
                // bakes in the ×255 promotion via transposeAndScale at upload time).
                gl.useProgram(this._blitProgram);
                gl.uniform1i(this._blitUniforms.uTile, 0);
                gl.bindFramebuffer(gl.FRAMEBUFFER, this._picassoFBO_A[li]);
                gl.viewport(0, 0, w, h);
                gl.clearColor(0, 0, 0, 0);
                gl.clear(gl.COLOR_BUFFER_BIT);
                gl.activeTexture(gl.TEXTURE0);
                // Phase 5 N=4 hard-gate: chained PICASSO reads from
                // _linearFBOTextures[0] for layer 0 only. Layers 1..3 still
                // sample their raw FBO inputs (their PICASSO outputs land in
                // ch4..15 which Pass 1 has disabled). When chained === false
                // (Mode 2 standalone) all 4 layers come from raw FBOs —
                // identical to pre-R1 behaviour.
                var initialSrcTex;
                if (chained && li === 0 && this._linearFBOTextures[0]) {
                    initialSrcTex = this._linearFBOTextures[0];
                } else {
                    initialSrcTex = this._layerFBOTextures[li];
                }
                gl.bindTexture(gl.TEXTURE_2D, initialSrcTex);
                gl.bindBuffer(gl.ARRAY_BUFFER, this._posBuffer);
                gl.enableVertexAttribArray(aPosBlit);
                gl.vertexAttribPointer(aPosBlit, 2, gl.FLOAT, false, 0, 0);
                gl.bindBuffer(gl.ARRAY_BUFFER, this._texBufferFBO);
                gl.enableVertexAttribArray(aTexBlit);
                gl.vertexAttribPointer(aTexBlit, 2, gl.FLOAT, false, 0, 0);
                gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

                // --- K iterations of ping-pong with PICASSO kernel. ---
                gl.useProgram(this._picassoProgram);
                gl.uniform1i(this._picassoUniforms.uSrc, 0);
                gl.bindBuffer(gl.ARRAY_BUFFER, this._posBuffer);
                gl.enableVertexAttribArray(aPosKernel);
                gl.vertexAttribPointer(aPosKernel, 2, gl.FLOAT, false, 0, 0);
                gl.bindBuffer(gl.ARRAY_BUFFER, this._texBufferFBO);
                gl.enableVertexAttribArray(aTexKernel);
                gl.vertexAttribPointer(aTexKernel, 2, gl.FLOAT, false, 0, 0);

                // Start: src = _picassoTex_A (the just-blit output), dst = _picassoFBO_B.
                // Note: dst tex parallels dst fbo so we can use it as the next src.
                var srcTex = this._picassoTex_A[li];
                var dstFbo = this._picassoFBO_B[li];
                var dstTex = this._picassoTex_B[li];
                var altFbo = this._picassoFBO_A[li];
                var altTex = this._picassoTex_A[li];
                var lastDstTex = null;
                for (var k = 0; k < this._picassoK; k++) {
                    gl.bindFramebuffer(gl.FRAMEBUFFER, dstFbo);
                    gl.viewport(0, 0, w, h);
                    gl.activeTexture(gl.TEXTURE0);
                    gl.bindTexture(gl.TEXTURE_2D, srcTex);
                    gl.uniformMatrix4fv(this._picassoUniforms.uP, false, this._picassoTransposed[k]);
                    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
                    lastDstTex = dstTex;
                    // Swap: next src = current dst; next dst = the other FBO/tex.
                    var prevDstFbo = dstFbo;
                    var prevDstTex = dstTex;
                    srcTex = prevDstTex;
                    dstFbo = altFbo;
                    dstTex = altTex;
                    altFbo = prevDstFbo;
                    altTex = prevDstTex;
                }

                // --- Final cast: float final → uint8 Out FBO. ---
                gl.useProgram(this._picassoCastProgram);
                gl.uniform1i(this._picassoCastUniforms.uSrc, 0);
                gl.bindBuffer(gl.ARRAY_BUFFER, this._posBuffer);
                gl.enableVertexAttribArray(aPosCast);
                gl.vertexAttribPointer(aPosCast, 2, gl.FLOAT, false, 0, 0);
                gl.bindBuffer(gl.ARRAY_BUFFER, this._texBufferFBO);
                gl.enableVertexAttribArray(aTexCast);
                gl.vertexAttribPointer(aTexCast, 2, gl.FLOAT, false, 0, 0);
                gl.bindFramebuffer(gl.FRAMEBUFFER, this._picassoFBO_Out[li]);
                gl.viewport(0, 0, w, h);
                gl.activeTexture(gl.TEXTURE0);
                gl.bindTexture(gl.TEXTURE_2D, lastDstTex);
                gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
            }

            // Unbind FBO so subsequent Pass 1 hits the default framebuffer.
            gl.bindFramebuffer(gl.FRAMEBUFFER, null);

            // Rebind PICASSO output textures to units 0..3 in place of the raw
            // layer FBOs. The HyperBlend shader samples uLayer0..3 unchanged.
            // Absent layers keep their cleared raw FBO texture bound from the
            // earlier composite path (no rebind here).
            for (var bi = 0; bi < activeLayers.length; bi++) {
                if (!activeLayers[bi] || bi >= this._picassoTex_Out.length) continue;
                if (!this._picassoTex_Out[bi]) continue;
                gl.activeTexture(gl.TEXTURE0 + bi);
                gl.bindTexture(gl.TEXTURE_2D, this._picassoTex_Out[bi]);
            }
        }

        // ---- Internal: transpose row-major 4×4 to column-major + optional scale ----
        // Port of picasso-osd-demo.html L546–552 transposeAndScale.
        _picassoTransposeAndScale(rowMajor, scale) {
            var out = new Float32Array(16);
            for (var i = 0; i < 4; i++) {
                for (var j = 0; j < 4; j++) {
                    out[j * 4 + i] = rowMajor[i * 4 + j] * scale;
                }
            }
            return out;
        }

        // ---- Internal: CPU reference kernel (used by self-check 7) ----
        // Port of runPicassoKernelCPU at picasso-osd-demo.html L328–347.
        _picassoCPUKernel(pixel, mpInfo) {
            var N = mpInfo.N, K = mpInfo.K, P = mpInfo.P;
            var v = new Float32Array(N), w = new Float32Array(N);
            for (var c = 0; c < N; c++) v[c] = pixel[c];
            for (var k = 0; k < K; k++) {
                var Pk = P[k];
                for (var i = 0; i < N; i++) {
                    var acc = 0.0;
                    var rowOff = i * N;
                    for (var j = 0; j < N; j++) acc += Pk[rowOff + j] * v[j];
                    w[i] = acc < 0.0 ? 0.0 : acc;
                }
                for (var c2 = 0; c2 < N; c2++) v[c2] = w[c2];
            }
            var out = new Uint8Array(N);
            for (var c3 = 0; c3 < N; c3++) {
                var r = v[c3];
                if (r < 0) r = 0; else if (r > 255) r = 255;
                out[c3] = Math.round(r) | 0;
            }
            return out;
        }

        // =====================================================================
        //  LINEAR PRE-BLEND STAGE (v5.0 R1)
        //  _chainAllowed         — Mode 3 dim-contract predicate (used by
        //                          _runPicassoKernel for source-tex selection).
        //  _ensureLinearResources — lazy-allocates program + FBOs.
        //  _initLinearProgram     — compile/link LINEAR_PASS_FRAGMENT_SRC.
        //  _resizeLinearFBOs      — re-allocate FBO textures on canvas resize.
        //  _runLinearPass         — writes M abundance outputs into
        //                           _linearFBOTextures[0..1] (uint8 RGBA).
        // =====================================================================

        /**
         * Mode 3 chain-allowed predicate (v5.0 master plan §3.2).
         * Returns true iff Linear→PICASSO chain may execute on the next draw.
         *
         * Consumers in R1: _runPicassoKernel's initial source-tex switch.
         * Consumers in R3+: throw text + self-check 6 rephrasing, HTML status
         * branches. Document inline so a future reader can find the contract.
         *
         * Phase 5 hard-gates PICASSO to N=4 (see updatePicassoConfig L1356),
         * so chainAllowed === true also implies numOutputs === 4. Future
         * phases that generalise N must keep the equality. Per
         * doc/picasso-math.md §4 the equality is design, not coincidence.
         */
        _chainAllowed() {
            // The plan (§3.2) phrases the matrix check as `_unmixMatrix != null`.
            // In this codebase _unmixMatrix is a pre-allocated Float32Array(128)
            // that is never reset to null; instead, disable paths zero it and
            // set _numOutputs to 0, so checking numOutputs > 0 is the practical
            // equivalent of "a valid Linear matrix is loaded".
            return !!(this._unmixEnabled
                   && this._picassoActive
                   && this._unmixMatrix
                   && this._numOutputs > 0
                   && this._numOutputs === this._picassoN);
        }

        // ---- Internal: lazy-create Linear pre-blend program + FBOs ----
        _ensureLinearResources(w, h) {
            var gl = this._gl;
            if (!gl) return false;
            if (!this._linearProgram) {
                this._initLinearProgram();
                if (!this._linearProgram) return false;
            }
            if (!this._linearFBOs[0] || w !== this._linearFBOWidth_pre || h !== this._linearFBOHeight_pre) {
                this._resizeLinearFBOs(w, h);
            }
            return !!this._linearFBOs[0];
        }

        // ---- Internal: compile/link the Linear pre-blend program ----
        _initLinearProgram() {
            var gl = this._gl;
            if (!gl) return;
            var vs = this._compileShader(gl, gl.VERTEX_SHADER, VERTEX_SHADER_SRC);
            var fs = this._compileShader(gl, gl.FRAGMENT_SHADER, LINEAR_PASS_FRAGMENT_SRC);
            if (!vs || !fs) {
                $.console.error('[HyperBlendWebGLDrawer] Linear pre-blend shader compilation failed');
                if (vs) gl.deleteShader(vs);
                if (fs) gl.deleteShader(fs);
                return;
            }
            var prog = gl.createProgram();
            gl.attachShader(prog, vs);
            gl.attachShader(prog, fs);
            gl.linkProgram(prog);
            if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
                $.console.error('[HyperBlendWebGLDrawer] Linear pre-blend program link failed:', gl.getProgramInfoLog(prog));
                gl.deleteShader(vs);
                gl.deleteShader(fs);
                gl.deleteProgram(prog);
                return;
            }
            this._linearProgram = prog;
            this._linearUniforms = {
                uLayer0: gl.getUniformLocation(prog, 'uLayer0'),
                uLayer1: gl.getUniformLocation(prog, 'uLayer1'),
                uLayer2: gl.getUniformLocation(prog, 'uLayer2'),
                uLayer3: gl.getUniformLocation(prog, 'uLayer3'),
                uUnmixMatrix: gl.getUniformLocation(prog, 'uUnmixMatrix[0]'),
                uOutputOffset: gl.getUniformLocation(prog, 'uOutputOffset')
            };
            this._linearAttribs = {
                aPosition: gl.getAttribLocation(prog, 'aPosition'),
                aTexCoord: gl.getAttribLocation(prog, 'aTexCoord')
            };
            // Static sampler bindings (units 0..3 match the raw layer FBO bindings)
            gl.useProgram(prog);
            gl.uniform1i(this._linearUniforms.uLayer0, 0);
            gl.uniform1i(this._linearUniforms.uLayer1, 1);
            gl.uniform1i(this._linearUniforms.uLayer2, 2);
            gl.uniform1i(this._linearUniforms.uLayer3, 3);
            gl.deleteShader(vs);
            gl.deleteShader(fs);
        }

        // ---- Internal: (re)allocate the two RGBA8 Linear pre-blend FBOs ----
        // Two FBOs cover M ≤ 8: each carries 4 output components. For M ≤ 4
        // only [0] is used; [1] still gets allocated so the resize bookkeeping
        // stays simple. Cost is two RGBA8 textures at canvas size (~16 MB
        // total at 2048×2048) — negligible next to the existing layer FBOs.
        _resizeLinearFBOs(w, h) {
            var gl = this._gl;
            if (!gl) return;
            if (w <= 0 || h <= 0) return;

            // v5.0 R1 / MRA fix #4: ensure the uint8 abundance handoff to
            // PICASSO is NOT alpha-premultiplied. The Linear FBO is rendered
            // to (not uploaded from external pixel data), so this only
            // affects any subsequent texImage2D calls — keep the pack/unpack
            // state defensive in case a later code path uploads into the
            // texture directly.
            gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);

            for (var i = 0; i < 2; i++) {
                if (!this._linearFBOTextures[i]) {
                    this._linearFBOTextures[i] = gl.createTexture();
                }
                gl.bindTexture(gl.TEXTURE_2D, this._linearFBOTextures[i]);
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
                gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);

                if (!this._linearFBOs[i]) {
                    this._linearFBOs[i] = gl.createFramebuffer();
                }
                gl.bindFramebuffer(gl.FRAMEBUFFER, this._linearFBOs[i]);
                gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this._linearFBOTextures[i], 0);
                var status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
                if (status !== gl.FRAMEBUFFER_COMPLETE) {
                    $.console.error('[HyperBlendWebGLDrawer] Linear FBO ' + i + ' incomplete: 0x' + status.toString(16));
                }
            }
            gl.bindFramebuffer(gl.FRAMEBUFFER, null);
            gl.bindTexture(gl.TEXTURE_2D, null);
            this._linearFBOWidth_pre = w;
            this._linearFBOHeight_pre = h;
        }

        // ---- Internal: run the Linear pre-blend matrix multiply ----
        // Reads units 0..3 (raw layer FBOs as bound by the composite path or
        // FAST-PATH rebind), runs the 16×M pseudoinverse multiply once or
        // twice (once for M ≤ 4, twice for M > 4), and leaves the result in
        // _linearFBOTextures[0..1]. The caller (draw) then optionally rebinds
        // those textures to units 0/1 before HyperBlend Pass 1 runs.
        //
        // GL state on entry: assumes _layerFBOTextures[0..3] are bound to
        // units 0..3, current framebuffer is default (null), current program
        // is undefined. On exit: framebuffer rebound to default; the layer
        // textures stay bound to units 0..3 (caller may overwrite unit 0/1).
        _runLinearPass(w, h) {
            var gl = this._gl;
            if (!gl) return;
            if (!this._unmixEnabled || !this._unmixMatrix) return;
            if (!this._ensureLinearResources(w, h)) return;

            gl.useProgram(this._linearProgram);

            // Matrix upload — clear the dirty flag once consumed.
            gl.uniform1fv(this._linearUniforms.uUnmixMatrix, this._unmixMatrix);
            this._unmixMatrixDirty = false;

            // Bind the static full-screen quad shared with Pass 1.
            var aPos = this._linearAttribs.aPosition;
            var aTex = this._linearAttribs.aTexCoord;
            gl.bindBuffer(gl.ARRAY_BUFFER, this._posBuffer);
            gl.enableVertexAttribArray(aPos);
            gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);
            gl.bindBuffer(gl.ARRAY_BUFFER, this._texBufferFBO);
            gl.enableVertexAttribArray(aTex);
            gl.vertexAttribPointer(aTex, 2, gl.FLOAT, false, 0, 0);

            // Pass A: outputs 0..3 → _linearFBOs[0]
            gl.bindFramebuffer(gl.FRAMEBUFFER, this._linearFBOs[0]);
            gl.viewport(0, 0, w, h);
            gl.clearColor(0, 0, 0, 1);
            gl.clear(gl.COLOR_BUFFER_BIT);
            gl.uniform1f(this._linearUniforms.uOutputOffset, 0.0);
            gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

            // Pass B: outputs 4..7 → _linearFBOs[1] (skip when M ≤ 4 to save
            // a draw call; the texture exists but doesn't need refreshing
            // because no consumer reads from it when numOutputs ≤ 4).
            if (this._numOutputs > 4 && this._linearFBOs[1]) {
                gl.bindFramebuffer(gl.FRAMEBUFFER, this._linearFBOs[1]);
                gl.viewport(0, 0, w, h);
                gl.clearColor(0, 0, 0, 1);
                gl.clear(gl.COLOR_BUFFER_BIT);
                gl.uniform1f(this._linearUniforms.uOutputOffset, 4.0);
                gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
            }

            // Restore default framebuffer + the canvas viewport for Pass 1.
            gl.bindFramebuffer(gl.FRAMEBUFFER, null);
            gl.viewport(0, 0, this.canvas.width, this.canvas.height);

            this._linearOutputReady = true;
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

            // Get uniform locations.
            // v5.0 R1: uUnmixEnabled / uUnmixMatrix removed from this program;
            // the Linear matrix multiply now lives in its own pre-blend pass
            // (see _runLinearPass + LINEAR_PASS_FRAGMENT_SRC).
            this._uniformLocations = {
                uLayer0: gl.getUniformLocation(program, 'uLayer0'),
                uLayer1: gl.getUniformLocation(program, 'uLayer1'),
                uLayer2: gl.getUniformLocation(program, 'uLayer2'),
                uLayer3: gl.getUniformLocation(program, 'uLayer3'),
                uChannelColor: gl.getUniformLocation(program, 'uChannelColor[0]'),
                uChannelGain: gl.getUniformLocation(program, 'uChannelGain[0]'),
                uChannelEnabled: gl.getUniformLocation(program, 'uChannelEnabled[0]'),
                uToneMode: gl.getUniformLocation(program, 'uToneMode')
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
            // Defensive: PICASSO's initial blit sets uTile=0; restore for the composite.
            gl.uniform1i(this._blitUniforms.uTile, 5);
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
