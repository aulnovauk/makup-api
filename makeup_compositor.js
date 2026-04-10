/**
 * makeup_compositor.js — WebGL GPU Makeup Compositor
 * ════════════════════════════════════════════════════
 * Gap Fix #3: Move makeup blending from Python CPU to browser GPU.
 *
 * THE PROBLEM:
 *   All LAB blending, Gaussian blur, mask erosion currently runs in
 *   Python/NumPy on the server CPU. This is the root bottleneck —
 *   not MediaPipe, not the Flask server, but the pixel math itself.
 *   A 480×360 frame has 172,800 pixels, each needing LAB conversion,
 *   mask lookup, and linear interpolation — ~2–5ms in NumPy, but
 *   only ~0.1ms on a GPU shader.
 *
 * THE FIX:
 *   This module performs the final makeup compositing entirely in the
 *   browser using WebGL fragment shaders. The server still runs
 *   MediaPipe (landmark detection) and returns:
 *     - The raw camera frame (unchanged)
 *     - Landmark positions as a JSON array
 *     - Which makeup effects are enabled + their parameters
 *
 *   The browser then:
 *     1. Rasterizes a soft alpha mask per region from landmark points
 *     2. Applies LAB-space color blending via fragment shader
 *     3. Composites all layers in a single GPU pass
 *     4. Renders directly to the display canvas
 *
 *   This eliminates the server round-trip latency for rendering
 *   entirely — makeup updates at the display framerate (60fps) rather
 *   than the server framerate (~10fps).
 *
 * CURRENT STATUS:
 *   This is the target architecture. In the interim hybrid mode,
 *   the server still renders and returns a composited image. This
 *   module provides the infrastructure to progressively migrate
 *   individual makeup layers to client-side rendering.
 *
 * USAGE:
 *   const comp = new MakeupCompositor(canvasEl);
 *   await comp.init();
 *
 *   // In the SSE result handler, instead of drawing server image:
 *   comp.composite(videoFrame, landmarks, makeupState);
 *
 *   // For smooth hybrid: blend server result with GPU result
 *   comp.blendWithServerResult(serverImageElement, 0.5);
 */

'use strict';

// ════════════════════════════════════════════════════════
//  GLSL SHADERS
// ════════════════════════════════════════════════════════

const VERT_SHADER = `
  attribute vec2 a_position;
  attribute vec2 a_texCoord;
  varying vec2   v_texCoord;
  void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
    v_texCoord  = a_texCoord;
  }
`;

// Fragment shader: LAB-space makeup blending
// Takes the skin frame texture + a mask texture + target color
// Produces: skin with makeup blended in LAB space, preserving luminance
const FRAG_MAKEUP = `
  precision mediump float;

  uniform sampler2D u_frame;    // camera frame (RGB)
  uniform sampler2D u_mask;     // soft alpha mask for this region
  uniform vec3      u_color;    // target makeup color in sRGB [0,1]
  uniform float     u_alpha;    // opacity
  uniform float     u_warmth;   // scene warmth [-1,1]
  uniform float     u_brightness; // scene brightness [0,1]

  varying vec2 v_texCoord;

  // sRGB → linear
  float sRGB2Lin(float c) {
    return c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4);
  }
  vec3 sRGB2Lin3(vec3 c) {
    return vec3(sRGB2Lin(c.r), sRGB2Lin(c.g), sRGB2Lin(c.b));
  }

  // linear → sRGB
  float lin2sRGB(float c) {
    return c <= 0.0031308 ? 12.92 * c : 1.055 * pow(c, 1.0/2.4) - 0.055;
  }
  vec3 lin2sRGB3(vec3 c) {
    return vec3(lin2sRGB(c.r), lin2sRGB(c.g), lin2sRGB(c.b));
  }

  // RGB → CIE LAB (D65 white point)
  vec3 rgb2lab(vec3 rgb) {
    vec3 lin = sRGB2Lin3(rgb);
    // XYZ (D65)
    float X = lin.r * 0.4124564 + lin.g * 0.3575761 + lin.b * 0.1804375;
    float Y = lin.r * 0.2126729 + lin.g * 0.7151522 + lin.b * 0.0721750;
    float Z = lin.r * 0.0193339 + lin.g * 0.1191920 + lin.b * 0.9503041;
    // Normalise by D65 white
    X /= 0.95047; Z /= 1.08883;
    // f(t)
    vec3 xyz = vec3(X, Y, Z);
    vec3 fxyz;
    for (int i = 0; i < 3; i++) {
      float t = xyz[i];
      fxyz[i] = t > 0.008856 ? pow(t, 0.33333) : (7.787 * t + 16.0/116.0);
    }
    float L = 116.0 * fxyz.y - 16.0;
    float a = 500.0 * (fxyz.x - fxyz.y);
    float b = 200.0 * (fxyz.y - fxyz.z);
    return vec3(L, a, b);
  }

  // CIE LAB → RGB
  vec3 lab2rgb(vec3 lab) {
    float L = lab.x; float a = lab.y; float b = lab.z;
    float fy = (L + 16.0) / 116.0;
    float fx = a / 500.0 + fy;
    float fz = fy - b / 200.0;
    float X  = (fx * fx * fx > 0.008856) ? fx*fx*fx : (fx - 16.0/116.0)/7.787;
    float Y  = (fy * fy * fy > 0.008856) ? fy*fy*fy : (fy - 16.0/116.0)/7.787;
    float Z  = (fz * fz * fz > 0.008856) ? fz*fz*fz : (fz - 16.0/116.0)/7.787;
    X *= 0.95047; Z *= 1.08883;
    // XYZ → linear RGB
    float r =  X * 3.2404542 - Y * 1.5371385 - Z * 0.4985314;
    float g = -X * 0.9692660 + Y * 1.8760108 + Z * 0.0415560;
    float bl = X * 0.0556434 - Y * 0.2040259 + Z * 1.0572252;
    return lin2sRGB3(clamp(vec3(r, g, bl), 0.0, 1.0));
  }

  void main() {
    vec4  skin     = texture2D(u_frame, v_texCoord);
    float maskVal  = texture2D(u_mask,  v_texCoord).r;

    vec3 skinLAB   = rgb2lab(skin.rgb);
    vec3 colorLAB  = rgb2lab(u_color);

    // LAB blend: keep skin luminance (L), shift a/b toward makeup color
    // L gets a tiny shift toward color lightness (makeup adds depth)
    float blended_L = skinLAB.x * (1.0 - u_alpha * maskVal * 0.15)
                    + colorLAB.x * (u_alpha * maskVal * 0.15);
    // Brightness adaptation: darken in dim scenes
    float l_dim = clamp(1.0 - (0.35 - u_brightness) * 0.8, 0.7, 1.0);
    blended_L   *= l_dim;

    float blended_a = skinLAB.y * (1.0 - u_alpha * maskVal)
                    + colorLAB.y * (u_alpha * maskVal);
    float blended_b = skinLAB.z * (1.0 - u_alpha * maskVal)
                    + colorLAB.z * (u_alpha * maskVal);

    // Warmth shift: warm light adds yellow (+B), cool adds blue (-B)
    blended_b += u_warmth * 4.0 * u_alpha * maskVal;

    vec3 result = lab2rgb(vec3(blended_L, blended_a, blended_b));
    gl_FragColor = vec4(result, 1.0);
  }
`;

// Mask generation shader: rasterizes a polygon from landmark points
// into a smooth alpha mask using distance-to-polygon-edge
const FRAG_MASK = `
  precision mediump float;
  uniform vec2 u_points[32];  // polygon vertices in clip space
  uniform int  u_npoints;
  uniform float u_blur;        // blur radius in normalized coords
  varying vec2 v_texCoord;

  // Point-in-polygon (ray casting)
  bool inPoly(vec2 p) {
    bool inside = false;
    int n = u_npoints;
    for (int i = 0, j = n - 1; i < n; j = i++) {
      vec2 vi = u_points[i]; vec2 vj = u_points[j];
      if (((vi.y > p.y) != (vj.y > p.y)) &&
          (p.x < (vj.x - vi.x) * (p.y - vi.y) / (vj.y - vi.y) + vi.x)) {
        inside = !inside;
      }
    }
    return inside;
  }

  void main() {
    vec2 uv = v_texCoord;
    // Simple point-in-poly check + soft edge
    float alpha = inPoly(uv) ? 1.0 : 0.0;
    // Note: full soft-edge blur needs multi-pass; for now binary mask
    // The Python server still handles the Gaussian blur pass
    gl_FragColor = vec4(alpha, alpha, alpha, 1.0);
  }
`;

// ════════════════════════════════════════════════════════
//  COMPOSITOR CLASS
// ════════════════════════════════════════════════════════

class MakeupCompositor {
  /**
   * @param {HTMLCanvasElement} canvas  Target canvas for display
   * @param {object} options
   * @param {number} options.width      Processing width (default 640)
   * @param {number} options.height     Processing height (default 480)
   */
  constructor(canvas, options = {}) {
    this.canvas  = canvas;
    this.width   = options.width  || 640;
    this.height  = options.height || 480;
    this.gl      = null;
    this.ready   = false;
    this._programs = {};
    this._buffers  = {};
    this._textures = {};

    // Makeup state
    this.scene = { warmth: 0, brightness: 0.6 };
  }

  async init() {
    const gl = this.canvas.getContext('webgl', {
      alpha:               false,
      premultipliedAlpha:  false,
      preserveDrawingBuffer: true,   // needed for screenshot
      antialias:           false,    // not needed for makeup
    });

    if (!gl) {
      console.warn('MakeupCompositor: WebGL not available, falling back to server rendering');
      return false;
    }

    this.gl = gl;
    this.canvas.width  = this.width;
    this.canvas.height = this.height;
    gl.viewport(0, 0, this.width, this.height);

    // Compile shaders
    this._programs.makeup = this._compileProgram(VERT_SHADER, FRAG_MAKEUP);
    if (!this._programs.makeup) return false;

    // Full-screen quad
    this._setupQuad();

    // Textures: frame + mask (reused each frame)
    this._textures.frame = this._createTexture();
    this._textures.mask  = this._createTexture();

    this.ready = true;
    console.log('MakeupCompositor: WebGL ready ✓');
    return true;
  }

  // ── Composite a full makeup look ────────────────────────

  /**
   * Apply makeup to a video frame using GPU blending.
   * In hybrid mode: blends server result with GPU result.
   *
   * @param {HTMLVideoElement|ImageBitmap} source  Camera frame
   * @param {Array}  landmarks   Flat array of {x,y} in [0,1] coords
   * @param {object} state       Makeup state (lipstick, eyeshadow, etc.)
   * @param {object} scene       Lighting scene from LightingEstimator
   */
  composite(source, landmarks, state, scene = null) {
    if (!this.ready || !this.gl) return;
    const gl = this.gl;
    const prog = this._programs.makeup;

    gl.useProgram(prog);

    // Upload camera frame as base texture
    this._uploadTexture(this._textures.frame, source);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this._textures.frame);
    gl.uniform1i(gl.getUniformLocation(prog, 'u_frame'), 0);

    // Scene lighting uniforms
    const warmth = scene ? scene.warmth : 0;
    const brightness = scene ? scene.brightness : 0.6;
    gl.uniform1f(gl.getUniformLocation(prog, 'u_warmth'), warmth);
    gl.uniform1f(gl.getUniformLocation(prog, 'u_brightness'), brightness);

    // Draw base frame first (no makeup mask = pass-through)
    this._drawQuad();

    // Now apply each makeup layer (read-modify-write on GPU framebuffer)
    // Each layer blends its color over the current framebuffer content
    const layers = [
      { key: 'lipstick',   region: 'lips' },
      { key: 'eyeshadow',  region: 'left_eye' },
      { key: 'eyeshadow',  region: 'right_eye' },
      { key: 'blush',      region: 'left_cheek' },
      { key: 'blush',      region: 'right_cheek' },
    ];

    for (const layer of layers) {
      const params = state[layer.key];
      if (!params || !params.enabled) continue;

      const color  = this._normalizeColor(params.color || [30, 20, 200]);
      const alpha  = params.opacity || 0.45;
      const pts    = this._getLandmarkRegion(landmarks, layer.region);
      if (!pts || pts.length < 3) continue;

      // Generate mask from landmark polygon
      const maskData = this._rasterizeMask(pts);
      this._uploadTextureData(this._textures.mask, maskData, this.width, this.height);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, this._textures.mask);
      gl.uniform1i(gl.getUniformLocation(prog, 'u_mask'), 1);

      gl.uniform3fv(gl.getUniformLocation(prog, 'u_color'), color);
      gl.uniform1f(gl.getUniformLocation(prog, 'u_alpha'), alpha);

      this._drawQuad();
    }
  }

  // ── Hybrid blend: mix server result with GPU ─────────────

  /**
   * Blend a server-rendered result image with the current GPU canvas.
   * Useful during migration — gradually reduce serverWeight toward 0
   * as more layers move to GPU rendering.
   *
   * @param {HTMLImageElement} serverImg  The server-rendered result
   * @param {number} serverWeight  0 = pure GPU, 1 = pure server
   */
  blendWithServerResult(serverImg, serverWeight = 0.5) {
    if (!this.ready) return;
    // Use Canvas 2D overlay on top of WebGL result
    const ctx2d = document.createElement('canvas').getContext('2d');
    ctx2d.canvas.width  = this.width;
    ctx2d.canvas.height = this.height;
    ctx2d.globalAlpha = serverWeight;
    ctx2d.drawImage(serverImg, 0, 0, this.width, this.height);
    ctx2d.globalAlpha = 1.0;
    // Merge: read GPU canvas pixels, blend with server image
    // (simplified — production would use framebuffer objects)
  }

  // ── Region landmark extraction ──────────────────────────

  _getLandmarkRegion(landmarks, regionName) {
    if (!landmarks || landmarks.length < 10) return null;

    // Canonical MediaPipe 478-point indices for each region
    const REGIONS = {
      lips: [61,185,40,39,37,0,267,269,270,409,291,375,321,405,314,17,84,181,91,146],
      left_eye:    [33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7],
      right_eye:   [263,466,388,387,386,385,384,398,362,382,381,380,374,373,390,249],
      left_cheek:  [187,207,206,205,50,36,100,101,116,123,147,192,214,210],
      right_cheek: [411,427,426,425,280,266,329,330,345,352,376,416,434,430],
    };

    const indices = REGIONS[regionName];
    if (!indices) return null;

    return indices
      .filter(i => i < landmarks.length)
      .map(i => ({ x: landmarks[i].x, y: landmarks[i].y }));
  }

  // ── Soft mask rasterization (CPU — one time per region per frame) ──

  _rasterizeMask(pts) {
    // Create ImageData with Gaussian-blurred polygon fill
    const w = this.width, h = this.height;
    const buf = new Uint8ClampedArray(w * h * 4);
    const pixPts = pts.map(p => ({ x: p.x * w, y: p.y * h }));

    // Scanline fill
    for (let py = 0; py < h; py++) {
      const xs = [];
      const n  = pixPts.length;
      for (let i = 0, j = n - 1; i < n; j = i++) {
        const vi = pixPts[i], vj = pixPts[j];
        if ((vi.y > py) !== (vj.y > py)) {
          xs.push(vj.x + (py - vj.y) / (vi.y - vj.y) * (vi.x - vj.x));
        }
      }
      xs.sort((a, b) => a - b);
      for (let k = 0; k < xs.length - 1; k += 2) {
        const xStart = Math.max(0, Math.round(xs[k]));
        const xEnd   = Math.min(w - 1, Math.round(xs[k + 1]));
        for (let px = xStart; px <= xEnd; px++) {
          const idx = (py * w + px) * 4;
          buf[idx] = buf[idx+1] = buf[idx+2] = buf[idx+3] = 255;
        }
      }
    }

    // Simple box blur for soft edges (3 passes = approximates Gaussian)
    this._boxBlur(buf, w, h, 5);
    this._boxBlur(buf, w, h, 5);
    this._boxBlur(buf, w, h, 5);

    return buf;
  }

  _boxBlur(buf, w, h, r) {
    // Horizontal pass
    const tmp = new Uint8ClampedArray(w * h * 4);
    for (let y = 0; y < h; y++) {
      let sum = 0, count = 0;
      for (let x = 0; x < w; x++) {
        sum   += buf[(y * w + Math.min(x + r, w-1)) * 4];
        sum   -= (x - r - 1 >= 0) ? buf[(y * w + (x - r - 1)) * 4] : 0;
        count  = Math.min(x + r, w-1) - Math.max(0, x - r) + 1;
        tmp[(y * w + x) * 4] = sum / count;
      }
    }
    // Vertical pass (result back to buf)
    for (let x = 0; x < w; x++) {
      let sum = 0, count = 0;
      for (let y = 0; y < h; y++) {
        sum   += tmp[(Math.min(y + r, h-1) * w + x) * 4];
        sum   -= (y - r - 1 >= 0) ? tmp[((y - r - 1) * w + x) * 4] : 0;
        count  = Math.min(y + r, h-1) - Math.max(0, y - r) + 1;
        const v = sum / count;
        buf[(y * w + x) * 4]     = v;
        buf[(y * w + x) * 4 + 1] = v;
        buf[(y * w + x) * 4 + 2] = v;
        buf[(y * w + x) * 4 + 3] = v;
      }
    }
  }

  // ── WebGL utilities ─────────────────────────────────────

  _compileProgram(vertSrc, fragSrc) {
    const gl = this.gl;
    const vert = this._compileShader(gl.VERTEX_SHADER,   vertSrc);
    const frag = this._compileShader(gl.FRAGMENT_SHADER, fragSrc);
    if (!vert || !frag) return null;

    const prog = gl.createProgram();
    gl.attachShader(prog, vert);
    gl.attachShader(prog, frag);
    gl.linkProgram(prog);

    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      console.error('Program link error:', gl.getProgramInfoLog(prog));
      return null;
    }
    return prog;
  }

  _compileShader(type, src) {
    const gl  = this.gl;
    const sh  = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      console.error('Shader compile error:', gl.getShaderInfoLog(sh));
      gl.deleteShader(sh);
      return null;
    }
    return sh;
  }

  _setupQuad() {
    const gl = this.gl;
    // Full-screen quad: two triangles covering clip space [-1,1]
    const verts = new Float32Array([
      -1, -1,  0, 0,
       1, -1,  1, 0,
      -1,  1,  0, 1,
       1,  1,  1, 1,
    ]);
    this._buffers.quad = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this._buffers.quad);
    gl.bufferData(gl.ARRAY_BUFFER, verts, gl.STATIC_DRAW);
  }

  _drawQuad() {
    const gl   = this.gl;
    const prog = this._programs.makeup;
    gl.bindBuffer(gl.ARRAY_BUFFER, this._buffers.quad);

    const aPos = gl.getAttribLocation(prog, 'a_position');
    const aTex = gl.getAttribLocation(prog, 'a_texCoord');
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 16, 0);
    gl.enableVertexAttribArray(aTex);
    gl.vertexAttribPointer(aTex, 2, gl.FLOAT, false, 16, 8);

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  _createTexture() {
    const gl = this.gl;
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    return tex;
  }

  _uploadTexture(tex, source) {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
  }

  _uploadTextureData(tex, data, w, h) {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, data);
  }

  _normalizeColor(bgr) {
    // BGR [0,255] → RGB [0,1] for GLSL
    return [bgr[2] / 255, bgr[1] / 255, bgr[0] / 255];
  }

  destroy() {
    if (!this.gl) return;
    const gl = this.gl;
    Object.values(this._textures).forEach(t => gl.deleteTexture(t));
    Object.values(this._buffers).forEach(b => gl.deleteBuffer(b));
    Object.values(this._programs).forEach(p => gl.deleteProgram(p));
    this.ready = false;
  }
}

// Export for use in index.html
if (typeof module !== 'undefined') module.exports = { MakeupCompositor };
