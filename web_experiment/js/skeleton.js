// skeleton.js — drawing MediaPipe frames and animating landmark sequences.

export const POSE_EDGES = [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24]];
export const HAND_EDGES = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
];
const FACE = [0, 2, 5, 7, 8, 9, 10];

function css(name, fallback) {
  try { return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback; } catch { return fallback; }
}
export const colors = () => ({ ref: css('--ref', '#2563eb'), you: css('--you', '#d9480f'), live: css('--live', '#16a34a') });

/**
 * Draw one frame. `map(p) -> [x, y]` converts a landmark to canvas pixels.
 */
export function drawFrame(ctx, frame, map, color, { lw = 3, alpha = 1, hipLimit = true } = {}) {
  if (!frame) return;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineCap = 'round'; ctx.lineWidth = lw;
  const line = (a, b) => { const p = map(a), q = map(b); ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(q[0], q[1]); ctx.stroke(); };
  const dot = (a, r) => { const p = map(a); ctx.beginPath(); ctx.arc(p[0], p[1], r, 0, 2 * Math.PI); ctx.fill(); };
  if (frame.pose) {
    const P = frame.pose;
    for (const [a, b] of POSE_EDGES) {
      if (hipLimit && (a >= 23 || b >= 23) && P[a].length > 3 && P[a][3] != null && P[a][3] < 0.3) continue;
      line(P[a], P[b]);
    }
    for (const i of FACE) dot(P[i], lw * 0.8);
    // wrists from pose are drawn only if no hand is detected
    if (!frame.left_hand) dot(P[15], lw * 1.3);
    if (!frame.right_hand) dot(P[16], lw * 1.3);
  }
  ctx.lineWidth = Math.max(1, lw * 0.6);
  for (const h of [frame.left_hand, frame.right_hand]) {
    if (!h) continue;
    for (const [a, b] of HAND_EDGES) line(h[a], h[b]);
    for (const p of h) dot(p, Math.max(1, lw * 0.5));
  }
  ctx.restore();
}

/** Mapping for normalized image coords onto a canvas showing the image (optionally mirrored). */
export const imageMap = (w, h, mirror) => (p) => [mirror ? (1 - p[0]) * w : p[0] * w, p[1] * h];

/** Mapping for shoulder-normalized coords (see similarity.normalizeSequence). */
export const normMap = (w, h, mirror = false) => {
  const s = Math.min(w / 4.4, h / 4.2);
  return (p) => [w / 2 + (mirror ? -p[0] : p[0]) * s, h * 0.36 + p[1] * s];
};

/**
 * Animate one or more landmark sequences on a canvas.
 * tracks: [{ frames, color, label, map? }]  (frames carry .t in ms)
 * alignment (optional): array of [i, j] index pairs into tracks[0] / tracks[1] (DTW path),
 * with times: { a: [ms...], b: [ms...] } giving the timestamps the indices refer to.
 */
export class SkeletonPlayer {
  constructor(canvas, { mirror = false } = {}) {
    this.canvas = canvas; this.ctx = canvas.getContext('2d');
    this.tracks = []; this.alignment = null; this.mirror = mirror;
    this.playing = false; this.speed = 1; this.loop = true; this._raf = null; this.pos = 0;
  }
  set(tracks, { alignment = null, times = null } = {}) {
    this.tracks = tracks.filter((t) => t && t.frames && t.frames.length);
    this.alignment = alignment; this.times = times; this.pos = 0;
    this.duration = Math.max(...this.tracks.map((t) => t.frames[t.frames.length - 1].t - t.frames[0].t), 1);
    this.draw();
  }
  _frameAtTime(frames, ms) {
    const t0 = frames[0].t; let lo = 0, hi = frames.length - 1;
    while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (frames[mid].t - t0 <= ms) lo = mid; else hi = mid - 1; }
    return frames[lo];
  }
  _frameAtT(frames, t) { return this._frameAtTime(frames, t - frames[0].t); }
  draw() {
    const { canvas, ctx } = this;
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth || canvas.width, h = canvas.clientHeight || canvas.height;
    if (canvas.width !== Math.round(w * dpr)) { canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr); }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    if (!this.tracks.length) return;
    const lw = Math.max(2, Math.min(w, h) / 120);
    this.tracks.forEach((tr, k) => {
      let f;
      if (this.alignment && this.tracks.length === 2) {
        const step = this.alignment[Math.min(this.alignment.length - 1, Math.floor(this.pos * this.alignment.length))];
        const tt = (k === 0 ? this.times.a : this.times.b)[step[k]];
        f = this._frameAtT(tr.frames, tt);
      } else {
        f = this._frameAtTime(tr.frames, this.pos * this.duration);
      }
      const map = tr.map ? tr.map(w, h) : normMap(w, h, this.mirror);
      drawFrame(ctx, f, map, tr.color, { lw, alpha: tr.alpha ?? 0.95 });
    });
  }
  play() {
    if (this.playing) return; this.playing = true;
    let last = performance.now();
    const tick = (now) => {
      if (!this.playing) return;
      const dur = this.alignment ? Math.max(this.duration, this.alignment.length * 40) : this.duration;
      this.pos += ((now - last) * this.speed) / dur; last = now;
      if (this.pos >= 1) { if (this.loop) this.pos = 0; else { this.pos = 1; this.playing = false; } }
      this.draw(); if (this.onTick) this.onTick(this.pos);
      this._raf = requestAnimationFrame(tick);
    };
    this._raf = requestAnimationFrame(tick);
  }
  pause() { this.playing = false; if (this._raf) cancelAnimationFrame(this._raf); }
  seek(p) { this.pos = Math.max(0, Math.min(1, p)); this.draw(); }
  destroy() { this.pause(); }
}
