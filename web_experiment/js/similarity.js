// similarity.js — pure functions (no DOM), usable in the browser and in Node.
//
// Input sequences use the same per-frame structure as the asl_telicity Python
// pipeline (MediaPipe Holistic-style):
//   { t: ms, pose: [[x,y,z,vis] x33] | null,
//     left_hand: [[x,y,z] x21] | null, right_hand: [[x,y,z] x21] | null }
// with x,y in normalized image coordinates (0..1). `aspect` = width/height of
// the source image, so x can be put on the same scale as y.
//
// Pipeline:  normalize (shoulder-centred, shoulder-width units)
//         -> per-frame features (wrists, elbows, hand shapes)
//         -> trim to the active signing window (hands raised)
//         -> resample to a fixed rate
//         -> DTW, computed directly and mirrored (left/right swapped); best is kept
//         -> score = 100 * exp(-distance / tau)

export const POSE = { NOSE: 0, L_SH: 11, R_SH: 12, L_EL: 13, R_EL: 14, L_WR: 15, R_WR: 16, L_HIP: 23, R_HIP: 24 };

export const DEFAULT_SIM = {
  resampleHz: 25,
  trim: 'hands',           // 'hands': keep first..last frame with a tracked hand (fallback: 'height')
                           // 'hands+height': a tracked hand AND a wrist raised above activeYThreshold
                           // 'height': wrist raised above activeYThreshold;  'none': keep everything
  activeYThreshold: 1.6,   // wrist above (shoulder line + 1.6 shoulder widths) = raised
  activePadFrames: 3,
  weights: { wrist: 1.0, elbow: 0.5, hand: 0.6 },
  handMissingPenalty: 0.6, // handshape cost when one sequence sees the hand and the other doesn't
  tau: 0.6,                // distance -> score scale (calibrated, see test/similarity_test.mjs)
  allowMirror: true,
};

const median = (a) => {
  if (!a.length) return NaN;
  const s = [...a].sort((x, y) => x - y);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};
const vis = (p, i) => (p[i].length > 3 && p[i][3] != null ? p[i][3] : 1);

/** Shoulder-centred, shoulder-width-scaled copy of a sequence (keeps every landmark). */
export function normalizeSequence(frames, aspect = 1) {
  const widths = [];
  for (const f of frames) {
    if (!f.pose) continue;
    const a = f.pose[POSE.L_SH], b = f.pose[POSE.R_SH];
    if (vis(f.pose, POSE.L_SH) < 0.3 || vis(f.pose, POSE.R_SH) < 0.3) continue;
    const w = Math.hypot((a[0] - b[0]) * aspect, a[1] - b[1]);
    if (w > 1e-4) widths.push(w);
  }
  let scale = median(widths);
  if (!isFinite(scale)) {
    const any = frames.filter((f) => f.pose).map((f) =>
      Math.hypot((f.pose[11][0] - f.pose[12][0]) * aspect, f.pose[11][1] - f.pose[12][1]));
    scale = median(any);
  }
  if (!isFinite(scale) || scale < 1e-4) return null;

  let cx = null, cy = null;
  const out = [];
  for (let i = 0; i < frames.length; i++) {
    const f = frames[i];
    if (f.pose) {
      const a = f.pose[POSE.L_SH], b = f.pose[POSE.R_SH];
      cx = ((a[0] + b[0]) / 2) * aspect;
      cy = (a[1] + b[1]) / 2;
    }
    if (cx === null) { out.push({ t: f.t ?? i * 40, pose: null, left_hand: null, right_hand: null }); continue; }
    const tp = (p) => [(p[0] * aspect - cx) / scale, (p[1] - cy) / scale, ((p[2] || 0) * aspect) / scale, p.length > 3 ? p[3] : 1];
    out.push({
      t: f.t ?? i * 40,
      pose: f.pose ? f.pose.map(tp) : null,
      left_hand: f.left_hand ? f.left_hand.map(tp) : null,
      right_hand: f.right_hand ? f.right_hand.map(tp) : null,
    });
  }
  return { frames: out, scale };
}

function handShape(hand) {
  if (!hand) return null;
  const w = hand[0], m = hand[9];
  const size = Math.hypot(m[0] - w[0], m[1] - w[1]);
  if (size < 1e-4) return null;
  const v = new Float32Array(40);
  for (let k = 1; k < 21; k++) {
    v[(k - 1) * 2] = (hand[k][0] - w[0]) / size;
    v[(k - 1) * 2 + 1] = (hand[k][1] - w[1]) / size;
  }
  return v;
}

/** Per-frame feature dicts from a normalized sequence. Frames without pose are dropped. */
export function extractFeatures(nframes) {
  const feats = [];
  for (const f of nframes) {
    if (!f.pose) continue;
    const p = f.pose;
    const wl = f.left_hand ? f.left_hand[0] : p[POSE.L_WR];
    const wr = f.right_hand ? f.right_hand[0] : p[POSE.R_WR];
    feats.push({
      t: f.t,
      wL: [wl[0], wl[1]], wR: [wr[0], wr[1]],
      eL: [p[POSE.L_EL][0], p[POSE.L_EL][1]], eR: [p[POSE.R_EL][0], p[POSE.R_EL][1]],
      hL: handShape(f.left_hand), hR: handShape(f.right_hand),
    });
  }
  return feats;
}

/** Keep frames between first and last "hands raised" frame (+ padding). */
export function trimActive(feats, thr = DEFAULT_SIM.activeYThreshold, pad = DEFAULT_SIM.activePadFrames) {
  let first = -1, last = -1;
  feats.forEach((f, i) => {
    if (Math.min(f.wL[1], f.wR[1]) < thr) { if (first < 0) first = i; last = i; }
  });
  if (first < 0) return { feats, trimmed: false, start: 0, end: feats.length - 1 };
  const s = Math.max(0, first - pad), e = Math.min(feats.length - 1, last + pad);
  return { feats: feats.slice(s, e + 1), trimmed: true, start: s, end: e };
}

/** Keep the first..last frame that passes `isActive` (+ padding); unchanged if none pass. */
function trimBy(feats, isActive, pad) {
  let first = -1, last = -1;
  feats.forEach((f, i) => { if (isActive(f)) { if (first < 0) first = i; last = i; } });
  if (first < 0) return null;
  const s = Math.max(0, first - pad), e = Math.min(feats.length - 1, last + pad);
  return { feats: feats.slice(s, e + 1), trimmed: true, start: s, end: e };
}

/** Trim the idle start/end of a sequence according to o.trim (see DEFAULT_SIM). */
export function trimSequence(feats, o = DEFAULT_SIM) {
  const pad = o.activePadFrames, thr = o.activeYThreshold;
  const hand = (f) => !!(f.hL || f.hR);
  const raised = (f) => Math.min(f.wL[1], f.wR[1]) < thr;
  const none = { feats, trimmed: false, start: 0, end: feats.length - 1 };
  if (o.trim === 'none') return none;
  if (o.trim === 'height') return trimActive(feats, thr, pad);
  if (o.trim === 'hands+height') return trimBy(feats, (f) => hand(f) && raised(f), pad) || trimActive(feats, thr, pad);
  return trimBy(feats, hand, pad) || trimActive(feats, thr, pad); // 'hands' (default)
}

/** Nearest-frame resampling on the timestamp axis. */
export function resample(feats, hz = 25) {
  if (feats.length < 2) return feats;
  const t0 = feats[0].t, t1 = feats[feats.length - 1].t, dt = 1000 / hz;
  const out = [];
  let j = 0;
  for (let t = t0; t <= t1 + 1e-6; t += dt) {
    while (j + 1 < feats.length && Math.abs(feats[j + 1].t - t) <= Math.abs(feats[j].t - t)) j++;
    out.push(feats[j]);
  }
  return out;
}

export function mirrorFeatures(feats) {
  const mp = (p) => [-p[0], p[1]];
  const mh = (h) => { if (!h) return null; const v = new Float32Array(h); for (let k = 0; k < 40; k += 2) v[k] = -v[k]; return v; };
  return feats.map((f) => ({ t: f.t, wL: mp(f.wR), wR: mp(f.wL), eL: mp(f.eR), eR: mp(f.eL), hL: mh(f.hR), hR: mh(f.hL) }));
}

const d2 = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
function handDist(a, b) {
  let s = 0;
  for (let k = 0; k < 40; k += 2) s += Math.hypot(a[k] - b[k], a[k + 1] - b[k + 1]);
  return s / 20;
}

function frameCost(a, b, o) {
  const wrist = (d2(a.wL, b.wL) + d2(a.wR, b.wR)) / 2;
  const elbow = (d2(a.eL, b.eL) + d2(a.eR, b.eR)) / 2;
  let hs = 0, n = 0;
  for (const [x, y] of [[a.hL, b.hL], [a.hR, b.hR]]) {
    if (x && y) { hs += handDist(x, y); n++; }
    else if (x || y) { hs += o.handMissingPenalty; n++; }
  }
  const hand = n ? hs / n : 0;
  const w = o.weights;
  return { total: (w.wrist * wrist + w.elbow * elbow + w.hand * hand) / (w.wrist + w.elbow + w.hand), wrist, elbow, hand };
}

/** Classic DTW; returns path-length-normalized cost plus component means along the path. */
export function dtw(A, B, o = DEFAULT_SIM) {
  const n = A.length, m = B.length;
  if (!n || !m) return null;
  const W = m + 1;
  const D = new Float64Array((n + 1) * W).fill(Infinity);
  const C = new Array(n * m);
  D[0] = 0;
  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      const c = frameCost(A[i - 1], B[j - 1], o);
      C[(i - 1) * m + (j - 1)] = c;
      D[i * W + j] = c.total + Math.min(D[(i - 1) * W + j], D[i * W + j - 1], D[(i - 1) * W + j - 1]);
    }
  }
  const path = [];
  let i = n, j = m;
  const comp = { wrist: 0, elbow: 0, hand: 0 };
  while (i > 0 && j > 0) {
    path.push([i - 1, j - 1]);
    const c = C[(i - 1) * m + (j - 1)];
    comp.wrist += c.wrist; comp.elbow += c.elbow; comp.hand += c.hand;
    const diag = D[(i - 1) * W + j - 1], up = D[(i - 1) * W + j], left = D[i * W + j - 1];
    if (diag <= up && diag <= left) { i--; j--; } else if (up < left) i--; else j--;
  }
  path.reverse();
  const L = path.length;
  return {
    distance: D[n * W + m] / L, path,
    components: { wrist: comp.wrist / L, elbow: comp.elbow / L, hand: comp.hand / L },
  };
}

/** Prepare a raw sequence: normalize, extract features, trim, resample. */
export function prepare(frames, aspect, o = DEFAULT_SIM) {
  const norm = normalizeSequence(frames, aspect);
  if (!norm) return null;
  const all = extractFeatures(norm.frames);
  const tr = trimSequence(all, o);
  return { norm, feats: resample(tr.feats, o.resampleHz), trim: tr };
}

export const toScore = (d, tau = DEFAULT_SIM.tau) => (d == null || !isFinite(d) ? null : 100 * Math.exp(-d / tau));

/**
 * Compare a reference sequence with a participant sequence.
 * Returns { score, distance, mirrored, direct, mirror, ... } (score in 0..100, higher = more similar).
 */
export function compareSequences(refFrames, refAspect, partFrames, partAspect, opts = {}) {
  const o = { ...DEFAULT_SIM, ...opts, weights: { ...DEFAULT_SIM.weights, ...(opts.weights || {}) } };
  const R = prepare(refFrames, refAspect, o);
  const P = prepare(partFrames, partAspect, o);
  if (!R || !P || R.feats.length < 2 || P.feats.length < 2) {
    return { score: null, distance: null, error: !P ? 'participant: no usable pose' : 'not enough frames' };
  }
  const direct = dtw(R.feats, P.feats, o);
  const mirror = o.allowMirror ? dtw(R.feats, mirrorFeatures(P.feats), o) : null;
  const useMirror = !!mirror && mirror.distance < direct.distance;
  const best = useMirror ? mirror : direct;
  const pack = (r) => r && { distance: r.distance, score: toScore(r.distance, o.tau), components: r.components };
  return {
    score: toScore(best.distance, o.tau),
    distance: best.distance,
    mirrored: useMirror,
    components: best.components,
    direct: pack(direct),
    mirror: pack(mirror),
    nRefFrames: R.feats.length,
    nPartFrames: P.feats.length,
    tau: o.tau,
    // not saved to data by default (large); used by the results viewer for aligned playback
    _path: best.path,
    _refTimes: R.feats.map((f) => f.t),
    _partTimes: P.feats.map((f) => f.t),
  };
}
