// framing.js — checks that the participant's shoulders and both hands are inside a
// centre box at a sensible distance from the camera. Used by the camera check
// (blocking) and by the recorder's ready screen (advisory).
// All coordinates are normalized image coords; the box is symmetric, so it is the same
// in the mirrored preview and in the unmirrored image the tracker sees.

export const DEFAULT_FRAMING = {
  box: { x0: 0.2, x1: 0.8, y0: 0.06, y1: 0.96 }, // centre area (fractions of width / height)
  shoulderY: [0.2, 0.7],          // shoulder line must be in this vertical range (head fits above)
  shoulderWidth: [0.15, 0.5],     // fraction of image width: smaller = too far, larger = too close
  requireHands: true,             // both hands must be detected inside the box
  holdMs: 1000,                   // all checks must pass continuously this long
};

const inBox = (p, b) => p[0] >= b.x0 && p[0] <= b.x1 && p[1] >= b.y0 && p[1] <= b.y1;

/** Returns { ok, checks: [{id, label, ok, hint}] } for one tracked frame. */
export function evaluateFraming(f, cfg = DEFAULT_FRAMING) {
  const b = cfg.box;
  const P = f && f.pose;
  const sh = P && [P[11], P[12]].every((p) => (p[3] == null || p[3] > 0.5) && inBox(p, b))
    && P[11][1] >= cfg.shoulderY[0] && P[11][1] <= cfg.shoulderY[1];
  const sw = P ? Math.abs(P[11][0] - P[12][0]) : 0;
  const dist = !!P && sw >= cfg.shoulderWidth[0] && sw <= cfg.shoulderWidth[1];
  const handOk = (h) => !!h && h.every((p) => inBox(p, b));
  const L = handOk(f && f.left_hand), R = handOk(f && f.right_hand);
  const checks = [
    { id: 'shoulders', label: 'Both shoulders inside the box', ok: !!sh,
      hint: !P ? 'We cannot see you yet.' : 'Move so both shoulders are inside the box, with your head below the top edge.' },
    { id: 'distance', label: 'Right distance from the camera', ok: dist,
      hint: !P ? '' : sw < cfg.shoulderWidth[0] ? 'Move a little closer to the camera.' : 'Move a little further from the camera.' },
  ];
  if (cfg.requireHands) {
    checks.push({ id: 'left', label: 'Your left hand inside the box', ok: L, hint: 'Raise both hands in front of your chest, inside the box.' });
    checks.push({ id: 'right', label: 'Your right hand inside the box', ok: R, hint: 'Raise both hands in front of your chest, inside the box.' });
  }
  return { ok: checks.every((c) => c.ok), checks };
}

/** Draws the centre box on a (non-mirrored) overlay canvas. */
export function drawBox(ctx, w, h, ok, cfg = DEFAULT_FRAMING) {
  const b = cfg.box;
  const x = b.x0 * w, y = b.y0 * h, bw = (b.x1 - b.x0) * w, bh = (b.y1 - b.y0) * h;
  const lw = Math.max(5, w / 120);
  ctx.save();
  if (ok) { ctx.fillStyle = 'rgba(34,197,94,0.12)'; ctx.fillRect(x, y, bw, bh); }
  ctx.lineWidth = lw;
  ctx.setLineDash(ok ? [] : [lw * 3, lw * 2]);   // solid green when OK, dashed white otherwise
  ctx.strokeStyle = ok ? '#22c55e' : 'rgba(255,255,255,0.95)';
  ctx.shadowColor = 'rgba(0,0,0,0.7)'; ctx.shadowBlur = 6;
  ctx.strokeRect(x, y, bw, bh);
  ctx.restore();
}
