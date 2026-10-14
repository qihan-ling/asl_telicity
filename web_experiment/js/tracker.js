// tracker.js — MediaPipe Tasks Vision wrapper (PoseLandmarker + HandLandmarker).
// processFrame(source, timestampMs) returns one frame in the asl_telicity Holistic-style format:
//   { pose: [[x,y,z,vis] x33] | null, left_hand: [[x,y,z] x21] | null, right_hand: ... }
// "left"/"right" are the signer's own hands. Hands are matched to the pose wrists
// (landmarks 15 = left wrist, 16 = right wrist), as Holistic does.
// The source is always the *unmirrored* image (camera frames are only mirrored on screen).

const r4 = (v) => (v == null ? null : Math.round(v * 10000) / 10000);

export async function createTracker(mp, { mock = false } = {}) {
  if (mock) return createMockTracker();

  const cdn = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${mp.version}`;
  const base = mp.source === 'local' ? new URL('../vendor/mediapipe', import.meta.url).href : cdn;
  const models = mp.source === 'local'
    ? { pose: `${base}/models/pose_landmarker_${mp.poseModel}.task`, hand: `${base}/models/hand_landmarker.task` }
    : {
        pose: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_${mp.poseModel}/float16/1/pose_landmarker_${mp.poseModel}.task`,
        hand: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
      };

  const vision = await import(`${base}/vision_bundle.mjs`);
  const { FilesetResolver, PoseLandmarker, HandLandmarker } = vision;
  const fileset = await FilesetResolver.forVisionTasks(`${base}/wasm`);

  async function build(delegate) {
    const pose = await PoseLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: models.pose, delegate },
      runningMode: 'VIDEO', numPoses: 1,
      minPoseDetectionConfidence: 0.5, minPosePresenceConfidence: 0.5, minTrackingConfidence: 0.5,
    });
    const hands = await HandLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: models.hand, delegate },
      runningMode: 'VIDEO', numHands: 2,
      minHandDetectionConfidence: 0.4, minHandPresenceConfidence: 0.4, minTrackingConfidence: 0.4,
    });
    return { pose, hands };
  }
  let delegate = mp.delegate || 'GPU', lm;
  try { lm = await build(delegate); } catch (e) { console.warn('GPU delegate failed, using CPU', e); delegate = 'CPU'; lm = await build('CPU'); }

  const crop = document.createElement('canvas');
  crop.width = crop.height = mp.handCropSize || 512;
  const cctx = crop.getContext('2d', { willReadFrequently: false });
  let lastTs = -1;

  function processFrame(source, tsMs) {
    const ts = Math.max(Math.round(tsMs), lastTs + 1); // MediaPipe needs strictly increasing timestamps
    lastTs = ts;
    const W = source.videoWidth || source.width, H = source.videoHeight || source.height;
    const pr = lm.pose.detectForVideo(source, ts);
    const P = pr.landmarks && pr.landmarks[0] ? pr.landmarks[0] : null;

    // optional crop around the body for hand detection
    let handSrc = source, box = null;
    if (mp.handCrop && P) {
      const ls = P[11], rs = P[12];
      const sw = Math.hypot((ls.x - rs.x) * W, (ls.y - rs.y) * H);
      const side = Math.max(64, sw * (mp.handCropScale || 4));
      if (side < 0.9 * Math.min(W, H)) {
        const cx = ((ls.x + rs.x) / 2) * W, cy = ((ls.y + rs.y) / 2) * H + 0.4 * sw;
        box = { x0: cx - side / 2, y0: cy - side / 2, side };
        const S = crop.width, k = S / side;
        const sx = Math.max(0, box.x0), sy = Math.max(0, box.y0);
        const ex = Math.min(W, box.x0 + side), ey = Math.min(H, box.y0 + side);
        cctx.fillStyle = '#000'; cctx.fillRect(0, 0, S, S);
        if (ex > sx && ey > sy) cctx.drawImage(source, sx, sy, ex - sx, ey - sy, (sx - box.x0) * k, (sy - box.y0) * k, (ex - sx) * k, (ey - sy) * k);
        handSrc = crop;
      }
    }
    const hr = lm.hands.detectForVideo(handSrc, ts);
    const toFull = (p) => box
      ? { x: (box.x0 + p.x * box.side) / W, y: (box.y0 + p.y * box.side) / H, z: (p.z * box.side) / W }
      : p;
    const handList = (hr.landmarks || []).map((h) => h.map(toFull));
    const labels = hr.handedness || hr.handednesses || [];
    const { left, right } = assignHands(handList, labels, P, W, H);

    return {
      pose: P ? P.map((p) => [r4(p.x), r4(p.y), r4(p.z), r4(p.visibility ?? null)]) : null,
      left_hand: left ? left.map((p) => [r4(p.x), r4(p.y), r4(p.z)]) : null,
      right_hand: right ? right.map((p) => [r4(p.x), r4(p.y), r4(p.z)]) : null,
    };
  }

  return {
    processFrame,
    info: { library: '@mediapipe/tasks-vision', version: mp.version, source: mp.source, models, delegate, handCrop: !!mp.handCrop, mock: false },
    close() { lm.pose.close(); lm.hands.close(); },
  };
}

export function assignHands(hands, labels, P, W, H) {
  const out = { left: null, right: null };
  if (!hands.length) return out;
  if (P) {
    const L = P[15], R = P[16];
    const d = (h, p) => Math.hypot((h[0].x - p.x) * W, (h[0].y - p.y) * H);
    if (hands.length === 1) {
      if (d(hands[0], L) <= d(hands[0], R)) out.left = hands[0]; else out.right = hands[0];
    } else {
      const [a, b] = hands;
      if (d(a, L) + d(b, R) <= d(a, R) + d(b, L)) { out.left = a; out.right = b; } else { out.left = b; out.right = a; }
    }
  } else {
    // MediaPipe handedness assumes a mirrored (selfie) image; our input is unmirrored,
    // so the label "Left" is the signer's right hand.
    hands.forEach((h, i) => {
      const lab = labels[i] && labels[i][0] ? labels[i][0].categoryName : null;
      if (lab === 'Left' && !out.right) out.right = h; else if (!out.left) out.left = h; else if (!out.right) out.right = h;
    });
  }
  return out;
}

// ---------------------------------------------------------------------------
// Mock tracker: synthetic upper-body skeleton, for testing the experiment flow
// without a camera model (?mock=1). The motion depends on time only.
function createMockTracker() {
  function frameAt(sec, seed = 0) {
    const pose = [];
    for (let i = 0; i < 33; i++) pose.push([0.5, 0.9, 0, 0.2]);
    const set = (i, x, y, v = 0.99) => (pose[i] = [x, y, 0, v]);
    set(0, 0.5, 0.25); for (let i = 1; i <= 10; i++) set(i, 0.5 + ((i % 2) ? -1 : 1) * 0.015 * Math.ceil(i / 2), 0.24 + (i > 8 ? 0.06 : 0));
    set(11, 0.42, 0.42); set(12, 0.58, 0.42); set(23, 0.44, 0.8); set(24, 0.56, 0.8);
    const ph = sec * (2.2 + seed * 0.3);
    const act = Math.max(0, Math.min(1, Math.sin(Math.PI * Math.min(1, sec / 3)) * 1.6));
    const rw = [0.47 + 0.05 * Math.cos(ph), 0.95 - act * (0.4 + 0.06 * Math.sin(ph))];
    const lw = [0.62, 0.95 - act * 0.25];
    set(16, rw[0], rw[1]); set(15, lw[0], lw[1]);
    set(14, (0.58 + rw[0]) / 2 + 0.04, (0.42 + rw[1]) / 2 + 0.05); set(13, (0.42 + lw[0]) / 2 - 0.02, (0.42 + lw[1]) / 2 + 0.05);
    const hand = (w, curl) => {
      const h = [[w[0], w[1], 0]];
      for (let f = 0; f < 5; f++) for (let j = 1; j <= 4; j++) {
        const ang = -Math.PI / 2 + (f - 2) * 0.35, len = 0.012 * j * (f === 0 ? 0.8 : 1) * (1 - curl * (j > 1 ? 0.5 : 0));
        h.push([w[0] + Math.cos(ang) * len, w[1] + Math.sin(ang) * len, 0]);
      }
      return h;
    };
    return {
      pose: pose.map((p) => p.map(r4)),
      right_hand: act > 0.2 ? hand(rw, 0.5 + 0.5 * Math.sin(ph)).map((p) => p.map(r4)) : null,
      left_hand: act > 0.5 ? hand(lw, 0.2).map((p) => p.map(r4)) : null,
    };
  }
  let t0 = null;
  return {
    processFrame(source, tsMs) {
      if (source && source.tagName === 'VIDEO' && !source.srcObject) return frameAt(source.currentTime, 0); // stimulus video
      if (t0 === null) t0 = tsMs;
      return frameAt(((tsMs - t0) / 1000) % 4, 0.3);
    },
    resetClock() { t0 = null; },
    info: { library: 'mock', version: '0', mock: true },
    close() {},
  };
}
