// Sanity/calibration check for js/similarity.js using the asl_telicity Holistic pickles
// exported to JSON ({ "asl_pkl/<name>": [frames], "asl_3nd_pkl/<name>": [frames] }).
//   node test/similarity_test.mjs path/to/pkl_seqs.json
// Same-sign pairs (same name in both folders) should be closer than different-sign pairs.
import fs from 'node:fs';
import { compareSequences, DEFAULT_SIM } from '../js/similarity.js';

const data = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const A = {}, B = {};
for (const [k, v] of Object.entries(data)) {
  const [dir, name] = k.split('/');
  const frames = v.map((f, i) => ({ ...f, t: i * (1000 / 30) }));
  (dir === 'asl_pkl' ? A : B)[name] = frames;
}
const common = Object.keys(A).filter((n) => B[n]);
console.log('common signs:', common.length);

const same = [], diff = [];
let mirrorUsed = 0;
for (const n of common) {
  const r = compareSequences(A[n], 1, B[n], 1);
  if (r.distance != null) { same.push(r.distance); if (r.mirrored) mirrorUsed++; }
  for (let k = 0; k < 4; k++) {
    const m = common[(common.indexOf(n) + 7 + k * 11) % common.length];
    if (m === n) continue;
    const q = compareSequences(A[n], 1, B[m], 1);
    if (q.distance != null) diff.push(q.distance);
  }
}
const med = (a) => [...a].sort((x, y) => x - y)[a.length >> 1];
let auc = 0;
for (const s of same) for (const d of diff) auc += s < d ? 1 : s === d ? 0.5 : 0;
auc /= same.length * diff.length;
const tau = DEFAULT_SIM.tau;
console.log(`same-sign  n=${same.length} median dist=${med(same).toFixed(3)} -> score ${(100 * Math.exp(-med(same) / tau)).toFixed(1)}`);
console.log(`diff-sign  n=${diff.length} median dist=${med(diff).toFixed(3)} -> score ${(100 * Math.exp(-med(diff) / tau)).toFixed(1)}`);
console.log(`AUC (same closer than different) = ${auc.toFixed(3)};  mirrored chosen for same-sign: ${mirrorUsed}/${same.length}`);
// Simulated imitation: time-warp (0.75x / 1.35x), different scale/position, landmark jitter,
// plus ~1 s idle frames (hands down) before/after.
let seed = 1; const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647 - 0.5);
function perturb(seq, speed, s, dx, dy, noise) {
  const out = [];
  const N = Math.round(seq.length / speed);
  const idle = (f) => {
    const g = JSON.parse(JSON.stringify(f));
    if (g.pose) { g.pose[15][1] = g.pose[16][1] = 1.2; } g.left_hand = g.right_hand = null; return g;
  };
  const tf = (p) => p && p.map((q) => [ (q[0] - 0.5) * s + 0.5 + dx + noise * rnd(), (q[1] - 0.5) * s + 0.5 + dy + noise * rnd(), q[2], q[3] ].slice(0, q.length));
  const push = (f) => out.push({ t: out.length * 33.3, pose: tf(f.pose), left_hand: tf(f.left_hand), right_hand: tf(f.right_hand) });
  for (let k = 0; k < 30; k++) push(idle(seq[0]));
  for (let k = 0; k < N; k++) push(seq[Math.min(seq.length - 1, Math.floor(k * speed))]);
  for (let k = 0; k < 30; k++) push(idle(seq[seq.length - 1]));
  return out;
}
const pert = [], pertDiff = [];
common.forEach((n, i) => {
  const sp = i % 2 ? 0.75 : 1.35;
  const p = perturb(B[n], sp, 0.8 + 0.4 * (i % 3) / 2, 0.05 * ((i % 5) - 2), 0.03, 0.01);
  const r = compareSequences(A[n], 1, p, 1); if (r.distance != null) pert.push(r.distance);
  const m = common[(i + 13) % common.length];
  const q = compareSequences(A[m], 1, p, 1); if (q.distance != null) pertDiff.push(q.distance);
});
let auc2 = 0; for (const s of pert) for (const d of pertDiff) auc2 += s < d ? 1 : s === d ? 0.5 : 0;
auc2 /= pert.length * pertDiff.length;
console.log(`simulated imitation, same sign: median dist=${med(pert).toFixed(3)} -> score ${(100 * Math.exp(-med(pert) / tau)).toFixed(1)}`);
console.log(`simulated imitation vs other sign: median dist=${med(pertDiff).toFixed(3)} -> score ${(100 * Math.exp(-med(pertDiff) / tau)).toFixed(1)};  AUC=${auc2.toFixed(3)}`);

const self = compareSequences(A[common[0]], 1, A[common[0]], 1);
console.log('self-comparison score (should be 100):', self.score.toFixed(1));
