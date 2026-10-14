// Recompute similarity scores offline (e.g. after changing tau or weights), CSV to stdout.
//   node tools/rescore.mjs data/references.json results/*.json [--tau 0.6] [--no-mirror]
import fs from 'node:fs';
import { compareSequences } from '../js/similarity.js';

const args = process.argv.slice(2);
const opts = {};
const files = [];
for (let i = 0; i < args.length; i++) {
  if (args[i] === '--tau') opts.tau = parseFloat(args[++i]);
  else if (args[i] === '--no-mirror') opts.allowMirror = false;
  else files.push(args[i]);
}
const refs = JSON.parse(fs.readFileSync(files.shift(), 'utf8'));
console.log('file,participant,trial,item,telicity,score,distance,mirrored,wrist,elbow,hand,direct_score,mirror_score');
for (const f of files) {
  const s = JSON.parse(fs.readFileSync(f, 'utf8'));
  for (const t of s.trials) {
    const ref = refs.items[t.item.id];
    const r = t.recording;
    const sim = compareSequences(ref.frames, ref.width / ref.height, r.frames, r.width / r.height, opts);
    const c = sim.components || {};
    console.log([f, s.participantId, t.index + 1, t.item.id, t.item.telicity, sim.score?.toFixed(2), sim.distance?.toFixed(4), sim.mirrored,
      c.wrist?.toFixed(4), c.elbow?.toFixed(4), c.hand?.toFixed(4), sim.direct?.score?.toFixed(2), sim.mirror?.score?.toFixed(2)].join(','));
  }
}
