// results.js — end-of-experiment summary: per-item similarity scores and
// side-by-side / overlaid MediaPipe movement playback (reference vs participant).

import { normalizeSequence } from './similarity.js';
import { SkeletonPlayer, imageMap, colors } from './skeleton.js';

const mirrorFrames = (frames) => frames.map((f) => {
  const m = (a) => a && a.map((p) => [-p[0], p[1], p[2], p[3]]);
  return { t: f.t, pose: m(f.pose), left_hand: m(f.right_hand), right_hand: m(f.left_hand) };
});

export function renderResults(container, session, refs, media) {
  const trials = session.trials;
  const scores = trials.map((t) => t.similarity && t.similarity.score).filter((s) => s != null);
  const mean = scores.length ? scores.reduce((a, b) => a + b, 0) / scores.length : null;
  const c = colors();
  container.innerHTML = `
    <h2>Your results</h2>
    <p class="muted">Similarity compares the movement of your body and hands with the model signer (0–100, higher is more similar).
      Average: <strong>${mean == null ? '–' : mean.toFixed(0)}</strong></p>
    <table class="results"><thead><tr><th>#</th><th>Sign</th><th>Similarity</th><th class="hide-narrow">Hand used</th><th class="hide-narrow">Views</th><th class="hide-narrow">Attempts</th><th></th></tr></thead><tbody></tbody></table>
    <p class="muted legend"><span style="color:${c.ref}">■</span> model signer &nbsp; <span style="color:${c.you}">■</span> you</p>`;
  const tbody = container.querySelector('tbody');
  let openRow = null;

  trials.forEach((t, i) => {
    const s = t.similarity || {};
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${i + 1}</td><td>${t.item.id}</td>
      <td><div class="bar"><span style="width:${s.score ?? 0}%"></span></div><span class="num">${s.score == null ? '–' : s.score.toFixed(0)}</span></td>
      <td class="hide-narrow">${s.score == null ? '–' : s.mirrored ? 'mirrored' : 'same'}</td>
      <td class="hide-narrow">${t.stimulus.totalPlays}</td><td class="hide-narrow">${t.recording.attempts}</td>
      <td><button class="btn small">Show</button></td>`;
    tbody.appendChild(tr);
    const btn = tr.querySelector('button');
    btn.onclick = () => {
      if (openRow) { openRow.cleanup(); openRow.row.remove(); const was = openRow.i; openRow = null; if (was === i) { btn.textContent = 'Show'; return; } }
      tbody.querySelectorAll('button').forEach((b) => (b.textContent = 'Show'));
      btn.textContent = 'Hide';
      const row = document.createElement('tr');
      row.className = 'detail';
      row.innerHTML = `<td colspan="7"><div class="compare">
          <figure><figcaption>Model</figcaption><div class="media"><video muted playsinline loop src="${media[t.item.id]}"></video></div></figure>
          <figure><figcaption>Movement overlay</figcaption><div class="media sk"><canvas></canvas></div></figure>
          <figure><figcaption>You (as captured)</figcaption><div class="media sk"><canvas></canvas></div></figure>
        </div>
        <div class="row">
          <button class="btn small b-play">Pause</button>
          <label><input type="checkbox" class="b-align" ${s._path ? '' : 'disabled'}> time-aligned (DTW)</label>
          <span class="muted">${s.score == null ? (s.error || '') : `wrist ${s.components.wrist.toFixed(2)} · elbow ${s.components.elbow.toFixed(2)} · hand shape ${s.components.hand.toFixed(2)} (distances)`}</span>
        </div></td>`;
      tr.after(row);
      const ref = refs.items[t.item.id];
      const pFrames = t.recording.frames;
      const pAspect = t.recording.width / t.recording.height;
      const nRef = normalizeSequence(ref.frames, ref.width / ref.height);
      const nPart = normalizeSequence(pFrames, pAspect);
      const [cvO, cvY] = row.querySelectorAll('canvas');
      const vid = row.querySelector('video');
      vid.play().catch(() => {});
      const pO = new SkeletonPlayer(cvO), pY = new SkeletonPlayer(cvY);
      const partNorm = nPart ? (s.mirrored ? mirrorFrames(nPart.frames) : nPart.frames) : null;
      const setO = (aligned) => pO.set(
        [nRef && { frames: nRef.frames, color: c.ref, alpha: 0.9 }, partNorm && { frames: partNorm, color: c.you, alpha: 0.9 }],
        aligned && s._path ? { alignment: s._path, times: { a: s._refTimes, b: s._partTimes } } : {});
      setO(false);
      pY.set([{ frames: pFrames, color: c.you, map: (w, h) => imageMap(w, h, true) }]);
      cvY.parentElement.style.aspectRatio = `${t.recording.width} / ${t.recording.height}`;
      pO.play(); pY.play();
      const bp = row.querySelector('.b-play');
      bp.onclick = () => {
        if (pO.playing) { pO.pause(); pY.pause(); vid.pause(); bp.textContent = 'Play'; }
        else { pO.play(); pY.play(); vid.play(); bp.textContent = 'Pause'; }
      };
      row.querySelector('.b-align').onchange = (e) => { const was = pO.playing; pO.pause(); setO(e.target.checked); if (was) pO.play(); };
      openRow = { i, row, cleanup: () => { pO.destroy(); pY.destroy(); vid.pause(); } };
    };
  });
}
