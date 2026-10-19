// results.js — end-of-experiment summary: per-item similarity scores, grouped into
// telic and atelic sets, with overlaid movement playback (model vs participant),
// and a "Want to know more?" page explaining the telic/atelic difference.

import { normalizeSequence } from './similarity.js';
import { SkeletonPlayer, imageMap, colors } from './skeleton.js';

// ---- Participant-facing text (edit freely) ----
const SETS = [
  { key: 'telic', title: 'Set 1' },
  { key: 'atelic', title: 'Set 2' },
];
const MORE = {
  title: 'Telic and atelic signs',
  telic: (list) => `The <b>telic</b> signs you saw (${list}) involve a distinctive change in the hands, in their
    <b>shape</b>, <b>orientation</b> or <b>position</b>, from the beginning to the end of the sign.`,
  atelic: (list) => `The <b>atelic</b> signs you saw (${list}) don't have such a change: the hands do not end up in a
    clearly different shape, orientation or position from where they started.`,
  extra: `In linguistics, <i>telic</i> verbs describe events with a built-in endpoint (such as <i>die</i> or <i>decide</i>),
    while <i>atelic</i> verbs describe activities without one (such as <i>run</i> or <i>talk</i>).`,
  tryIt: 'Open the movement playback for a telic sign and compare the first and last moments of the model\'s hands.',
};

const mirrorFrames = (frames) => frames.map((f) => {
  const m = (a) => a && a.map((p) => [-p[0], p[1], p[2], p[3]]);
  return { t: f.t, pose: m(f.pose), left_hand: m(f.right_hand), right_hand: m(f.left_hand) };
});
const avg = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null);
const fmt = (x) => (x == null ? '–' : x.toFixed(0));

export function renderResults(container, session, refs, media) {
  const trials = session.trials;
  const c = colors();
  const scoreOf = (t) => (t.similarity ? t.similarity.score : null);
  const overall = avg(trials.map(scoreOf).filter((s) => s != null));
  const groups = SETS.map((g) => ({ ...g, trials: trials.filter((t) => t.item.telicity === g.key) })).filter((g) => g.trials.length);

  container.innerHTML = `
    <div class="res-main">
      <h2>Your results</h2>
      <p class="muted">Similarity compares the movement of your body and hands with the model signer
        (0–100, higher is more similar). Average: <strong>${fmt(overall)}</strong></p>
      <p class="legend">In the movement overlays: <span class="key" style="background:${c.ref}"></span> model signer
        &nbsp; <span class="key" style="background:${c.you}"></span> you</p>
      ${groups.map((g, gi) => `
        <section class="set">
          <h3>${g.title} <span class="muted small">· average ${fmt(avg(g.trials.map(scoreOf).filter((s) => s != null)))}</span></h3>
          <table class="results" data-g="${gi}"><thead><tr><th>Sign</th><th>Similarity</th><th class="hide-narrow">Hand used</th>
            <th class="hide-narrow">Views</th><th class="hide-narrow">Attempts</th><th></th></tr></thead><tbody></tbody></table>
        </section>`).join('')}
      <div class="row"><button class="btn" id="more">Want to know more?</button></div>
    </div>
    <div class="res-more" hidden>
      <h2>${MORE.title}</h2>
      <p>${groups.map((g) => `The signs in ${g.title} are <b>${g.key}</b>.`).join(' ')}</p>
      ${groups.map((g) => `<p>${MORE[g.key](g.trials.map((t) => `<i>${t.item.id}</i>`).join(', '))}</p>`).join('')}
      <p>${MORE.extra}</p>
      <p class="muted">${MORE.tryIt}</p>
      <div class="row"><button class="btn primary" id="back">Back to my results</button></div>
    </div>`;

  const main = container.querySelector('.res-main'), more = container.querySelector('.res-more');
  let openRow = null;
  const closeOpen = () => { if (openRow) { openRow.cleanup(); openRow.row.remove(); openRow.btn.textContent = 'Show'; openRow = null; } };
  container.querySelector('#more').onclick = () => { closeOpen(); main.hidden = true; more.hidden = false; container.scrollIntoView({ behavior: 'smooth' }); };
  container.querySelector('#back').onclick = () => { more.hidden = true; main.hidden = false; container.scrollIntoView({ behavior: 'smooth' }); };

  groups.forEach((g, gi) => {
    const tbody = container.querySelector(`table[data-g="${gi}"] tbody`);
    g.trials.forEach((t) => {
      const s = t.similarity || {};
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${t.item.id}</td>
        <td><div class="bar"><span style="width:${s.score ?? 0}%"></span></div><span class="num">${fmt(s.score)}</span></td>
        <td class="hide-narrow">${s.score == null ? '–' : s.mirrored ? 'mirrored' : 'same'}</td>
        <td class="hide-narrow">${t.stimulus.totalPlays}</td><td class="hide-narrow">${t.recording.attempts}</td>
        <td><button class="btn small">Show</button></td>`;
      tbody.appendChild(tr);
      const btn = tr.querySelector('button');
      btn.onclick = () => {
        const wasThis = openRow && openRow.tr === tr;
        closeOpen();
        if (wasThis) return;
        btn.textContent = 'Hide';
        openRow = openDetail(tr, t, s, refs, media, c);
        openRow.btn = btn;
      };
    });
  });
}

function openDetail(tr, t, s, refs, media, c) {
  const row = document.createElement('tr');
  row.className = 'detail';
  row.innerHTML = `<td colspan="6"><div class="compare">
      <figure><figcaption>Model</figcaption><div class="media"><video muted playsinline loop src="${media[t.item.id]}"></video></div></figure>
      <figure><figcaption>Movement overlay</figcaption><div class="media sk"><canvas></canvas></div></figure>
    </div>
    <div class="row">
      <button class="btn small b-play">Pause</button>
      <label><input type="checkbox" class="b-align" ${s._path ? 'checked' : 'disabled'}> time-aligned</label>
    </div></td>`;
  tr.after(row);
  const ref = refs.items[t.item.id];
  const pFrames = t.recording.frames;
  const nRef = normalizeSequence(ref.frames, ref.width / ref.height);
  const nPart = normalizeSequence(pFrames, t.recording.width / t.recording.height);
  const [cvO] = row.querySelectorAll('canvas');
  const vid = row.querySelector('video');
  vid.play().catch(() => {});
  const pO = new SkeletonPlayer(cvO);
  const partNorm = nPart ? (s.mirrored ? mirrorFrames(nPart.frames) : nPart.frames) : null;
  const setO = (aligned) => pO.set(
    [nRef && { frames: nRef.frames, color: c.ref, alpha: 0.9 }, partNorm && { frames: partNorm, color: c.you, alpha: 0.9 }],
    aligned && s._path ? { alignment: s._path, times: { a: s._refTimes, b: s._partTimes } } : {});
  setO(!!s._path); // default: time-aligned, i.e. exactly the (trimmed) segments that were compared
  pO.play();
  const bp = row.querySelector('.b-play');
  bp.onclick = () => {
    if (pO.playing) { pO.pause(); vid.pause(); bp.textContent = 'Play'; }
    else { pO.play(); vid.play(); bp.textContent = 'Pause'; }
  };
  row.querySelector('.b-align').onchange = (e) => { const was = pO.playing; pO.pause(); setO(e.target.checked); if (was) pO.play(); };
  return { tr, row, cleanup: () => { pO.destroy(); vid.pause(); } };
}
