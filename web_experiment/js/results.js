// results.js — end-of-experiment page: for each critical sign, the model video next to an
// overlay of the model's and the participant's movements, grouped into telic and atelic sets
// (unnamed until "Want to know more?"). No scores are shown (they are still saved in the data).
// "Want to know more?" explains the telic/atelic difference and the meaning manipulation.

import { normalizeSequence } from './similarity.js';
import { SkeletonPlayer, colors } from './skeleton.js';

// ---- Participant-facing text (edit freely) ----
const SETS = [
  { key: 'telic', title: 'Set 1' },
  { key: 'atelic', title: 'Set 2' },
];
const TEXT = {
  intro: 'Here you can compare your movements with the video.',
  moreTitle: 'Telic and atelic signs',
  telic: (list) => `The <b>telic</b> signs you saw (${list}) involve a distinctive change in the hands, in their
    <b>shape</b>, <b>orientation</b> or <b>position</b>, from the beginning to the end of the sign.`,
  atelic: (list) => `The <b>atelic</b> signs you saw (${list}) don't have such a change: the hands do not end up in a
    clearly different shape, orientation or position from where they started.`,
  extra: `In linguistics, <i>telic</i> verbs describe events with a built-in endpoint (such as <i>die</i> or <i>decide</i>),
    while <i>atelic</i> verbs describe activities without one (such as <i>run</i> or <i>talk</i>).`,
  meaningTitle: 'About the meanings we showed you',
  meaningDesign: `Before each sign, we told you what it means. Not everyone was told the truth: each participant is
    randomly placed in one of three groups. One group sees each sign's <b>real meaning</b>. One group sees a
    <b>wrong meaning of the same type</b> (a telic meaning for a telic sign, an atelic meaning for an atelic sign).
    The third group sees a <b>wrong meaning of the opposite type</b> (an atelic meaning for a telic sign, and the other way round).
    Comparing the groups lets us study how the meaning people have in mind relates to how they produce a sign.`,
  yourGroup: {
    correct: 'You were in the group that saw the <b>real meanings</b>.',
    same: 'You were in the group that saw <b>wrong meanings of the same type</b>.',
    opposite: 'You were in the group that saw <b>wrong meanings of the opposite type</b>.',
  },
  meaningNote: 'The signs in this study come from Sign Language of the Netherlands (NGT).',
};

const mirrorFrames = (frames) => frames.map((f) => {
  const m = (a) => a && a.map((p) => [-p[0], p[1], p[2], p[3]]);
  return { t: f.t, pose: m(f.pose), left_hand: m(f.right_hand), right_hand: m(f.left_hand) };
});

export function renderResults(container, session, refs, media) {
  const crit = session.trials.filter((t) => t.block !== 'practice' && !t.item.practice);
  const c = colors();
  const label = (t) => `Sign ${crit.indexOf(t) + 1}`;
  const groups = SETS.map((g) => ({ ...g, trials: crit.filter((t) => t.item.telicity === g.key) })).filter((g) => g.trials.length);
  const cond = session.meaningCondition;
  const withMeaning = crit.filter((t) => t.meaning);

  container.innerHTML = `
    <div class="res-main">
      <h2>Your movements</h2>
      <p class="muted">${TEXT.intro}</p>
      <p class="legend"><span class="key" style="background:${c.ref}"></span> model signer
        &nbsp; <span class="key" style="background:${c.you}"></span> you</p>
      ${groups.map((g, gi) => `
        <section class="set">
          <h3>${g.title}</h3>
          ${g.trials.map((t) => `<div class="item" data-i="${crit.indexOf(t)}"><h4>${label(t)}</h4></div>`).join('')}
        </section>`).join('')}
      <div class="row"><button class="btn" id="more">Want to know more?</button></div>
    </div>
    <div class="res-more" hidden>
      <h2>${TEXT.moreTitle}</h2>
      <p>${groups.map((g) => `The signs in ${g.title} are <b>${g.key}</b>.`).join(' ')}</p>
      ${groups.map((g) => `<p>${TEXT[g.key](g.trials.map(label).join(', '))}</p>`).join('')}
      <p>${TEXT.extra}</p>
      ${withMeaning.length ? `
        <h3>${TEXT.meaningTitle}</h3>
        <p>${TEXT.meaningDesign}</p>
        ${cond && TEXT.yourGroup[cond] ? `<p>${TEXT.yourGroup[cond]}</p>` : ''}
        <table class="results meanings"><thead><tr><th></th><th>We told you it means</th><th>It actually means</th><th>Type</th></tr></thead>
          <tbody>${withMeaning.map((t) => `<tr><td>${label(t)}</td><td>${t.meaning.shown}</td><td><b>${t.meaning.correct}</b></td><td>${t.item.telicity}</td></tr>`).join('')}</tbody></table>
        <p class="muted small">${TEXT.meaningNote}</p>` : ''}
      <div class="row"><button class="btn primary" id="back">Back</button></div>
    </div>`;

  const main = container.querySelector('.res-main'), more = container.querySelector('.res-more');
  const players = [];
  container.querySelectorAll('.item').forEach((el) => players.push(renderItem(el, crit[+el.dataset.i], refs, media, c)));
  container.querySelector('#more').onclick = () => { players.forEach((p) => p.pause()); main.hidden = true; more.hidden = false; container.scrollIntoView({ behavior: 'smooth' }); };
  container.querySelector('#back').onclick = () => { more.hidden = true; main.hidden = false; players.forEach((p) => p.resume()); container.scrollIntoView({ behavior: 'smooth' }); };
}

function renderItem(el, t, refs, media, c) {
  const s = t.similarity || {};
  el.insertAdjacentHTML('beforeend', `<div class="compare">
      <figure><figcaption>Model</figcaption><div class="media"><video muted playsinline loop src="${media[t.item.id]}"></video></div></figure>
      <figure><figcaption>Movement overlay</figcaption><div class="media sk"><canvas></canvas></div></figure>
    </div>
    <div class="row">
      <button class="btn small b-play">Pause</button>
      <label><input type="checkbox" class="b-align" ${s._path ? 'checked' : 'disabled'}> time-aligned</label>
    </div>`);
  const ref = refs.items[t.item.id];
  const nRef = normalizeSequence(ref.frames, ref.width / ref.height);
  const nPart = normalizeSequence(t.recording.frames, t.recording.width / t.recording.height);
  const vid = el.querySelector('video');
  vid.play().catch(() => {});
  const pO = new SkeletonPlayer(el.querySelector('canvas'));
  const partNorm = nPart ? (s.mirrored ? mirrorFrames(nPart.frames) : nPart.frames) : null;
  const setO = (aligned) => pO.set(
    [nRef && { frames: nRef.frames, color: c.ref, alpha: 0.9 }, partNorm && { frames: partNorm, color: c.you, alpha: 0.9 }],
    aligned && s._path ? { alignment: s._path, times: { a: s._refTimes, b: s._partTimes } } : {});
  setO(!!s._path); // default: time-aligned, i.e. exactly the (trimmed) segments that were compared
  pO.play();
  const bp = el.querySelector('.b-play');
  let userPaused = false;
  bp.onclick = () => {
    if (pO.playing) { pO.pause(); vid.pause(); bp.textContent = 'Play'; userPaused = true; }
    else { pO.play(); vid.play(); bp.textContent = 'Pause'; userPaused = false; }
  };
  el.querySelector('.b-align').onchange = (e) => { const was = pO.playing; pO.pause(); setO(e.target.checked); if (was) pO.play(); };
  return {
    pause() { pO.pause(); vid.pause(); },
    resume() { if (!userPaused) { pO.play(); vid.play().catch(() => {}); } },
  };
}
