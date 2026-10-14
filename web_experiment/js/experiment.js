// experiment.js — flow: start -> camera check -> trials (watch, sign, [question]) -> results + data.

import { CONFIG, applyUrlOverrides } from './config.js';
import { orderItems } from './stimuli.js';
import { createTracker } from './tracker.js';
import { Camera, frameLoop } from './camera.js';
import { StimulusPlayer } from './stimulusPlayer.js';
import { Recorder } from './recorder.js';
import { loadReferences, computeReference } from './references.js';
import { compareSequences } from './similarity.js';
import { newSession, backup, loadBackup, clearBackup, download } from './data.js';
import { renderResults } from './results.js';
import { drawFrame, imageMap, colors } from './skeleton.js';

const cfg = applyUrlOverrides(CONFIG);
const app = document.getElementById('app');
const screen = (html) => { app.innerHTML = html; return app; };
const waitClick = (el) => new Promise((r) => el.addEventListener('click', r, { once: true }));
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

async function start() {
  const old = await loadBackup();
  screen(`
    <h1>Repeat the signs you see!</h1>
    <p>You will watch short videos of signs. After each video, imitate the sign you just watched.</p>
    <p class="muted">Your camera is used only to track the positions of your body and hands on this device.
      No video or images are recorded or stored.</p>
    <label class="field">Participant ID <input id="pid" autocomplete="off" value="${esc(cfg.pid)}"></label>
    <div class="row"><button class="btn primary" id="go">Start</button></div>
    <p class="status" id="msg"></p>
    ${old && !old.completed ? `<p class="muted small">An unfinished session (${esc(old.participantId || 'no id')}, ${old.trials.length} trials) is stored on this device.
      <a href="#" id="dl-old">Download it</a> · <a href="#" id="rm-old">Delete it</a></p>` : ''}`);
  if (old) {
    const dl = document.getElementById('dl-old'), rm = document.getElementById('rm-old');
    if (dl) dl.onclick = (e) => { e.preventDefault(); download(old); };
    if (rm) rm.onclick = async (e) => { e.preventDefault(); await clearBackup(); e.target.parentElement.remove(); };
  }
  const go = document.getElementById('go'), msg = document.getElementById('msg');
  for (;;) {
    await waitClick(go);
    const pid = document.getElementById('pid').value.trim();
    if (!pid) { msg.textContent = 'Please enter a participant ID.'; continue; }
    go.disabled = true;
    try { await setup(pid, msg); return; }
    catch (e) { console.error(e); msg.textContent = 'Problem: ' + (e.message || e); go.disabled = false; }
  }
}

async function setup(pid, msg) {
  msg.textContent = 'Starting camera…';
  const camera = new Camera();
  await camera.start();
  msg.textContent = 'Loading the tracking model…';
  const tracker = await createTracker(cfg.mediapipe, { mock: cfg.mock });

  msg.textContent = 'Loading stimuli…';
  const items = orderItems(cfg.itemOrder, cfg.nItems);
  // keep every stimulus in memory (instant, seekable playback; no network needed during trials)
  const media = {};
  for (const it of items) {
    const r = await fetch(cfg.stimuliBase + it.file);
    if (!r.ok) throw new Error('Cannot load ' + cfg.stimuliBase + it.file);
    media[it.id] = URL.createObjectURL(await r.blob());
  }
  let refs = await loadReferences(cfg.referencesUrl);
  const missing = items.filter((it) => !(refs && refs.items && refs.items[it.id]));
  if (missing.length) {
    refs = refs || { items: {} };
    refs.computedLive = true;
    for (let k = 0; k < missing.length; k++) {
      refs.items[missing[k].id] = await computeReference(tracker, media[missing[k].id], {
        onProgress: (i, n) => (msg.textContent = `Preparing stimuli ${k + 1}/${missing.length} (${Math.round((100 * i) / n)}%)…`),
      });
    }
  }

  const session = newSession(cfg, pid, {
    tracker: tracker.info,
    camera: camera.settings,
    references: { source: refs.computedLive ? 'computed-live' : cfg.referencesUrl, createdAt: refs.createdAt || null, tracker: refs.tracker || null },
    itemOrder: items.map((i) => i.id),
  });
  await cameraCheck(camera, tracker);
  await instructions(items.length);
  await runTrials({ session, items, camera, tracker, refs, media });
  await finish(session, refs, camera, tracker, media);
}

async function cameraCheck(camera, tracker) {
  screen(`
    <h2>Camera check</h2>
    <p>Sit so that your head, shoulders, arms and hands are in view. You should see lines following your body and hands.</p>
    <div class="media cam-wrap" id="cw"><canvas class="overlay"></canvas></div>
    <p class="status" id="st">Looking for you…</p>
    <div class="row"><button class="btn primary" id="ok" disabled>Continue</button></div>`);
  const cw = document.getElementById('cw');
  cw.prepend(camera.video); cw.style.aspectRatio = `${camera.width} / ${camera.height}`;
  camera.video.play().catch(() => {});
  const ov = cw.querySelector('canvas'), ctx = ov.getContext('2d');
  const ok = document.getElementById('ok'), st = document.getElementById('st');
  let good = 0;
  const loop = frameLoop(camera.video, (now) => {
    const f = tracker.processFrame(camera.video, now);
    ov.width = camera.width; ov.height = camera.height;
    drawFrame(ctx, f, imageMap(ov.width, ov.height, true), colors().live, { lw: Math.max(2, ov.width / 250) });
    const shoulders = f.pose && [11, 12].every((i) => f.pose[i][3] == null || f.pose[i][3] > 0.5)
      && f.pose[11][1] < 0.85 && f.pose[0][1] > 0.02;
    good = shoulders ? good + 1 : 0;
    if (good > 15) { ok.disabled = false; st.textContent = 'Looks good. Raise your hands to check they are tracked, then continue.'; }
    else if (!f.pose) st.textContent = 'Looking for you…';
    else st.textContent = 'Please move so your head and both shoulders are clearly visible.';
  });
  await waitClick(ok);
  loop.stop();
}

async function instructions(n) {
  const sp = cfg.stimulusPlayer;
  const watch = sp.mode === 'fixed'
    ? `Each video will play ${sp.plays} time${sp.plays > 1 ? 's' : ''}.`
    : sp.maxReplays === Infinity ? 'You can replay each video as often as you like.' : sp.maxReplays > 0 ? `You can replay each video up to ${sp.maxReplays} time${sp.maxReplays > 1 ? 's' : ''}.` : 'Each video plays once.';
  screen(`
    <h2>Instructions</h2>
    <ol class="steps">
      <li>Watch the sign. ${watch}</li>
      <li>Press <b>Start recording</b>. After a short countdown, copy the sign as closely as you can, then press <b>Stop</b>.</li>
      <li>You will see how your signing was captured. Press <b>Start over</b> to try again, or <b>Confirm &amp; submit</b> to continue.</li>
    </ol>
    <p>There are ${n} signs.</p>
    <div class="row"><button class="btn primary" id="go">Begin</button></div>`);
  await waitClick(document.getElementById('go'));
}

async function runTrials({ session, items, camera, tracker, refs, media }) {
  for (let i = 0; i < items.length; i++) {
    const item = items[i];
    const trial = { index: i, item, startedAt: new Date().toISOString() };
    screen(`<div class="trial-head"><span>Sign ${i + 1} of ${items.length}</span><span class="phase" id="ph">Watch</span></div><div id="stage"></div>`);
    const stage = document.getElementById('stage'), ph = document.getElementById('ph');

    // 1) watch
    const player = new StimulusPlayer(cfg.stimulusPlayer).mount(stage);
    trial.stimulus = await player.run(media[item.id]);
    player.destroy();

    // 2) sign
    ph.textContent = 'Your turn';
    const rec = new Recorder({ camera, tracker, cfg }).mount(stage);
    const r = await rec.run();
    rec.destroy();
    trial.recording = { ...r, width: camera.width, height: camera.height };

    // 3) optional question
    if (cfg.postTrialQuestion) trial.response = await askQuestion(stage, ph, cfg.postTrialQuestion);

    // similarity
    const ref = refs.items[item.id];
    try {
      const sim = compareSequences(ref.frames, ref.width / ref.height, r.frames, camera.width / camera.height, cfg.similarity);
      if (cfg.includeAlignmentInData && sim._path) sim.alignment = { path: sim._path, refTimes: sim._refTimes, partTimes: sim._partTimes };
      trial.similarity = sim;
    } catch (e) { trial.similarity = { score: null, error: String(e) }; }

    trial.endedAt = new Date().toISOString();
    session.trials.push(trial);
    await backup(session);
  }
}

function askQuestion(stage, ph, q) {
  ph.textContent = 'Question';
  stage.innerHTML = `<label class="field block">${esc(q.prompt)}<textarea id="ans" rows="3"></textarea></label>
    <div class="row"><button class="btn primary" id="nx">Next</button></div>`;
  const t0 = performance.now();
  return new Promise((res) => {
    document.getElementById('nx').onclick = () => {
      const v = document.getElementById('ans').value.trim();
      if (q.required && !v) return;
      res({ type: q.type, prompt: q.prompt, answer: v, rtMs: Math.round(performance.now() - t0) });
    };
  });
}

async function finish(session, refs, camera, tracker, media) {
  camera.stop(); tracker.close();
  session.endedAt = new Date().toISOString();
  session.completed = true;
  await backup(session);
  screen(`
    <h1>Thank you!</h1>
    <p>The study is complete. Please download the data file${cfg.showResultsToParticipant ? ' (your results are shown below)' : ''}.</p>
    <div class="row"><button class="btn primary" id="dl">Download data</button></div>
    <p class="muted small" id="dls"></p>
    <section id="res"></section>`);
  document.getElementById('dl').onclick = async () => {
    download(session);
    document.getElementById('dls').textContent = 'Downloaded. You may close this page.';
    await clearBackup();
  };
  if (cfg.showResultsToParticipant || new URLSearchParams(location.search).get('results') === '1') {
    renderResults(document.getElementById('res'), session, refs, media);
  }
}

start();
