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
import { evaluateFraming, drawBox, DEFAULT_FRAMING } from './framing.js';

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
    ${cfg.isDemo ? `<p class="muted">This is a short demo with ${cfg.demo.items.length} signs.</p>` : ''}
    <p class="muted">No video or pictures of you are saved or sent anywhere. Only the movements of your body and hands are saved, as numbers.</p>
    <label class="field" ${cfg.isDemo ? 'hidden' : ''}>Participant ID <input id="pid" autocomplete="off" value="${esc(cfg.pid)}"></label>
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
  const tracker = await createTracker(cfg.mediapipe, { mock: cfg.mock, mockFail: cfg.mockFail });

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
  const fr = cfg.framing || DEFAULT_FRAMING;
  screen(`
    <h2>Camera check</h2>
    <p>Sit so that your shoulders are inside the box, then raise both hands in front of your chest.
      Lines will follow your body and hands.</p>
    <div class="media cam-wrap" id="cw"><canvas class="overlay"></canvas></div>
    <ul class="checks" id="checks"></ul>
    <p class="status" id="st">Looking for you…</p>
    <div class="row"><button class="btn primary" id="ok" disabled>Continue</button></div>`);
  const cw = document.getElementById('cw');
  cw.prepend(camera.video); cw.style.aspectRatio = `${camera.width} / ${camera.height}`;
  camera.video.play().catch(() => {});
  const ov = cw.querySelector('canvas'), ctx = ov.getContext('2d');
  const ok = document.getElementById('ok'), st = document.getElementById('st'), ul = document.getElementById('checks');
  let goodSince = null, passed = false, tracked = 0;
  const t0 = performance.now();
  const loop = frameLoop(camera.video, (now) => {
    const f = tracker.processFrame(camera.video, now);
    ov.width = camera.width; ov.height = camera.height;
    const ev = evaluateFraming(f, fr);
    drawBox(ctx, ov.width, ov.height, ev.ok, fr);
    if (f.pose) tracked++;
    drawFrame(ctx, f, imageMap(ov.width, ov.height, true), colors().live, { lw: Math.max(2, ov.width / 250) });
    ul.innerHTML = ev.checks.map((c) => `<li class="${c.ok ? 'ok' : ''}">${c.ok ? '✓' : '○'} ${c.label}</li>`).join('');
    if (ev.ok) {
      if (goodSince === null) goodSince = now;
      if (now - goodSince >= fr.holdMs) passed = true;
    } else goodSince = null;
    const ts = tracker.status || {};
    if (passed) { ok.disabled = false; st.textContent = 'All set. You can continue.'; }
    else if (ts.rebuilding) st.textContent = 'Switching to a more compatible tracking mode… (a few seconds)';
    else if (ev.ok) st.textContent = 'Hold still…';
    else st.textContent = ev.checks.find((c) => !c.ok).hint;
  });
  // If tracking never starts (no frames, or every frame fails), say so instead of freezing.
  const diag = document.createElement('div');
  diag.className = 'note'; diag.hidden = true;
  st.after(diag);
  const watch = setInterval(() => {
    const ts = tracker.status || {}, ls = loop.stats;
    const stalled = performance.now() - t0 > 10000 && tracked === 0 && !ts.rebuilding;
    if (!stalled && !ts.fatal) return;
    const why = ls.calls === 0 ? 'No camera frames are arriving in the page.'
      : (ts.errors || ls.errors) ? 'Body tracking is failing on this device.' : 'No body has been detected yet.';
    diag.hidden = false;
    diag.innerHTML = `<strong>${why}</strong>
      <p>Please make sure no other app or tab is using the camera, use an up-to-date Chrome, Edge or Safari, and reload.
        If it still does not work, please send a screenshot of this box to the researcher.</p>
      <p class="small muted">${esc(navigator.userAgent)}<br>
        tracker: ${esc(ts.delegate || '?')}${ts.fellBack ? ' (fell back from GPU)' : ''} · frames: ${ls.calls} (loop: ${esc(ls.mode)}, restarts ${ls.restarts}) ·
        detected: ${tracked} · errors: ${(ts.errors || 0) + ls.errors}<br>${esc(ts.lastError || ls.lastError || '')}</p>
      <div class="row"><button class="btn" id="reload">Reload page</button></div>`;
    diag.querySelector('#reload').onclick = () => location.reload();
    if (tracked > 0) { diag.hidden = true; }
  }, 2000);
  await waitClick(ok);
  clearInterval(watch);
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
      <li>Watch your recording. Press <b>Start over</b> to record again, or <b>Confirm &amp; submit</b> to go on to the next sign.</li>
    </ol>
    <div class="note">
      <strong>What is saved</strong>
      <p>No video or pictures of you are saved or sent anywhere. After each attempt, your recording is replayed to you
        on this screen so you can check it; that clip exists only temporarily in this page and is deleted as soon as you
        press <b>Start over</b> or <b>Confirm &amp; submit</b>. While you sign, the page also computes the
        positions of points on your body and hands (for example your shoulders, elbows, wrists and finger joints)
        and saves only these numbers, which describe your movements. These points include a few
        points that locate your head (nose, eyes, ears, mouth corners), but no image of your face is kept, and those points are not shown on screen.
        At the end you will see your movements as simple stick figures.</p>
    </div>
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
  session.trackerStatus = { ...(tracker.status || {}) };
  session.endedAt = new Date().toISOString();
  session.completed = true;
  await backup(session);
  screen(`
    <h1>Thank you!</h1>
    <p>${cfg.isDemo ? 'The demo is complete. Your results are below. You can also download your movement data (numbers only) if you like.'
      : `The study is complete. Please download the data file${cfg.showResultsToParticipant ? ' (your results are shown below)' : ''}.`}</p>
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
