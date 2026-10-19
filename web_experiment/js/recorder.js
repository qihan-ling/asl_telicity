// recorder.js — records landmarks (saved) and, for the participant's own review only, a
// temporary video clip kept in memory.
// Flow: ready (live preview + centre box) -> countdown -> recording -> review
//       -> "Start over" (back to ready) or "Confirm & submit" (resolves).
// The review clip is never saved or sent: it lives in an in-memory Blob that is discarded
// as soon as the participant starts over or submits (and when the recorder is destroyed).

import { frameLoop } from './camera.js';
import { drawFrame, imageMap, colors } from './skeleton.js';
import { evaluateFraming, drawBox, DEFAULT_FRAMING } from './framing.js';

export class Recorder {
  constructor({ camera, tracker, cfg }) {
    this.camera = camera; this.tracker = tracker; this.cfg = cfg;
    this.el = document.createElement('div');
    this.el.className = 'rec';
    this.el.innerHTML = `
      <div class="media cam-wrap">
        <canvas class="overlay"></canvas>
        <div class="countdown" hidden></div>
        <div class="rec-dot" hidden>REC <span class="rec-time"></span></div>
      </div>
      <div class="media done-wrap" hidden>
        <video class="review mirror" muted playsinline webkit-playsinline loop hidden></video>
        <div class="done-msg"></div>
      </div>
      <div class="status rec-status"></div>
      <div class="row">
        <button class="btn primary b-start">Start recording</button>
        <button class="btn b-stop" hidden>Stop</button>
        <button class="btn b-over" hidden>Start over</button>
        <button class="btn primary b-submit" hidden>Confirm &amp; submit</button>
      </div>`;
    this.$ = (s) => this.el.querySelector(s);
  }

  mount(container) {
    container.appendChild(this.el);
    const wrap = this.$('.cam-wrap');
    wrap.prepend(this.camera.video);
    const ar = `${this.camera.width} / ${this.camera.height}`;
    wrap.style.aspectRatio = ar; this.$('.done-wrap').style.aspectRatio = ar;
    this.camera.video.play().catch(() => {});
    return this;
  }

  destroy() {
    if (this.loop) this.loop.stop();
    this._discardClip();
    this.el.remove();
  }

  // ---- temporary review clip (MediaRecorder, in memory only) ----
  _canRecordVideo() {
    return this.cfg.recording.reviewVideo !== false && typeof MediaRecorder !== 'undefined' && !!this.camera.stream;
  }
  _startClip() {
    this._discardClip();
    if (!this._canRecordVideo()) return;
    const types = ['video/mp4', 'video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'];
    const mimeType = types.find((t) => MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(t));
    try {
      const mr = new MediaRecorder(this.camera.stream, { ...(mimeType ? { mimeType } : {}), videoBitsPerSecond: 2500000 });
      const chunks = [];
      mr.ondataavailable = (e) => { if (e.data && e.data.size) chunks.push(e.data); };
      this._clipDone = new Promise((res) => { mr.onstop = () => res(chunks.length ? new Blob(chunks, { type: mr.mimeType || mimeType || 'video/webm' }) : null); });
      mr.start();
      this._mr = mr;
    } catch (e) { console.warn('review video unavailable', e); this._mr = null; }
  }
  async _stopClip() {
    const mr = this._mr; this._mr = null;
    if (!mr) return null;
    if (mr.state !== 'inactive') mr.stop();
    return this._clipDone;
  }
  _discardClip() {
    if (this._mr && this._mr.state !== 'inactive') { try { this._mr.stop(); } catch {} }
    this._mr = null;
    const v = this.$('.review');
    if (v) { v.pause(); v.removeAttribute('src'); v.load(); v.hidden = true; }
    if (this._clipUrl) { URL.revokeObjectURL(this._clipUrl); this._clipUrl = null; }
  }

  run() {
    const rc = this.cfg.recording;
    const fr = this.cfg.framing || DEFAULT_FRAMING;
    const t0 = performance.now();
    const log = { attempts: 0, discarded: [] };
    const overlay = this.$('.overlay'), octx = overlay.getContext('2d');
    const status = this.$('.rec-status');
    const show = (ids) => ['.b-start', '.b-stop', '.b-over', '.b-submit'].forEach((s) => (this.$(s).hidden = !ids.includes(s)));
    let state = 'ready', frames = [], recStart = 0;
    let stopRecording = () => {};

    const sizeOverlay = () => {
      const w = this.camera.width, h = this.camera.height;
      if (overlay.width !== w) { overlay.width = w; overlay.height = h; }
    };
    // live tracking loop (preview + recording)
    this.loop = frameLoop(this.camera.video, (now) => {
      if (state !== 'ready' && state !== 'countdown' && state !== 'recording') return;
      const f = this.tracker.processFrame(this.camera.video, now);
      sizeOverlay();
      const W = overlay.width, H = overlay.height;
      octx.clearRect(0, 0, W, H);
      const c = colors();
      if (state === 'ready') {
        // advisory only: shoulders in the box (hands may rest before recording)
        const ev = evaluateFraming(f, { ...fr, requireHands: false });
        drawBox(octx, W, H, ev.ok, fr);
        const bad = ev.checks.find((x) => !x.ok);
        status.textContent = ev.ok ? 'Ready. Keep your shoulders and hands inside the box while you sign.' : bad.hint;
      }
      // the skeleton is shown only on the camera-check screen unless recording.showSkeleton is true
      if (rc.showSkeleton) drawFrame(octx, f, imageMap(W, H, true), state === 'recording' ? c.you : c.live, { lw: Math.max(2, W / 250) });
      if (state === 'recording') {
        frames.push({ t: Math.round(now - recStart), ...f });
        const sec = (now - recStart) / 1000;
        this.$('.rec-time').textContent = sec.toFixed(1) + ' s';
        this.$('.b-stop').disabled = sec < rc.minDurationSec;
        if (sec >= rc.maxDurationSec) stopRecording();
      }
    });

    return new Promise((resolve) => {
      const startRecording = async () => {
        state = 'countdown'; show([]);
        status.textContent = 'Get ready…';
        const cd = this.$('.countdown'); cd.hidden = false;
        for (let k = rc.countdownSec; k > 0; k--) { cd.textContent = k; await new Promise((r) => setTimeout(r, 1000)); }
        cd.hidden = true;
        frames = []; recStart = performance.now();
        if (this.tracker.resetClock) this.tracker.resetClock();
        log.attempts++;
        state = 'recording';
        this._startClip();
        this.$('.rec-dot').hidden = false; status.textContent = 'Recording… sign now. Press Stop when you are done.';
        show(['.b-stop']); this.$('.b-stop').disabled = true;
      };
      stopRecording = async () => {
        if (state !== 'recording') return;
        state = 'done';
        show([]);
        this.$('.rec-dot').hidden = true;
        octx.clearRect(0, 0, overlay.width, overlay.height);
        const n = frames.length, poseRatio = n ? frames.filter((f) => f.pose).length / n : 0;
        const handRatio = n ? frames.filter((f) => f.left_hand || f.right_hand).length / n : 0;
        this.current = { frames, durationMs: n ? frames[n - 1].t : 0, nFrames: n, poseRatio, handRatio, recordedAt: Math.round(recStart - t0) };
        const poor = poseRatio < rc.minPoseFrameRatio;
        const secs = (this.current.durationMs / 1000).toFixed(1);
        const advice = poor ? 'Your body was not tracked well. You may want to start over.' : 'Happy with it? Submit, or start over to try again.';
        this.$('.cam-wrap').hidden = true; this.$('.done-wrap').hidden = false;
        const msg = this.$('.done-msg'), vid = this.$('.review');
        msg.hidden = false; msg.innerHTML = '<span>Preparing your recording…</span>';
        status.textContent = '';
        const blob = await this._stopClip();
        if (state !== 'done') return;
        if (blob) {
          this._clipUrl = URL.createObjectURL(blob);
          vid.src = this._clipUrl; vid.hidden = false; msg.hidden = true;
          vid.play().catch(() => {});
          this.current.review = 'video';
          status.textContent = `Your recording (${secs} s, shown only to you and not saved). ${advice}`;
        } else {
          this.current.review = 'none';
          msg.innerHTML = `<strong>Recorded ${secs} s</strong><span>${advice}</span>`;
        }
        show(['.b-over', '.b-submit']);
      };
      const startOver = () => {
        this._discardClip();
        const c = this.current;
        log.discarded.push(rc.saveDiscardedAttempts ? c : { durationMs: c.durationMs, nFrames: c.nFrames, poseRatio: c.poseRatio, recordedAt: c.recordedAt });
        this.current = null;
        this.$('.done-wrap').hidden = true; this.$('.cam-wrap').hidden = false;
        this.camera.video.play().catch(() => {});
        state = 'ready'; show(['.b-start']);
      };
      const submit = () => {
        state = 'submitted';
        this._discardClip();
        this.loop.stop();
        show([]);
        resolve({ ...this.current, attempts: log.attempts, discarded: log.discarded, submittedAt: Math.round(performance.now() - t0) });
      };
      this.$('.b-start').onclick = startRecording;
      this.$('.b-stop').onclick = stopRecording;
      this.$('.b-over').onclick = startOver;
      this.$('.b-submit').onclick = submit;
      show(['.b-start']);
    });
  }
}
