// recorder.js — camera recording of landmarks only (no video is stored).
// Flow: ready (live preview) -> countdown -> recording -> review (skeleton playback)
//       -> "Start over" (back to ready) or "Confirm & submit" (resolves).

import { frameLoop } from './camera.js';
import { drawFrame, imageMap, SkeletonPlayer, colors } from './skeleton.js';

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
      <div class="media review-wrap" hidden><canvas class="review"></canvas></div>
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
    wrap.style.aspectRatio = ar; this.$('.review-wrap').style.aspectRatio = ar;
    this.camera.video.play().catch(() => {});
    return this;
  }

  destroy() {
    if (this.loop) this.loop.stop();
    if (this.player) this.player.destroy();
    clearInterval(this._timer);
    this.el.remove();
  }

  run() {
    const rc = this.cfg.recording;
    const t0 = performance.now();
    const log = { attempts: 0, discarded: [], startedAt: 0 };
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
      octx.clearRect(0, 0, overlay.width, overlay.height);
      const c = colors();
      drawFrame(octx, f, imageMap(overlay.width, overlay.height, true), state === 'recording' ? c.you : c.live, { lw: Math.max(2, overlay.width / 250) });
      if (state === 'ready') status.textContent = f.pose ? 'Make sure your head, shoulders and hands are visible.' : 'We cannot see you yet — move into the camera view.';
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
        const cd = this.$('.countdown'); cd.hidden = false;
        for (let k = rc.countdownSec; k > 0; k--) { cd.textContent = k; await new Promise((r) => setTimeout(r, 1000)); }
        cd.hidden = true;
        frames = []; recStart = performance.now();
        if (this.tracker.resetClock) this.tracker.resetClock();
        log.attempts++;
        state = 'recording';
        this.$('.rec-dot').hidden = false; status.textContent = 'Recording… sign now. Press Stop when you are done.';
        show(['.b-stop']); this.$('.b-stop').disabled = true;
      };
      stopRecording = () => {
        if (state !== 'recording') return;
        state = 'review';
        this.$('.rec-dot').hidden = true;
        octx.clearRect(0, 0, overlay.width, overlay.height);
        this.$('.cam-wrap').hidden = true; this.$('.review-wrap').hidden = false;
        const n = frames.length, poseRatio = n ? frames.filter((f) => f.pose).length / n : 0;
        const handRatio = n ? frames.filter((f) => f.left_hand || f.right_hand).length / n : 0;
        this.current = { frames, durationMs: n ? frames[n - 1].t : 0, nFrames: n, poseRatio, handRatio, recordedAt: Math.round(recStart - t0) };
        status.textContent = poseRatio < rc.minPoseFrameRatio
          ? 'Your body was not tracked well in this recording. You may want to start over.'
          : 'This is how your signing was captured. Start over or confirm.';
        this.player = new SkeletonPlayer(this.$('.review'));
        this.player.set([{ frames, color: colors().you, map: (w, h) => imageMap(w, h, true) }]);
        this.player.play();
        show(['.b-over', '.b-submit']);
      };
      const startOver = () => {
        if (this.player) { this.player.destroy(); this.player = null; }
        const c = this.current;
        log.discarded.push(rc.saveDiscardedAttempts ? c : { durationMs: c.durationMs, nFrames: c.nFrames, poseRatio: c.poseRatio, recordedAt: c.recordedAt });
        this.current = null;
        this.$('.review-wrap').hidden = true; this.$('.cam-wrap').hidden = false;
        this.camera.video.play().catch(() => {});
        state = 'ready'; show(['.b-start']);
      };
      const submit = () => {
        state = 'done';
        if (this.player) this.player.pause();
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
