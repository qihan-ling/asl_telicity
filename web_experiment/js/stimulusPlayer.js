// stimulusPlayer.js — self-contained stimulus presentation module.
// It knows nothing about the camera, MediaPipe or the experiment; it only shows a video
// according to a policy and reports what was watched. Swap or reconfigure freely.
//
// Policies:
//   { mode: 'replay', maxReplays: Infinity | n, gapMs, continueLabel }
//       plays once automatically; participant may replay (up to maxReplays) and continues
//       when ready (continue becomes available after the first complete viewing).
//   { mode: 'fixed', plays: k, gapMs, continueLabel, autoAdvance: false }
//       plays exactly k times with a gap, no replay; then shows continue (or advances).
//
// run(src) resolves with:
//   { mode, plays: [{ start, end, completed }], totalPlays, replaysUsed, readyAt, firstFrameAt }
// (times are ms since run() was called).

export class StimulusPlayer {
  constructor(policy = { mode: 'replay', maxReplays: Infinity }) {
    this.policy = { gapMs: 600, continueLabel: 'Continue', ...policy };
    this.el = document.createElement('div');
    this.el.className = 'stim';
    this.el.innerHTML = `
      <div class="media"><video class="stim-video" muted playsinline webkit-playsinline preload="auto" disablepictureinpicture></video>
        <button class="btn overlay-play" hidden>Play video</button></div>
      <div class="status stim-status"></div>
      <div class="row">
        <button class="btn stim-replay" hidden>Replay</button>
        <button class="btn primary stim-continue" hidden></button>
      </div>`;
    this.video = this.el.querySelector('video');
    this.btnReplay = this.el.querySelector('.stim-replay');
    this.btnContinue = this.el.querySelector('.stim-continue');
    this.btnPlay = this.el.querySelector('.overlay-play');
    this.status = this.el.querySelector('.stim-status');
    this.btnContinue.textContent = this.policy.continueLabel;
  }

  mount(container) { container.appendChild(this.el); return this; }
  destroy() { this.video.pause(); this.video.removeAttribute('src'); this.video.load(); this.el.remove(); }

  async run(src) {
    const p = this.policy, v = this.video, t0 = performance.now();
    const now = () => Math.round(performance.now() - t0);
    const log = { mode: p.mode, policy: { ...p, maxReplays: p.maxReplays === Infinity ? 'Infinity' : p.maxReplays }, plays: [], totalPlays: 0, replaysUsed: 0, readyAt: null };
    v.src = src;
    await new Promise((res, rej) => {
      if (v.readyState >= 2) return res();
      v.addEventListener('loadeddata', res, { once: true });
      v.addEventListener('error', () => rej(new Error('Could not load ' + src)), { once: true });
    });

    const playOnce = () => new Promise((resolve) => {
      const rec = { start: now(), end: null, completed: false };
      log.plays.push(rec); log.totalPlays++;
      v.currentTime = 0;
      const onEnd = () => { rec.end = now(); rec.completed = true; v.removeEventListener('ended', onEnd); resolve(rec); };
      v.addEventListener('ended', onEnd);
      const pr = v.play();
      if (pr && pr.catch) pr.catch(() => {
        // autoplay blocked (rare: muted video after a click is allowed) -> ask for a tap
        this.btnPlay.hidden = false;
        this.btnPlay.onclick = () => { this.btnPlay.hidden = true; v.play(); };
      });
    });
    const wait = (ms) => new Promise((r) => setTimeout(r, ms));

    return new Promise(async (resolveRun) => {
      const finish = () => {
        log.readyAt = now();
        this.btnReplay.hidden = true; this.btnContinue.hidden = true;
        resolveRun(log);
      };
      this.btnContinue.onclick = finish;

      if (p.mode === 'fixed') {
        const k = Math.max(1, p.plays || 1);
        for (let i = 0; i < k; i++) {
          this.status.textContent = k > 1 ? `Viewing ${i + 1} of ${k}` : '';
          await playOnce();
          if (i < k - 1) await wait(p.gapMs);
        }
        this.status.textContent = '';
        if (p.autoAdvance) return finish();
        this.btnContinue.hidden = false;
        return;
      }

      // replay mode
      const max = p.maxReplays ?? Infinity;
      const refresh = (playing) => {
        const left = max - log.replaysUsed;
        this.btnReplay.hidden = max === 0;
        this.btnReplay.disabled = playing || left <= 0;
        this.btnReplay.textContent = max === Infinity ? 'Replay' : `Replay (${Math.max(0, left)} left)`;
        this.btnContinue.hidden = log.plays.every((x) => !x.completed);
        this.btnContinue.disabled = playing;
      };
      this.btnReplay.onclick = async () => {
        if (log.replaysUsed >= max) return;
        log.replaysUsed++; refresh(true);
        await playOnce(); refresh(false);
      };
      refresh(true);
      await playOnce();
      refresh(false);
    });
  }
}
