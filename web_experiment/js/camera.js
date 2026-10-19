// camera.js — one shared front-camera stream + a frame loop that feeds the tracker.

export class Camera {
  constructor() {
    this.video = document.createElement('video');
    this.video.className = 'cam-video mirror';
    this.video.muted = true; this.video.playsInline = true;
    this.video.setAttribute('playsinline', ''); this.video.setAttribute('webkit-playsinline', '');
    this.stream = null;
  }
  async start() {
    if (this.stream) return;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error('Camera access is not available. Open this page over https:// (or http://localhost).');
    }
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
    });
    this.video.srcObject = this.stream;
    await this.video.play();
    if (!this.video.videoWidth) await new Promise((r) => this.video.addEventListener('loadedmetadata', r, { once: true }));
  }
  get width() { return this.video.videoWidth; }
  get height() { return this.video.videoHeight; }
  get settings() {
    const t = this.stream && this.stream.getVideoTracks()[0];
    const s = t && t.getSettings ? t.getSettings() : {};
    return { width: this.width, height: this.height, frameRate: s.frameRate ?? null, facingMode: s.facingMode ?? null };
  }
  stop() { if (this.stream) this.stream.getTracks().forEach((t) => t.stop()); this.stream = null; }
}

/**
 * Runs cb(nowMs) for every new camera frame until stop() is called.
 * Uses requestVideoFrameCallback when available; a watchdog restarts playback and switches to a
 * requestAnimationFrame loop if frames stop arriving (some browsers pause a moved <video> or never
 * fire video-frame callbacks). Errors thrown by cb are counted and the last one is kept in
 * loop.stats instead of failing silently.
 */
export function frameLoop(video, cb) {
  let on = true, handle = null, rafHandle = null;
  let useRVFC = 'requestVideoFrameCallback' in HTMLVideoElement.prototype;
  let lastTime = -1, lastFire = performance.now();
  const stats = { calls: 0, errors: 0, lastError: null, mode: useRVFC ? 'video-frame' : 'animation-frame', restarts: 0 };
  const run = () => {
    lastFire = performance.now(); stats.calls++;
    try { cb(performance.now()); } catch (e) { stats.errors++; stats.lastError = String((e && e.message) || e); if (stats.errors <= 3) console.error(e); }
  };
  const stepRVFC = () => { if (!on || !useRVFC) return; run(); handle = video.requestVideoFrameCallback(stepRVFC); };
  const stepRAF = () => {
    if (!on) return;
    if (video.currentTime !== lastTime) { lastTime = video.currentTime; run(); }
    rafHandle = requestAnimationFrame(stepRAF);
  };
  if (useRVFC) handle = video.requestVideoFrameCallback(stepRVFC); else rafHandle = requestAnimationFrame(stepRAF);
  const watchdog = setInterval(() => {
    if (!on) return;
    if (performance.now() - lastFire < 1500) return;
    stats.restarts++;
    if (video.paused) video.play().catch(() => {});
    if (useRVFC) { // video-frame callbacks have stalled: fall back to polling
      useRVFC = false; stats.mode = 'animation-frame';
      try { video.cancelVideoFrameCallback && video.cancelVideoFrameCallback(handle); } catch {}
      rafHandle = requestAnimationFrame(stepRAF);
    }
    lastFire = performance.now();
  }, 1000);
  return {
    stats,
    stop() {
      on = false; clearInterval(watchdog);
      if (rafHandle) cancelAnimationFrame(rafHandle);
      try { if (handle && video.cancelVideoFrameCallback) video.cancelVideoFrameCallback(handle); } catch {}
    },
  };
}
