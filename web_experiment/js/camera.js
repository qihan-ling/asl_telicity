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

/** Runs cb(nowMs) for every new camera frame until stop() is called. */
export function frameLoop(video, cb) {
  let on = true, handle = null;
  const useRVFC = 'requestVideoFrameCallback' in HTMLVideoElement.prototype;
  let lastTime = -1;
  const step = (now) => {
    if (!on) return;
    if (useRVFC || video.currentTime !== lastTime) { lastTime = video.currentTime; try { cb(performance.now()); } catch (e) { console.error(e); } }
    handle = useRVFC ? video.requestVideoFrameCallback(step) : requestAnimationFrame(step);
  };
  handle = useRVFC ? video.requestVideoFrameCallback(step) : requestAnimationFrame(step);
  return { stop() { on = false; if (!useRVFC && handle) cancelAnimationFrame(handle); } };
}
