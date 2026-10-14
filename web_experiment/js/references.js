// references.js — landmarks for the stimulus videos, made with the *same* browser tracker
// as the participant data. Built once with precompute.html -> data/references.json.
// If that file is missing, the experiment computes them at start-up (slower).

import { drawFrame, imageMap, colors } from './skeleton.js';

export async function loadReferences(url) {
  try {
    const r = await fetch(url, { cache: 'no-cache' });
    if (!r.ok) return null;
    return await r.json();
  } catch { return null; }
}

/** Run the tracker over every frame of a video file (seeking frame by frame). */
export async function computeReference(tracker, src, { fps = 25, onProgress, canvas, host = document.body } = {}) {
  const v = document.createElement('video');
  v.muted = true; v.playsInline = true; v.setAttribute('playsinline', ''); v.preload = 'auto';
  v.style.cssText = 'position:absolute;width:2px;height:2px;opacity:0;pointer-events:none';
  host.appendChild(v);
  // Load as a blob: always seekable, even from servers without HTTP range support.
  const blobUrl = src.startsWith('blob:') ? src : URL.createObjectURL(await (await fetch(src)).blob());
  v.src = blobUrl;
  await new Promise((res, rej) => { v.onloadeddata = res; v.onerror = () => rej(new Error('cannot load ' + src)); });
  const n = Math.max(1, Math.floor(v.duration * fps + 1e-6));
  const frames = [];
  const base = performance.now();
  const ctx = canvas ? canvas.getContext('2d') : null;
  for (let i = 0; i < n; i++) {
    const target = Math.min(v.duration - 1e-3, (i + 0.5) / fps);
    await new Promise((res) => { v.onseeked = res; v.currentTime = target; });
    if (Math.abs(v.currentTime - target) > 0.5 / fps) throw new Error('Video is not seekable: ' + src);
    const t = Math.round((i * 1000) / fps);
    const f = tracker.processFrame(v, base + t);
    frames.push({ t, ...f });
    if (ctx) {
      canvas.width = v.videoWidth; canvas.height = v.videoHeight;
      ctx.drawImage(v, 0, 0);
      drawFrame(ctx, f, imageMap(canvas.width, canvas.height, false), colors().ref, { lw: 2 });
    }
    if (onProgress) onProgress(i + 1, n);
    if (i % 5 === 4) await new Promise((r) => setTimeout(r, 0));
  }
  const out = { width: v.videoWidth, height: v.videoHeight, fps, duration: v.duration, frames };
  v.remove();
  if (blobUrl !== src) URL.revokeObjectURL(blobUrl);
  return out;
}

export function detectionStats(ref) {
  const n = ref.frames.length || 1;
  const c = (k) => ref.frames.filter((f) => f[k]).length / n;
  return { pose: c('pose'), left_hand: c('left_hand'), right_hand: c('right_hand') };
}
