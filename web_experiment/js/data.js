// data.js — session record, crash backup (IndexedDB, this device only) and JSON download.

export function newSession(cfg, pid, extra = {}) {
  return {
    schema: 'sign-imitation-landmarks/v1',
    studyId: cfg.studyId,
    participantId: pid,
    sessionId: (crypto.randomUUID ? crypto.randomUUID() : String(Date.now()) + Math.random().toString(16).slice(2)),
    startedAt: new Date().toISOString(),
    endedAt: null,
    completed: false,
    userAgent: navigator.userAgent,
    screen: { w: screen.width, h: screen.height, dpr: window.devicePixelRatio, viewport: [innerWidth, innerHeight] },
    config: cfg,
    landmarkFormat: 'per frame {t: ms, pose: [[x,y,z,visibility] x33] | null, left_hand: [[x,y,z] x21] | null, right_hand: ...}; x,y normalized image coords of the UNMIRRORED camera image; left/right = signer\'s own hands',
    ...extra,
    trials: [],
  };
}

const replacer = (k, v) => {
  if (typeof k === 'string' && k.startsWith('_')) return undefined; // in-memory helpers
  if (v === Infinity) return 'Infinity';
  return v;
};
export const serialize = (s) => JSON.stringify(s, replacer);

export function download(session, name) {
  const blob = new Blob([serialize(session)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
  a.download = name || `${session.studyId}_${session.participantId || 'anon'}_${stamp}.json`;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 10000);
}

// ---- IndexedDB backup (survives an accidental reload; one session at a time) ----
const DB = 'sign-imitation-backup', STORE = 's';
function db() {
  return new Promise((res, rej) => {
    const r = indexedDB.open(DB, 1);
    r.onupgradeneeded = () => r.result.createObjectStore(STORE);
    r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error);
  });
}
export async function backup(session) {
  try {
    const d = await db();
    await new Promise((res, rej) => { const tx = d.transaction(STORE, 'readwrite'); tx.objectStore(STORE).put(serialize(session), 'current'); tx.oncomplete = res; tx.onerror = () => rej(tx.error); });
  } catch (e) { console.warn('backup failed', e); }
}
export async function loadBackup() {
  try {
    const d = await db();
    return await new Promise((res) => { const r = d.transaction(STORE).objectStore(STORE).get('current'); r.onsuccess = () => res(r.result ? JSON.parse(r.result) : null); r.onerror = () => res(null); });
  } catch { return null; }
}
export async function clearBackup() {
  try { const d = await db(); d.transaction(STORE, 'readwrite').objectStore(STORE).delete('current'); } catch {}
}
