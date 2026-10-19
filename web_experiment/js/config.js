// config.js — every study setting lives here. URL parameters can override a few:
//   ?pid=P01        participant id (otherwise asked on the start screen)
//   ?list=1..4      use a Wagner Exp. 1A counterbalancing list order
//   ?n=6            only the first n items (after ordering)
//   ?plays=2        stimulus player in 'fixed' mode, 2 plays per item
//   ?replays=3      stimulus player in 'replay' mode, up to 3 replays ('inf' = unlimited)
//   ?demo=1         demo version (CONFIG.demo: 3 fixed items, no ID prompt, results shown)
//   ?mock=1         run without MediaPipe/camera model (synthetic landmarks; for testing the flow)

export const CONFIG = {
  studyId: 'ngt_sign_imitation',
  stimuliBase: 'stimuli/',
  referencesUrl: 'data/references.json',

  // 'random' | 'fixed' (stimuli.js order) | 'list1'..'list4' (Wagner et al. Exp. 1A lists)
  itemOrder: 'random',
  nItems: null, // null = all items

  // ---- Stimulus player (independent module, js/stimulusPlayer.js) ----
  // mode 'replay': plays once, then participant may replay up to maxReplays times (Infinity = unlimited).
  // mode 'fixed' : plays exactly `plays` times automatically, no replay button (equal exposure).
  stimulusPlayer: { mode: 'replay', maxReplays: Infinity, gapMs: 600, continueLabel: "I'm ready to sign" },
  // stimulusPlayer: { mode: 'fixed', plays: 2, gapMs: 800, continueLabel: "I'm ready to sign" },

  // ---- Recording ----
  recording: {
    countdownSec: 3,
    maxDurationSec: 8,        // auto-stop
    minDurationSec: 0.8,
    minPoseFrameRatio: 0.6,   // warn if the body was found in fewer frames than this
    saveDiscardedAttempts: false, // also keep landmarks of attempts the participant started over
    reviewVideo: true,        // after Stop, replay a temporary in-memory clip to the participant (never saved)
    showSkeleton: false,      // draw the tracked skeleton over the camera during trials (camera check always shows it)
  },

  // ---- Camera framing check (js/framing.js) ----
  // Participants can only continue from the camera check when shoulders and both hands
  // are inside the centre box at a sensible distance, held for holdMs.
  framing: {
    box: { x0: 0.2, x1: 0.8, y0: 0.06, y1: 0.96 },
    shoulderY: [0.2, 0.7],
    shoulderWidth: [0.15, 0.5],
    requireHands: true,
    holdMs: 1000,
  },

  // ---- Demo version (?demo=1 or demo.html) ----
  // Fixed short item list, no participant-ID prompt, results shown at the end.
  demo: { items: ['buy', 'talk', 'die'], pidPrefix: 'demo' },

  // ---- Optional post-trial question (none for now) ----
  // e.g. { type: 'text', prompt: 'What do you think this sign means?', required: true }
  postTrialQuestion: null,

  // ---- MediaPipe (Tasks Vision: PoseLandmarker + HandLandmarker) ----
  mediapipe: {
    source: 'cdn',            // 'cdn' or 'local' (run tools/fetch_mediapipe_assets.sh first; files go to vendor/)
    version: '0.10.14',
    poseModel: 'full',        // 'lite' | 'full' | 'heavy'  ('lite' is faster on older iPads)
    delegate: 'GPU',          // falls back to CPU automatically
    handCrop: true,           // detect hands on an upscaled crop around the body (helps small/far signers)
    handCropScale: 4.0,       // crop side = handCropScale x shoulder width
    handCropSize: 512,
  },

  // ---- Similarity (js/similarity.js) ----
  similarity: { tau: 0.6, allowMirror: true },

  // ---- End of experiment ----
  showResultsToParticipant: true,
  includeAlignmentInData: false, // DTW path per trial (larger file)
};

export function applyUrlOverrides(cfg) {
  const q = new URLSearchParams(location.search);
  const c = structuredClone(cfg);
  if (q.get('list')) c.itemOrder = 'list' + q.get('list');
  if (q.get('n')) c.nItems = parseInt(q.get('n'), 10);
  if (q.get('order')) c.itemOrder = q.get('order');
  if (q.get('plays')) c.stimulusPlayer = { ...c.stimulusPlayer, mode: 'fixed', plays: parseInt(q.get('plays'), 10) };
  if (q.get('replays')) c.stimulusPlayer = { ...c.stimulusPlayer, mode: 'replay', maxReplays: q.get('replays') === 'inf' ? Infinity : parseInt(q.get('replays'), 10) };
  c.mock = q.get('mock') === '1';
  c.pid = q.get('pid') || '';
  c.isDemo = q.get('demo') === '1';
  if (c.isDemo) {
    c.itemOrder = 'ids:' + c.demo.items.join(',');
    c.nItems = null;
    c.showResultsToParticipant = true;
    if (!c.pid) c.pid = c.demo.pidPrefix + '-' + Math.random().toString(36).slice(2, 7);
  }
  return c;
}
