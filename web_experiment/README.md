# Sign imitation web experiment

Participants watch a sign video and then copy it in front of their camera. MediaPipe (Tasks Vision
PoseLandmarker + HandLandmarker) tracks the body and both hands in the browser. **Only landmarks are
stored; no video is recorded.** At the end, each item gets a similarity score (0–100) against the
model signer, and the movements can be replayed as skeletons (reference vs. participant). The layout
works in laptop and iPad browsers (Chrome, Safari, Edge; portrait or landscape).

Stimuli: the 12 NGT signs from Wagner et al. 2024 (`stimuli/*.mp4`, converted from the original `.mov`
files to H.264 MP4 at 640×360, 25 fps, with no audio).

## Run it

```bash
cd web_experiment
python3 -m http.server 8000        # then open http://localhost:8000
```

- The camera only works on `https://` or `http://localhost`. To run on an iPad, host the folder over
  HTTPS, for example on GitHub Pages from this repo, or through a tunnel such as `cloudflared tunnel --url http://localhost:8000`.
- By default, MediaPipe loads from its CDN, so an internet connection is needed. For offline lab use, run
  `bash tools/fetch_mediapipe_assets.sh` once and set `mediapipe.source: 'local'` in `js/config.js`.

### One-time step: build the reference landmarks
Open `http://localhost:8000/precompute.html` and click **Build references**. It runs the *same*
tracker as the experiment over every frame of each stimulus and shows per-item detection rates.
Download `references.json` and save it as `data/references.json`. Rebuild it whenever the stimuli
or the MediaPipe settings change. If the file is missing, the experiment computes the references
while it loads (this works, but it is slower).

### URL options
| | |
|---|---|
| `?pid=P01` | participant ID (otherwise the start screen asks for it) |
| `?list=1` … `?list=4` | Wagner Exp. 1A counterbalancing order (default: random) |
| `?n=6` | only the first *n* items |
| `?plays=2` | stimulus player in **fixed** mode: every item plays exactly 2 times, with no replays |
| `?replays=3` / `?replays=inf` | stimulus player in **replay** mode, with a replay limit |
| `?mock=1` | synthetic landmarks instead of MediaPipe (for testing the flow) |

Everything else is in `js/config.js`: item order, stimulus policy, countdown, maximum recording
duration, MediaPipe model size (`lite` is faster on older iPads), similarity settings, and whether
participants see their results.

## Trial flow
1. **Watch** (`js/stimulusPlayer.js`): a standalone module that knows nothing about the camera or
   the experiment. Policy `replay` (one automatic play, then optional replays) or `fixed`
   (*k* automatic plays, which gives every participant equal exposure). Every play is logged.
2. **Sign** (`js/recorder.js`): a live mirrored preview with the skeleton overlay, then **Start recording**,
   a 3-2-1 countdown, recording, and **Stop** (or auto-stop at `maxDurationSec`). The captured skeleton
   is replayed, followed by **Start over** or **Confirm & submit**. Discarded attempts are counted;
   their landmarks are kept only if `saveDiscardedAttempts: true`.
3. *(Optional)* post-trial question: set `postTrialQuestion` in the config (off by default).

The data is backed up to the browser's IndexedDB after every trial. If the page is reloaded by
accident, the start screen offers to download the unfinished session.

## Data file (`<study>_<pid>_<time>.json`)
One file per session, with the session info, device, camera resolution, tracker and model versions,
the full config, and for each trial:
`item`, `stimulus` (play log), `recording` (`frames`, `attempts`, `discarded`, `durationMs`,
`poseRatio`, `handRatio`, `width`, `height`) and `similarity`.

Each frame has the same structure as `processed_data/asl_pkl`:
`{t, pose: [[x,y,z,visibility]×33], left_hand: [[x,y,z]×21] | null, right_hand: … }`.
Coordinates are normalized to the **unmirrored** camera image. Left and right are the signer's own
hands, matched to pose wrists 15 and 16, as Holistic does.

```bash
python tools/load_results.py results/*.json --csv scores.csv --pkl-dir ../processed_data/imitation_pkl
node   tools/rescore.mjs data/references.json results/*.json --tau 0.6 > rescored.csv
```
`load_results.py` writes one pickle per trial in the asl_telicity format (a list of dicts of numpy
arrays), so your existing scripts can read participant productions directly.

## Similarity score (`js/similarity.js`)
1. Normalize each sequence: center on the shoulder midpoint and scale by the median shoulder width.
   Shoulders are used instead of the torso because the NGT videos are cropped above the hips.
2. Per-frame features: both wrist positions, both elbows, and each hand's shape (20 finger joints relative
   to the wrist, scaled by hand size).
3. Trim to the active window, i.e. the frames where a wrist is raised above the rest position, then
   resample to 25 Hz.
4. Run DTW (so speed differences don't count). The per-frame cost is a weighted wrist + elbow + hand-shape
   distance, with a penalty when one side sees a hand and the other doesn't.
5. Run it both **directly** and **mirrored** (left and right swapped, as when a participant mirrors the
   model). The better result is kept as `score`, and both are saved (`direct`, `mirror`, `mirrored`).
6. `score = 100 · exp(−distance / τ)`, with τ = 0.6. The raw `distance` and its components are also
   saved, so you can recalibrate τ with `tools/rescore.mjs` once you have pilot data.

Sanity check on the asl_telicity pickles (`node test/similarity_test.mjs pkl_seqs.json`):
AUC ≈ 0.98 for same-sign vs. different-sign pairs. Identical input scores 100, a simulated imitation
(different speed, size, position and noise) scores about 66, and a different sign about 28.

## Files
```
index.html          experiment
precompute.html     builds data/references.json
js/config.js        all settings          js/stimuli.js        items + Wagner lists
js/stimulusPlayer.js  watch module        js/recorder.js       record / start over / submit
js/tracker.js       MediaPipe wrapper     js/similarity.js     normalization + DTW score
js/skeleton.js      drawing/animation     js/results.js        end-of-study results view
js/camera.js, js/references.js, js/data.js, js/experiment.js
tools/              asset fetcher, Python loader, offline re-scorer
```
