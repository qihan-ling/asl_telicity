"""
Load data files from the web experiment.

  python tools/load_results.py results/*.json --pkl-dir ../processed_data/imitation_pkl --csv scores.csv

- --csv      one row per trial (participant, item, telicity, score, mirrored, plays, attempts, ...)
- --pkl-dir  one .pkl per trial in the same format as processed_data/asl_pkl:
             a list of frames {'pose': (33,4) array | None, 'left_hand': (21,3) | None, 'right_hand': (21,3) | None}
             named <participant>_<telicity>_<item>.pkl, so the asl_telicity scripts can read them.

In Python:  from load_results import load_session, trial_to_holistic
"""
import argparse, csv, json, os, pickle
import numpy as np


def _arr(a, ncol):
    if a is None:
        return None
    x = np.array([[np.nan if v is None else v for v in row[:ncol]] for row in a], dtype=float)
    return x


def trial_to_holistic(trial):
    """Frames of the submitted attempt as a list of dicts with numpy arrays (asl_telicity format)."""
    return [
        {"pose": _arr(f["pose"], 4), "left_hand": _arr(f["left_hand"], 3), "right_hand": _arr(f["right_hand"], 3), "t_ms": f["t"]}
        for f in trial["recording"]["frames"]
    ]


def load_session(path):
    with open(path) as fh:
        return json.load(fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--csv")
    ap.add_argument("--pkl-dir")
    a = ap.parse_args()
    rows = []
    for p in a.files:
        s = load_session(p)
        for t in s["trials"]:
            sim = t.get("similarity") or {}
            rec = t["recording"]
            rows.append({
                "file": os.path.basename(p), "participant": s.get("participantId"), "session": s.get("sessionId"),
                "trial": t["index"] + 1, "item": t["item"]["id"], "telicity": t["item"].get("telicity"),
                "domain": t["item"].get("domain"), "score": sim.get("score"), "distance": sim.get("distance"),
                "mirrored": sim.get("mirrored"), "wrist": (sim.get("components") or {}).get("wrist"),
                "elbow": (sim.get("components") or {}).get("elbow"), "hand": (sim.get("components") or {}).get("hand"),
                "plays": t["stimulus"]["totalPlays"], "attempts": rec["attempts"], "duration_ms": rec["durationMs"],
                "n_frames": rec["nFrames"], "pose_ratio": rec["poseRatio"], "hand_ratio": rec["handRatio"],
                "response": (t.get("response") or {}).get("answer"),
            })
            if a.pkl_dir:
                os.makedirs(a.pkl_dir, exist_ok=True)
                name = f"{s.get('participantId')}_{t['item'].get('telicity')}_{t['item']['id']}.pkl"
                with open(os.path.join(a.pkl_dir, name), "wb") as fh:
                    pickle.dump(trial_to_holistic(t), fh)
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    for r in rows:
        sc = "-" if r["score"] is None else f"{r['score']:.1f}"
        print(f"{r['participant']:>8} {r['trial']:>3} {r['item']:<10} {sc:>6} mirrored={r['mirrored']} plays={r['plays']} attempts={r['attempts']}")


if __name__ == "__main__":
    main()
