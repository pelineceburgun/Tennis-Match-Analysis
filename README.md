# Tennis Match Analysis Pipeline

An end-to-end computer vision pipeline for tennis match analysis, combining object detection, action classification, player tracking, and rule-based point prediction.

---

## Pipeline Architecture

```
Video Input
    ↓
[YOLOv11n — Detection]
    → player-front bbox
    → player-back bbox
    → tennis-ball bbox
    ↓
[ByteTrack — Tracking]
    → Consistent player IDs across frames
    ↓
[YOLOv11m-cls — Action Classification]
    → forehand / backhand / serve / ready_position
    ↓
[Rule-based Analytics]
    → Rally detection
    → Point prediction (ball_out, double_fault)
    → Shot distribution per player
    ↓
Annotated Video Output
```

---

## Models

### Model 1 — Player & Ball Detection
- **Architecture:** YOLOv11n (Nano)
- **Dataset:** tennis-tracker (Roboflow Universe) — 2,489 images
- **Classes:** `player-front`, `player-back`, `tennis-ball`
- **Training:** 50 epochs, ImageNet pretrained checkpoint, 640x640

| Class | mAP@50 |
|-------|--------|
| player-front | 99.5% |
| player-back | 99.5% |
| tennis-ball | 56.5% |
| **Overall** | **85.2%** |

### Model 2 — Action Classifier
- **Architecture:** YOLOv11m-cls (Medium)
- **Dataset:** Tennis Player Actions Dataset (Kaggle) — 2,000 images
- **Classes:** `forehand`, `backhand`, `serve`, `ready_position`
- **Training:** 50 epochs, ImageNet pretrained, dropout=0.3, early stopping

| Metric | Score |
|--------|-------|
| Top-1 Accuracy | 91.5% |
| Top-5 Accuracy | 100% |

Confusion matrix highlights:
- `serve`: 1.00 — perfectly classified
- `forehand`: 0.95
- `ready_position`: 0.94
- `backhand`: 0.90 (7% confused with forehand — expected, similar body mechanics)

---

## Features

- **Dual-model pipeline** — detection and classification are separate, modular models
- **ByteTrack integration** — consistent player ID assignment across frames
- **Ball trail visualization** — top N ball positions drawn as fading trail
- **Temporal smoothing** — action labels smoothed over last 10 frames to reduce noise
- **Rally detection** — ball visibility used as rally state signal
- **Point prediction (rule-based)**
  - `ball_out`: ball disappears after a shot → last player to hit loses the point
  - `double_fault`: same player serves 3+ times in 5 frames
- **Live scoreboard overlay** — score + rally status rendered on each frame
- **Match analytics summary** — shot distribution per player at end of video

---

## Datasets

| Dataset | Source | Usage |
|---------|--------|-------|
| tennis-tracker | Roboflow Universe (tennistracker-dogbm) | Detection training |
| Tennis Player Actions Dataset | Kaggle (orvile) | Action classification training |

---

## Limitations

- **Tennis ball detection is weak (56.5% mAP)** — small object + motion blur. A dedicated model like TrackNetV2 would significantly improve this.
- **Point prediction is rule-based** — not learned from real match data. Ground truth point labels would enable a proper ML-based predictor.
- **Tracker instability** — ByteTrack requires continuous video. Static image sequences cause ID fragmentation.
- **No real broadcast video in testing** — pipeline was validated on image sequences due to download constraints. Performance on real match footage is expected to be significantly better.
- **Two-player assumption** — pipeline maps all detections to two player slots. Multi-player scenes (practice, doubles) are not supported.

---

## Future Work

- Replace ball detection with TrackNetV2 for motion-aware ball tracking
- Add pose estimation layer (YOLOv11-pose or MediaPipe) between detection and classification
- Collect ground truth point labels from real match footage to train an ML-based point predictor
- Add court line detection for court zone analytics (which court zone does each shot come from)
- Extend to doubles matches

---

## Tech Stack

`Python` `PyTorch` `Ultralytics YOLOv11` `OpenCV` `Supervision` `Roboflow` `Kaggle`
