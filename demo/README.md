# Demo

- `demo.mp4`: one-minute walkthrough (business question, live replay of a real drive, the safety
  report, the fleet answer). Rebuilt from the trained model by `make_demo.py`.
- `screenshots/`: the Streamlit app's detection and trajectory tabs.

Everything here is generated from real nuScenes LiDAR and the trained planner:

```bash
python fleet_review.py        # writes inference_outputs/fleet_*.csv
python demo/make_demo.py      # GIFs, report images, demo.mp4
```
