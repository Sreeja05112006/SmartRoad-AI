# SmartRoad AI
**Context-Aware Object Detection and Tracking for Road Safety**

A Streamlit research prototype that analyzes road videos with pretrained YOLO detections, ByteTrack, and a small context-aware tracker.

## Setup — Windows
Recommended Python: 3.10 or 3.11.
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Setup — macOS/Linux
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```
Open the local URL printed by Streamlit, usually http://localhost:8501. The first run downloads `yolo11n.pt`, so internet access is needed.

## Dataset plan
1. Prototype with a short road video you are allowed to use.
2. For research evaluation, consider BDD100K. It includes driving videos and a multi-object tracking task. See https://github.com/bdd100k/bdd100k and check its current access terms and annotations.
3. Do not download the full dataset for the first prototype. Start with a small sample.

## Method
- Baseline: YOLO detections + ByteTrack.
- Proposed prototype: YOLO detections + association based on predicted motion, bounding-box IoU, and a short class history, using Hungarian assignment.
- The context-aware tracker is an educational starting point and has not been benchmark-validated. Do not claim it improves accuracy until you evaluate it.

## Evaluation
Run both methods on the same videos and settings. FPS and counts are operational statistics, not tracking-accuracy metrics. For claims about tracking quality, use annotated ground-truth sequences and established metrics such as IDF1, HOTA, MOTA, and ID switches.

## Limitations
This is an academic prototype, not a certified road-safety or autonomous-driving system. Detections may fail in low light, occlusion, or crowded scenes.
