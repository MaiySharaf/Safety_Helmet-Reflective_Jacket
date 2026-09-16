# 🦺 PPE Detection — Helmet & Safety Vest Compliance

An object detection system that checks whether a person in an image, video, or live camera feed is wearing the required **Personal Protective Equipment (PPE)** — specifically a **safety helmet** and a **high-visibility vest/jacket** — and flags non-compliance in real time.

Built for workplace safety monitoring on construction sites, factories, and warehouses.

---

## 📌 Overview

The pipeline detects people and their PPE items in an image, then applies a simple compliance rule on top of the detections:

| Detected | Status |
|---|---|
| Helmet ✅ + Vest ✅ | 🟢 Compliant — Wearing both |
| Helmet ✅ + Vest ❌ | 🟠 Partial — Helmet only |
| Helmet ❌ + Vest ✅ | 🟠 Partial — Vest only |
| Helmet ❌ + Vest ❌ | 🔴 Non-compliant — Wearing neither |

Three detection architectures were trained and compared on the same dataset to evaluate the accuracy/speed trade-off between a two-stage detector and modern single-stage YOLO models:

- **Faster R-CNN** — two-stage, region-proposal based, used as an accuracy baseline
- **YOLOv8** — single-stage, real-time
- **YOLOv11** — single-stage, latest Ultralytics architecture

The best-performing / fastest model is served through a **Streamlit** web app for interactive image and video inference.

---



## 🧠 Models

| Model | Type | Framework | Notes |
|---|---|---|---|
| Faster R-CNN | Two-stage detector | tensorflow | Higher accuracy, slower inference — used as baseline |
| YOLOv8 | One-stage detector | Ultralytics | Fast, good accuracy/speed trade-off |
| YOLOv11 | One-stage detector | Ultralytics | Latest architecture, used as the primary deployed model |

Trained weights are not committed to this repo (see [Model Weights](#-model-weights)).

### Results

| Model | mAP@0.5 | mAP@0.5:0.95 | Inference speed (FPS) |
|---|---|---|---|
| Faster R-CNN | _TBD_ | _TBD_ | _TBD_ |
| YOLOv8 | _0.9_ | _0.57_ |_13.2_ |
| YOLOv11 | _0.92_ | _0.59_ | _6_ |

> Fill in this table with your actual evaluation numbers once training/evaluation is complete.

---

## 📂 Dataset

**[Personal-Protective-Equipment (PPE) Dataset](https://www.kaggle.com/datasets/ndomalau/personal-protective-equipment-ppe-dataset)** — Kaggle, by [ndomalau](https://www.kaggle.com/ndomalau)

A YOLO-format object detection dataset built for workplace safety and PPE compliance monitoring. It contains **4,060 images** of people in industrial/construction-style settings, annotated for the presence and absence of key safety gear rather than just the gear itself — which is what makes the compliance-status logic in this project possible directly from the raw labels.

| | |
|---|---|
| **Images** | 4,060 |
| **Format** | YOLO (`class x_center y_center width height`, normalized `[0,1]`) |
| **Classes** | `helmet`, `no_helmet`, `vest`, `no_vest`, `person` |
| **Splits** | `train / val / test` |
| **Source** | Kaggle |
| **Task** | Object detection / PPE compliance monitoring |

**Why this dataset:** most public helmet/vest datasets only label the equipment itself (`helmet`, `vest`), which means "not wearing PPE" has to be inferred indirectly (e.g. a `person` box with no gear box inside it). This dataset instead labels the *negative* cases explicitly (`no_helmet`, `no_vest`), giving direct, per-instance ground truth for compliance vs. non-compliance — closer to how a real safety-monitoring system needs to reason.

Download it locally via `kagglehub`:

```python
import kagglehub
path = kagglehub.dataset_download("ndomalau/personal-protective-equipment-ppe-dataset")
```

The `notebooks/` folder contains a preprocessing and validation notebook that:
- Downloads the dataset via `kagglehub`
- Auto-detects the split folders and class list (reads `data.yaml` if present)
- Verifies image/label pairing
- Validates label file format (class range, normalized coordinates)
- Checks for degenerate (zero width/height) boxes
- Plots class distribution per split
- Visualizes sample images with bounding boxes and PPE compliance status




## 🏋️ Training

### YOLOv8 / YOLOv11 (Ultralytics)

```bash
yolo detect train data=data.yaml model=yolov8n.pt epochs=100 imgsz=640
yolo detect train data=data.yaml model=yolo11n.pt epochs=100 imgsz=640
```

### Faster R-CNN

```bash
python src/models/faster_rcnn/train.py \
    --data data.yaml \
    --epochs 50 \
    --batch-size 4
```


---

## 🚀 Running the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

The app lets you:
- Upload an image 
- Choose which model to run inference with (Faster R-CNN / YOLOv8 / YOLOv11)
- View detected helmet/vest boxes and the overall compliance status per person

---

## 📦 Model Weights

Trained weights aren't committed to the repository due to size. Download them from:

- `weights/faster_rcnn.pth` — [link to release / cloud storage]
- `weights/yolov8_best.pt` — [link to release / cloud storage]
- `weights/yolo11_best.pt` — [link to release / cloud storage]

Place them in the `weights/` folder before running training resumes or the Streamlit app.

---



## 🙏 Acknowledgments

- Dataset: [Personal Protective Equipment (PPE) Dataset — Kaggle](https://www.kaggle.com/datasets/ndomalau/personal-protective-equipment-ppe-dataset)
- [Ultralytics YOLOv8 / YOLOv11](https://github.com/ultralytics/ultralytics)
- [Torchvision Faster R-CNN](https://pytorch.org/vision/stable/models.html#object-detection)
- [Streamlit](https://streamlit.io/)
