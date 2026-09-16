# backend/main.py
import os
import time
import cv2
import numpy as np
import base64
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torchvision
from torchvision.transforms import functional as F
from ultralytics import YOLO

app = FastAPI(title="SafeSight AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_REGISTRY = {
    "YOLO11s (Attention Enhanced)": {"file": "best.pt", "type": "ultralytics"},
    "YOLOv8n (Lightweight)": {"file": "best8n.pt", "type": "ultralytics"},
    "RetinaNet (Baseline)": {"file": "retinanet_ppe_best.pth", "type": "torchvision"}
}

loaded_models = {}

def get_model(model_name):
    if model_name in loaded_models:
        return loaded_models[model_name]
    
    meta = MODEL_REGISTRY.get(model_name)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    if meta["type"] == "ultralytics":
        model = YOLO(meta["file"]) if os.path.exists(meta["file"]) else YOLO("yolov8n.pt")
        loaded_models[model_name] = (model, meta["type"])
    elif meta["type"] == "torchvision":
        model = torchvision.models.detection.retinanet_resnet50_fpn(weights=None, weights_backbone=None, num_classes=6)
        if os.path.exists(meta["file"]):
            model.load_state_dict(torch.load(meta["file"], map_location=device))
        model.to(device)
        model.eval()
        loaded_models[model_name] = (model, meta["type"])
        
    return loaded_models[model_name]

def is_inside(gear_box, person_box):
    gx_center = (gear_box[0] + gear_box[2]) / 2.0
    gy_center = (gear_box[1] + gear_box[3]) / 2.0
    px1, py1, px2, py2 = person_box
    return (px1 <= gx_center <= px2) and (py1 <= gy_center <= py2)

def draw_clean_tag(img, text, x, y, bg_color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), _ = cv2.getTextSize(text, font, 0.52, 2)
    y1 = max(0, y - h - 10)
    y2 = max(h + 10, y)
    cv2.rectangle(img, (x, y1), (x + w + 12, y2), bg_color, -1)
    cv2.putText(img, text, (x + 6, y2 - 5), font, 0.52, (255, 255, 255), 2, cv2.LINE_AA)

@app.post("/api/inspect")
async def inspect_image(
    file: UploadFile = File(...),
    model_name: str = Form("YOLO11s (Attention Enhanced)"),
    person_thresh: float = Form(0.50),
    gear_thresh: float = Form(0.35)
):
    start_t = time.perf_counter()
    
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    model_instance, model_type = get_model(model_name)
    persons, helmets, vests = [], [], []

    if model_type == "ultralytics":
        results = model_instance.predict(source=img_bgr, save=False, verbose=False)[0]
        names = model_instance.names
        for box in results.boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0].cpu().numpy())
            label = names[int(box.cls[0].cpu().numpy())].lower()
            if label == 'person' and conf >= person_thresh:
                persons.append({'box': xyxy, 'conf': conf})
            elif label in ['helmet', 'no_helmet'] and conf >= gear_thresh:
                helmets.append({'box': xyxy, 'conf': conf, 'label': label})
            elif label in ['vest', 'no_vest'] and conf >= gear_thresh:
                vests.append({'box': xyxy, 'conf': conf, 'label': label})

    elif model_type == "torchvision":
        retina_names = {1: 'helmet', 2: 'no_helmet', 3: 'no_vest', 4: 'person', 5: 'vest'}
        device = next(model_instance.parameters()).device
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = F.to_tensor(img_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            prediction = model_instance(img_tensor)[0]
        for i in range(len(prediction['boxes'])):
            conf = float(prediction['scores'][i].cpu().numpy())
            cls_id = int(prediction['labels'][i].cpu().numpy())
            xyxy = prediction['boxes'][i].cpu().numpy().astype(int)
            if cls_id in retina_names:
                label = retina_names[cls_id]
                if label == 'person' and conf >= person_thresh:
                    persons.append({'box': xyxy, 'conf': conf})
                elif label in ['helmet', 'no_helmet'] and conf >= gear_thresh:
                    helmets.append({'box': xyxy, 'conf': conf, 'label': label})
                elif label in ['vest', 'no_vest'] and conf >= gear_thresh:
                    vests.append({'box': xyxy, 'conf': conf, 'label': label})

    annotated = img_bgr.copy()
    safe_cnt, warn_cnt, crit_cnt = 0, 0, 0
    worker_audit = []

    # 1. Draw Persons and assign status
    for idx, p in enumerate(persons):
        p_box = p['box']
        px1, py1, px2, py2 = p_box
        has_helmet = any(is_inside(h['box'], p_box) and h['label'] == 'helmet' for h in helmets)
        has_vest = any(is_inside(v['box'], p_box) and v['label'] == 'vest' for v in vests)

        if has_helmet and has_vest:
            safe_cnt += 1; color = (0, 200, 0)
            status_text = f"Worker #{idx+1} [COMPLIANT]"
            h_stat, v_stat, final_stat = "Equipped", "Equipped", "Safe"
        elif has_helmet and not has_vest:
            warn_cnt += 1; color = (0, 140, 255)
            status_text = f"Worker #{idx+1} [NO VEST]"
            h_stat, v_stat, final_stat = "Equipped", "Missing", "Warning"
        elif not has_helmet and has_vest:
            warn_cnt += 1; color = (0, 140, 255)
            status_text = f"Worker #{idx+1} [NO HELMET]"
            h_stat, v_stat, final_stat = "Missing", "Equipped", "Warning"
        else:
            crit_cnt += 1; color = (0, 0, 255)
            status_text = f"Worker #{idx+1} [CRITICAL]"
            h_stat, v_stat, final_stat = "Missing", "Missing", "Critical"

        cv2.rectangle(annotated, (px1, py1), (px2, py2), color, 3)
        draw_clean_tag(annotated, status_text, px1, py1, color)
        
        worker_audit.append({
            "id": f"Worker #{idx+1}",
            "conf": f"{p['conf']:.2f}",
            "helmet": "✅ Equipped" if h_stat == "Equipped" else "❌ Missing",
            "vest": "✅ Equipped" if v_stat == "Equipped" else "❌ Missing",
            "status": final_stat
        })

    # 2. Draw Helmets & Violations (FIXED: Added this back)
    for h in helmets:
        hx1, hy1, hx2, hy2 = h['box']
        c = (255, 120, 0) if h['label'] == 'helmet' else (0, 0, 255)
        cv2.rectangle(annotated, (hx1, hy1), (hx2, hy2), c, 2)
        draw_clean_tag(annotated, f"{h['label']} {h['conf']:.2f}", hx1, max(0, hy2 - 4), c)

    # 3. Draw Vests & Violations (FIXED: Added this back)
    for v in vests:
        vx1, vy1, vx2, vy2 = v['box']
        c = (0, 220, 220) if v['label'] == 'vest' else (0, 0, 255)
        cv2.rectangle(annotated, (vx1, vy1), (vx2, vy2), c, 2)
        draw_clean_tag(annotated, f"{v['label']} {v['conf']:.2f}", vx1, max(0, vy2 - 4), c)

    _, buffer = cv2.imencode('.jpg', annotated)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    inf_ms = (time.perf_counter() - start_t) * 1000.0

    return JSONResponse({
        "image": f"data:image/jpeg;base64,{img_base64}",
        "telemetry": {
            "total": len(persons),
            "safe": safe_cnt,
            "warn": warn_cnt,
            "crit": crit_cnt,
            "latency_ms": round(inf_ms, 1)
        },
        "audit": worker_audit
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
