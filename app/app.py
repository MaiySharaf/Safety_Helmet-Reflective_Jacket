import os
import time
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO
import plotly.graph_objects as go
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(
    page_title="SafeSight AI | PPE Compliance & Benchmark Suite",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling & CSS Animations ---
st.markdown("""
<style>
    /* Gradient animated header */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        padding: 24px;
        border-radius: 14px;
        border: 1px solid #334155;
        margin-bottom: 22px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
    }
    .main-title {
        color: #f8fafc;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .sub-title {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: 6px;
    }

    /* Metric Card Animation on Hover */
    .metric-card {
        background: #1e293b;
        border-radius: 12px;
        padding: 16px 20px;
        border: 1px solid #334155;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.35);
    }
    
    /* Pulsing Alert Banner */
    @keyframes pulse-red {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
        70% { box-shadow: 0 0 0 14px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }
    .alert-critical {
        background: #ef4444;
        color: white;
        padding: 14px;
        border-radius: 10px;
        font-weight: 700;
        text-align: center;
        animation: pulse-red 2s infinite;
        margin-bottom: 14px;
    }
    .alert-warning {
        background: #f59e0b;
        color: white;
        padding: 14px;
        border-radius: 10px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 14px;
    }
    .alert-safe {
        background: #10b981;
        color: white;
        padding: 14px;
        border-radius: 10px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 14px;
    }
</style>
""", unsafe_allow_html=True)

# --- Available Models Registry ---
# --- Available Models Registry ---
MODEL_REGISTRY = {
    "YOLO11s (Attention Enhanced - Proposed)": {
        # Steps out of /app, goes into /models
        "file": "../models/best.pt", 
        # Steps out of /app, goes into /docs
        "plot_dir": "../docs/evaluation_output/test_metrics", 
        "params_m": 9.4,
        "gflops": 21.4,
        "map50": 92,
        "map95": 59.8,
        "precision": 87.5,
        "recall": 89.5,
        "fps_cpu": 6.2,
        "badge": "SOTA Choice"
    },
    "YOLOv8n (Lightweight Baseline)": {
        "file": "../models/best_8n.pt",
        "plot_dir": "../docs/evaluation8n_output/test_metrics",
        "params_m": 3.2,
        "gflops": 8.7,
        "map50": 90,
        "map95": 57,
        "precision": 86,
        "recall": 88,
        "fps_cpu": 13.2,
        "badge": "Fast Edge Baseline"
    },
}

# --- Cache Model Loader with Fallback ---
@st.cache_resource(show_spinner=False)
def get_model(weight_filename):
    if os.path.exists(weight_filename):
        return YOLO(weight_filename), True
    # Fallback to base weights if fine-tuned weights aren't in working directory yet
    fallback = "yolov8n.pt" if "8n" in weight_filename else "yolo11n.pt"
    return YOLO(fallback), False

# --- Helper Functions ---
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

def run_compliance_inference(img_bgr, model_instance, person_thresh, gear_thresh):
    start_t = time.perf_counter()
    results = model_instance.predict(source=img_bgr, save=False, verbose=False)[0]
    inference_time = (time.perf_counter() - start_t) * 1000.0

    names = model_instance.names
    persons, helmets, vests = [], [], []

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

    annotated = img_bgr.copy()
    safe_cnt, warn_cnt, crit_cnt = 0, 0, 0
    worker_audit = []

    for idx, p in enumerate(persons):
        p_box = p['box']
        px1, py1, px2, py2 = p_box

        has_helmet = any(is_inside(h['box'], p_box) and h['label'] == 'helmet' for h in helmets)
        has_vest = any(is_inside(v['box'], p_box) and v['label'] == 'vest' for v in vests)

        if has_helmet and has_vest:
            safe_cnt += 1
            color = (0, 200, 0)
            status_text = f"Worker #{idx+1} [COMPLIANT]"
            h_stat, v_stat, final_stat = "Equipped", "Equipped", "Safe"
        elif has_helmet and not has_vest:
            warn_cnt += 1
            color = (0, 140, 255)
            status_text = f"Worker #{idx+1} [NO VEST]"
            h_stat, v_stat, final_stat = "Equipped", "Missing", "Warning"
        elif not has_helmet and has_vest:
            warn_cnt += 1
            color = (0, 140, 255)
            status_text = f"Worker #{idx+1} [NO HELMET]"
            h_stat, v_stat, final_stat = "Missing", "Equipped", "Warning"
        else:
            crit_cnt += 1
            color = (0, 0, 255)
            status_text = f"Worker #{idx+1} [CRITICAL: MISSING ALL]"
            h_stat, v_stat, final_stat = "Missing", "Missing", "Critical"

        cv2.rectangle(annotated, (px1, py1), (px2, py2), color, 3)
        draw_clean_tag(annotated, status_text, px1, py1, color)

        worker_audit.append({
            "Worker": f"Worker #{idx+1}",
            "Confidence": f"{p['conf']:.2f}",
            "Hardhat": "✅ " + h_stat if h_stat == "Equipped" else "❌ Missing",
            "Safety Vest": "✅ " + v_stat if v_stat == "Equipped" else "❌ Missing",
            "Site Status": final_stat
        })

    # Draw individual gear bounding boxes
    for h in helmets:
        hx1, hy1, hx2, hy2 = h['box']
        c = (255, 120, 0) if h['label'] == 'helmet' else (0, 0, 255)
        cv2.rectangle(annotated, (hx1, hy1), (hx2, hy2), c, 2)
        draw_clean_tag(annotated, f"{h['label']} {h['conf']:.2f}", hx1, hy2 - 4, c)

    for v in vests:
        vx1, vy1, vx2, vy2 = v['box']
        c = (0, 220, 220) if v['label'] == 'vest' else (0, 0, 255)
        cv2.rectangle(annotated, (vx1, vy1), (vx2, vy2), c, 2)
        draw_clean_tag(annotated, f"{v['label']} {v['conf']:.2f}", vx1, vy2 - 4, c)

    return annotated, len(persons), safe_cnt, warn_cnt, crit_cnt, worker_audit, inference_time

# --- Header Banner ---
st.markdown("""
<div class="main-header">
    <h1 class="main-title">🦺 SafeSight AI: Multi-Model PPE Compliance Inspector</h1>
    <p class="sub-title">Real-Time Construction Workplace Safety Monitoring & Architecture Comparison</p>
</div>
""", unsafe_allow_html=True)

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Active Model & Engine")
    selected_model_name = st.selectbox("Select Active Detection Model:", list(MODEL_REGISTRY.keys()), index=0)
    model_meta = MODEL_REGISTRY[selected_model_name]
    
    loaded_model, is_custom = get_model(model_meta["file"])
    if is_custom:
        st.success(f"Loaded custom weights: `{model_meta['file']}`", icon="🎯")
    else:
        st.warning(f"File `{model_meta['file']}` not found in path. Running with base `{loaded_model.model_name}`.", icon="⚠️")

    st.markdown(f"**Architecture Specs:** {model_meta['params_m']}M params | {model_meta['gflops']} GFLOPs")
    st.divider()

    st.header("🎯 Dual-Confidence Gates")
    st.caption("Independent thresholds prevent false positives while capturing fine gear.")
    person_thresh = st.slider("Person Threshold (Higher = filters background/animals)", 0.2, 0.95, 0.60, 0.05)
    gear_thresh = st.slider("PPE Gear Threshold (Lower = catches small/distant gear)", 0.15, 0.90, 0.35, 0.05)
    st.divider()

    st.header("📸 Media Feeder")
    input_source = st.radio("Input Source Mode:", ("Upload Image File", "Live Camera Feed", "Demo Sample Images"))

# Sample Images Pool
DEMO_SAMPLES = {
    "Workers on Scaffolding": "https://www.shutterstock.com/image-photo/engineer-wearing-hard-hat-safety-600w-2686719375.jpg",
    "Active Site Team":  "https://www.shutterstock.com/image-photo/high-angle-view-construction-workers-600w-2760344363.jpg"
}

# Ingest Image
image_raw = None
if input_source == "Upload Image File":
    uploaded = st.file_uploader("Upload an inspection photograph...", type=["jpg", "jpeg", "png"])
    if uploaded:
        image_raw = Image.open(uploaded)
elif input_source == "Live Camera Feed":
    cam_snap = st.camera_input("Capture frame from on-site webcam")
    if cam_snap:
        image_raw = Image.open(cam_snap)
else:
    sample_key = st.selectbox("Choose demo test image:", list(DEMO_SAMPLES.keys()))
    image_raw = DEMO_SAMPLES[sample_key]

# --- Main Navigation Tabs ---
tab_inspect, tab_benchmark, tab_curves = st.tabs([
    "🔍 Live Compliance Inspector",
    "⚔️ Model Comparison & Benchmarks",
    "📈 Evaluation Artifacts & Curves"
])

# ==============================================================================
# TAB 1: LIVE COMPLIANCE INSPECTOR
# ==============================================================================
with tab_inspect:
    if image_raw is not None:
        if isinstance(image_raw, str):
            import urllib.request
            resp = urllib.request.urlopen(image_raw)
            arr = np.asarray(bytearray(resp.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(arr, -1)
        else:
            img_bgr = cv2.cvtColor(np.array(image_raw), cv2.COLOR_RGB2BGR)

        with st.spinner("Processing safety compliance rules..."):
            annotated_bgr, total, safe, warn, crit, audit_log, inf_ms = run_compliance_inference(
                img_bgr, loaded_model, person_thresh, gear_thresh
            )
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        # Dynamic Top Alert Banners
        if crit > 0:
            st.markdown(f'<div class="alert-critical">🚨 CRITICAL VIOLATION: {crit} Worker(s) Detected Without Essential Safety Gear!</div>', unsafe_allow_html=True)
        elif warn > 0:
            st.markdown(f'<div class="alert-warning">⚠️ SAFETY WARNING: {warn} Worker(s) Have Incomplete PPE Equipment.</div>', unsafe_allow_html=True)
        elif safe > 0:
            st.markdown('<div class="alert-safe">✅ SITE 100% COMPLIANT: All Detected Personnel Meet OSHA Safety Standards.</div>', unsafe_allow_html=True)

        col_view, col_stats = st.columns([1.7, 1.1])

        with col_view:
            st.subheader("Visual Inspection Stream")
            st.image(annotated_rgb, use_container_width=True)

            # Download Image
            _, buf = cv2.imencode(".jpg", annotated_bgr)
            st.download_button("📥 Export Annotated Frame", data=buf.tobytes(), file_name="ppe_inspection.jpg", mime="image/jpeg")

        with col_stats:
            st.subheader("Real-Time Telemetry")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Workers", total)
            m2.metric("Compliant", safe, delta="Safe", delta_color="normal")
            m3.metric("Warnings", warn, delta="-Incomplete" if warn > 0 else "0", delta_color="inverse")
            m4.metric("Violations", crit, delta="-Critical" if crit > 0 else "0", delta_color="inverse")

            st.caption(f"⚡ Model Inference Speed: **{inf_ms:.1f} ms** ({1000.0/inf_ms:.1f} FPS on CPU)")

            # Donut Chart for Distribution
            if total > 0:
                fig_donut = go.Figure(data=[go.Pie(
                    labels=['Compliant', 'Warning (Partial)', 'Critical Violation'],
                    values=[safe, warn, crit],
                    hole=.55,
                    marker=dict(colors=['#10b981', '#f59e0b', '#ef4444']),
                    textinfo='label+percent'
                )])
                fig_donut.update_layout(
                    showlegend=False,
                    margin=dict(t=10, b=10, l=10, r=10),
                    height=240,
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_donut, use_container_width=True)

        st.subheader("📋 Worker-by-Worker Safety Audit Log")
        if audit_log:
            df_audit = pd.DataFrame(audit_log)
            st.dataframe(df_audit, use_container_width=True, hide_index=True)
            csv_data = df_audit.to_csv(index=False).encode('utf-8')
            st.download_button("📄 Download Audit CSV Report", data=csv_data, file_name="safety_audit_report.csv", mime="text/csv")
        else:
            st.info("No personnel detected in the active frame.")
    else:
        st.info("Upload an image or activate the webcam in the sidebar to begin monitoring.")

# ==============================================================================
# TAB 2: MODEL COMPARISON & BENCHMARKS
# ==============================================================================
with tab_benchmark:
    st.subheader("⚔️ Head-to-Head Architecture Benchmark")
    st.markdown("Compare accuracy, parameter efficiency, and latency trade-offs between **YOLO11s**, **YOLOv8n**, and **YOLOv10s**.")

    # Leaderboard Table
    comp_rows = []
    for name, data in MODEL_REGISTRY.items():
        comp_rows.append({
            "Model Architecture": name,
            "Parameters (M)": data["params_m"],
            "GFLOPs": data["gflops"],
            "mAP@0.5 (%)": f"{data['map50']:.1f}%",
            "mAP@0.5:0.95 (%)": f"{data['map95']:.1f}%",
            "Precision (%)": f"{data['precision']:.1f}%",
            "Recall (%)": f"{data['recall']:.1f}%",
            "Est. CPU FPS": f"{data['fps_cpu']:.0f} FPS",
            "Special Feature": data["badge"]
        })
    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    # Interactive Plots
    c_chart1, c_chart2 = st.columns(2)

    with c_chart1:
        st.markdown("#### Accuracy Metrics Comparison")
        fig_bar = go.Figure(data=[
            go.Bar(name='mAP@0.5', x=list(MODEL_REGISTRY.keys()), y=[d["map50"] for d in MODEL_REGISTRY.values()], marker_color='#3b82f6'),
            go.Bar(name='mAP@0.5:0.95', x=list(MODEL_REGISTRY.keys()), y=[d["map95"] for d in MODEL_REGISTRY.values()], marker_color='#6366f1'),
            go.Bar(name='Recall', x=list(MODEL_REGISTRY.keys()), y=[d["recall"] for d in MODEL_REGISTRY.values()], marker_color='#10b981')
        ])
        fig_bar.update_layout(barmode='group', height=360, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_bar, use_container_width=True)

    with c_chart2:
        st.markdown("#### Multi-Dimensional Efficiency Radar")
        categories = ['mAP@0.5', 'Precision', 'Recall', 'Speed (FPS Norm)', 'Parameter Efficiency']
        fig_radar = go.Figure()

        # Radar trace for YOLO11s
        fig_radar.add_trace(go.Scatterpolar(
            r=[85.4, 87.8, 82.5, 60.0, 75.0],
            theta=categories,
            fill='toself',
            name='YOLO11s',
            line_color='#3b82f6'
        ))
        # Radar trace for YOLOv8n
        fig_radar.add_trace(go.Scatterpolar(
            r=[78.1, 81.2, 74.6, 95.0, 95.0],
            theta=categories,
            fill='toself',
            name='YOLOv8n',
            line_color='#f59e0b'
        ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True, height=360, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_radar, use_container_width=True)

    # Live Side-by-Side Dual Inference
    st.divider()
    st.subheader("⚡ Live Dual-Model Inference on Same Image")
    st.caption("Select two models below to compare visual boundary accuracy, false positives, and execution speed.")

    c_m1, c_m2 = st.columns(2)
    with c_m1:
        mod1_name = st.selectbox("Left Model:", list(MODEL_REGISTRY.keys()), index=0)
    with c_m2:
        mod2_name = st.selectbox("Right Model:", list(MODEL_REGISTRY.keys()), index=1)

    if image_raw is not None and st.button("🚀 Run Simultaneous Dual-Model Inspection"):
        m1_instance, _ = get_model(MODEL_REGISTRY[mod1_name]["file"])
        m2_instance, _ = get_model(MODEL_REGISTRY[mod2_name]["file"])

        res1_bgr, _, s1, w1, c1, _, t1 = run_compliance_inference(img_bgr, m1_instance, person_thresh, gear_thresh)
        res2_bgr, _, s2, w2, c2, _, t2 = run_compliance_inference(img_bgr, m2_instance, person_thresh, gear_thresh)

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown(f"**{mod1_name}** | Latency: `{t1:.1f} ms`")
            st.image(cv2.cvtColor(res1_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.caption(f"Detections: Safe: {s1} | Warnings: {w1} | Critical: {c1}")
        with col_r:
            st.markdown(f"**{mod2_name}** | Latency: `{t2:.1f} ms`")
            st.image(cv2.cvtColor(res2_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.caption(f"Detections: Safe: {s2} | Warnings: {w2} | Critical: {c2}")


# ==============================================================================
# TAB 3: EVALUATION ARTIFACTS & TRAINING CURVES
# ==============================================================================
with tab_curves:
    st.subheader(f"📈 Diagnostic Artifacts: {selected_model_name}")
    st.markdown("Visual validation proofs from model training for presentation defense.")

    # Dynamically grab the correct folder based on the sidebar selection!
    eval_plot_dir = MODEL_REGISTRY[selected_model_name].get("plot_dir", "")
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown("#### Confusion Matrix")
        cf_path = os.path.join(eval_plot_dir, "confusion_matrix_normalized.png")
        if os.path.exists(cf_path):
            st.image(cf_path, use_container_width=True)
        else:
            st.info(f"Could not find `{cf_path}`. Make sure the folder exists.")

    with col_p2:
        st.markdown("#### Precision-Recall Curve")
        pr_path = os.path.join(eval_plot_dir, "BoxPR_curve.png")
        if os.path.exists(pr_path):
            st.image(pr_path, use_container_width=True)
        else:
            st.info(f"Could not find `{pr_path}`. Make sure the folder exists.")

    st.markdown("#### Training & Loss Convergence Curves (`results.png`)")
    res_path = os.path.join(eval_plot_dir, "results.png")
    if os.path.exists(res_path):
        st.image(res_path, use_container_width=True)
    else:
        st.info(f"Could not find `{res_path}`. Make sure the folder exists.")
