import os
import time
import torch
import numpy as np
from ultralytics import YOLO

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_WEIGHTS = "best8n.pt"
DATA_YAML = "data.yaml"
OUTPUT_DIR = "evaluation8n_output"
EVAL_SPLIT = "val"        # Uses the 'val' path in data.yaml (set to test/images)
IMG_SIZE = 640
BENCHMARK_ITERATIONS = 100
# ==========================================

def run_evaluation():
    print("=" * 60)
    print(" 1. LOADING MODEL & COMPUTING STATIC COMPLEXITY")
    print("=" * 60)
    
    if not os.path.exists(MODEL_WEIGHTS):
        raise FileNotFoundError(f"Weights file '{MODEL_WEIGHTS}' not found in current folder.")
    if not os.path.exists(DATA_YAML):
        raise FileNotFoundError(f"Configuration file '{DATA_YAML}' not found in current folder.")

    model = YOLO(MODEL_WEIGHTS)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Target Execution Device: {device.upper()}")

    # 1. Parameter and FLOPs extraction
    # model.info() computes parameters and FLOPs directly
    model_info = model.info(detailed=False, verbose=False)
    
    total_params = sum(p.numel() for p in model.model.parameters())
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)

    print(f"\n[Model Architecture Overview]")
    print(f" - Layers:           {len(list(model.model.modules()))}")
    print(f" - Total Parameters: {total_params:,} ({total_params / 1e6:.2f} M)")
    print(f" - Trainable Params: {trainable_params:,}")

    # ==========================================
    # 2. DATASET EVALUATION & PLOT GENERATION
    # ==========================================
    print("\n" + "=" * 60)
    print(" 2. RUNNING TEST SPLIT VALIDATION & GENERATING PLOTS")
    print("=" * 60)
    
    metrics = model.val(
        data=DATA_YAML,
        split=EVAL_SPLIT,
        imgsz=IMG_SIZE,
        batch=16,
        conf=0.25,
        iou=0.6,
        device=device,
        plots=True,                 # Forces generation of confusion matrix, PR curves, etc.
        save_json=False,
        project=OUTPUT_DIR,
        name="test_metrics",
        exist_ok=True
    )

    # ==========================================
    # 3. LATENCY & FPS BENCHMARKING (REAL-TIME CLAIMS)
    # ==========================================
    print("\n" + "=" * 60)
    print(" 3. BENCHMARKING SPEED, LATENCY & FPS")
    print("=" * 60)

    # Standard dummy tensor for timing
    dummy_input = torch.zeros((1, 3, IMG_SIZE, IMG_SIZE), device=device)

    # Warm-up (crucial to prime CUDA kernels and cache)
    print(f"Warming up inference engine on {device.upper()}...")
    for _ in range(20):
        _ = model.predict(dummy_input, verbose=False)

    # Benchmark Loop
    latencies = []
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    print(f"Benchmarking {BENCHMARK_ITERATIONS} iterations...")
    for _ in range(BENCHMARK_ITERATIONS):
        start_time = time.perf_counter()
        _ = model.predict(dummy_input, verbose=False)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000.0)  # Convert to ms

    avg_latency_ms = np.mean(latencies)
    std_latency_ms = np.std(latencies)
    fps = 1000.0 / avg_latency_ms

    # Validation pipeline speed breakdowns recorded by Ultralytics
    # metrics.speed is a dict: {'preprocess': ms, 'inference': ms, 'loss': ms, 'postprocess': ms}
    val_preprocess = metrics.speed.get('preprocess', 0.0)
    val_inference = metrics.speed.get('inference', 0.0)
    val_postprocess = metrics.speed.get('postprocess', 0.0)
    total_pipeline_ms = val_preprocess + val_inference + val_postprocess
    pipeline_fps = 1000.0 / total_pipeline_ms if total_pipeline_ms > 0 else 0.0

    # ==========================================
    # 4. PRINT FORMATTED AUDIT REPORT
    # ==========================================
    print("\n" + "=" * 60)
    print("               FINAL EVALUATION & PERFORMANCE REPORT")
    print("=" * 60)
    
    print("\n[Accuracy Metrics]")
    print(f" - Overall mAP@0.5:      {metrics.box.map50 * 100:.2f} %")
    print(f" - Overall mAP@0.5:0.95: {metrics.box.map * 100:.2f} %")
    print(f" - Mean Precision (P):   {metrics.box.mp * 100:.2f} %")
    print(f" - Mean Recall (R):      {metrics.box.mr * 100:.2f} %")

    print("\n[Per-Class Performance Breakdown]")
    class_names = [model.names[i] for i in range(len(model.names))]
    print(f"{'Class Name':<15} | {'P (%)':<8} | {'R (%)':<8} | {'mAP@0.5 (%)':<12} | {'mAP@0.5:0.95 (%)':<16}")
    print("-" * 70)
    
    for idx, c_name in enumerate(class_names):
        p = metrics.box.p[idx] * 100 if len(metrics.box.p) > idx else 0.0
        r = metrics.box.r[idx] * 100 if len(metrics.box.r) > idx else 0.0
        m50 = metrics.box.all_ap[idx, 0] * 100 if metrics.box.all_ap.shape[1] > 0 else 0.0
        m95 = metrics.box.ap[idx] * 100 if len(metrics.box.ap) > idx else 0.0
        print(f"{c_name:<15} | {p:<8.2f} | {r:<8.2f} | {m50:<12.2f} | {m95:<16.2f}")

    print("\n[Latency, Speed & Real-Time Performance]")
    print(f" - Isolated Pure Inference:  {avg_latency_ms:.2f} ± {std_latency_ms:.2f} ms per image")
    print(f" - Isolated Inference FPS:   {fps:.2f} FPS")
    print(f" - Full Pipeline Latency:    {total_pipeline_ms:.2f} ms")
    print(f"   (Preprocess: {val_preprocess:.2f}ms | Inference: {val_inference:.2f}ms | Postprocess: {val_postprocess:.2f}ms)")
    print(f" - Full Pipeline Throughput: {pipeline_fps:.2f} FPS")
    
    is_realtime = "YES (Real-Time Ready >= 30 FPS)" if pipeline_fps >= 30.0 else "EDGE/SEMI REAL-TIME (< 30 FPS)"
    print(f" - Real-Time Claim Status:   {is_realtime}")

    print("\n" + "=" * 60)
    print(" 5. GENERATED VISUALIZATION PLOTS LOCATION")
    print("=" * 60)
    save_path = os.path.join(OUTPUT_DIR, "test_metrics")
    print(f"All evaluation plots are saved inside: {os.path.abspath(save_path)}")
    print("\nKey files to copy into your slides:")
    print(" - confusion_matrix.png            (Raw confusion counts)")
    print(" - confusion_matrix_normalized.png (Normalized classification accuracy)")
    print(" - BoxPR_curve.png                 (Precision-Recall trade-off)")
    print(" - BoxF1_curve.png                 (Optimal confidence threshold curve)")
    print(" - results.png                     (Training/validation loss history)")
    print(" - val_batch0_pred.jpg             (Ground truth vs predicted bounding boxes)")
    print("=" * 60)

if __name__ == "__main__":
    run_evaluation()