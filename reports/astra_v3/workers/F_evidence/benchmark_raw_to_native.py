import sys
import os
import time
import json
import psutil
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.abspath("src"))
from self_audit_pseudolabel.system_v3 import AdaptiveAnnotationStudent

def benchmark_raw_to_native(student, raw_shapes=[(256, 256), (216, 256), (320, 320)], model_res=224, repeats=30, warmups=10, num_threads=4):
    torch.set_num_threads(num_threads)
    results = {}
    
    for raw_h, raw_w in raw_shapes:
        shape_key = f"{raw_h}x{raw_w}_to_{model_res}x{model_res}"
        results[shape_key] = {}
        
        # Simulate raw 2.5D slice: 3 numpy float32 channels
        np.random.seed(42)
        raw_slice = np.random.randn(3, raw_h, raw_w).astype(np.float32) * 50.0 + 100.0
        
        for profile in ["compact", "balanced", "accurate"]:
            # Warmup
            for _ in range(warmups):
                # 1. Preprocessing: intensity norm + to torch + resize to model_res
                t_raw = torch.from_numpy(raw_slice)
                t_norm = (t_raw - t_raw.mean()) / (t_raw.std() + 1e-6)
                t_in = F.interpolate(t_norm.unsqueeze(0), size=(model_res, model_res), mode="bilinear", align_corners=False)
                # 2. Forward
                out = student(t_in, profile=profile)
                logits = out["final_logits"]
                # 3. Postprocessing: resize back to raw shape + argmax + numpy
                out_native = F.interpolate(logits, size=(raw_h, raw_w), mode="bilinear", align_corners=False)
                mask = out_native.argmax(dim=1).squeeze(0).byte().numpy()
            
            pre_times = []
            fwd_times = []
            post_times = []
            total_times = []
            
            for _ in range(repeats):
                t0 = time.perf_counter()
                
                # Preprocessing
                t_raw = torch.from_numpy(raw_slice)
                t_norm = (t_raw - t_raw.mean()) / (t_raw.std() + 1e-6)
                t_in = F.interpolate(t_norm.unsqueeze(0), size=(model_res, model_res), mode="bilinear", align_corners=False)
                t1 = time.perf_counter()
                
                # Pure forward
                out = student(t_in, profile=profile)
                logits = out["final_logits"]
                t2 = time.perf_counter()
                
                # Postprocessing
                out_native = F.interpolate(logits, size=(raw_h, raw_w), mode="bilinear", align_corners=False)
                mask = out_native.argmax(dim=1).squeeze(0).byte().numpy()
                t3 = time.perf_counter()
                
                pre_times.append((t1 - t0) * 1000.0)
                fwd_times.append((t2 - t1) * 1000.0)
                post_times.append((t3 - t2) * 1000.0)
                total_times.append((t3 - t0) * 1000.0)
                
            results[shape_key][profile] = {
                "pre_ms": {
                    "mean": float(np.mean(pre_times)),
                    "p50": float(np.percentile(pre_times, 50)),
                    "p95": float(np.percentile(pre_times, 95)),
                },
                "forward_ms": {
                    "mean": float(np.mean(fwd_times)),
                    "p50": float(np.percentile(fwd_times, 50)),
                    "p95": float(np.percentile(fwd_times, 95)),
                },
                "post_ms": {
                    "mean": float(np.mean(post_times)),
                    "p50": float(np.percentile(post_times, 50)),
                    "p95": float(np.percentile(post_times, 95)),
                },
                "total_e2e_ms": {
                    "mean": float(np.mean(total_times)),
                    "p50": float(np.percentile(total_times, 50)),
                    "p95": float(np.percentile(total_times, 95)),
                },
                "forward_fraction_of_total": float(np.mean(fwd_times) / np.mean(total_times)),
                "pre_post_overhead_fraction": float((np.mean(pre_times) + np.mean(post_times)) / np.mean(total_times)),
            }
    return results

if __name__ == "__main__":
    student = AdaptiveAnnotationStudent(width=32, window_k=8).eval()
    res = benchmark_raw_to_native(student)
    out_path = "reports/astra_v3/workers/F_evidence/raw_to_native_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Raw-to-native benchmark written to {out_path}")
