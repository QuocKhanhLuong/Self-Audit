import sys
import os
import time
import json
import platform
import psutil
import torch
import torch.nn as nn
import numpy as np

# Ensure src is in python path
sys.path.insert(0, os.path.abspath("src"))

from self_audit_pseudolabel.system_v3 import (
    AdaptiveAnnotationStudent,
    CinePseudoTeacher,
    PROFILES,
    NUM_CLASSES,
)
import thop

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def detailed_param_breakdown(student):
    breakdown = {}
    
    # Total
    tot, trn = count_parameters(student)
    breakdown["total"] = tot
    
    # Encoder
    enc_tot, _ = count_parameters(student.encoder)
    breakdown["encoder"] = {
        "total": enc_tot,
        "e0": count_parameters(student.encoder.net[0])[0],
        "e1": count_parameters(student.encoder.net[1])[0],
        "e2": count_parameters(student.encoder.net[2])[0],
    }
    
    # A0 head
    a0_tot, _ = count_parameters(student.a0_head)
    breakdown["a0_head"] = a0_tot
    
    # Refiner (AnnotationExpert)
    ref_tot, _ = count_parameters(student.refiner)
    ref_breakdown = {
        "total": ref_tot,
        "input_projection": count_parameters(student.refiner.input_projection)[0],
        "refinement_block": {
            "total": count_parameters(student.refiner.refinement_block)[0],
            "generator": count_parameters(student.refiner.refinement_block.generator)[0],
            "generator_turn_embed": count_parameters(student.refiner.refinement_block.generator.turn_embedding)[0],
            "generator_iter_embed": count_parameters(student.refiner.refinement_block.generator.iteration_embedding)[0],
            "generator_state_net": count_parameters(student.refiner.refinement_block.generator.state_net)[0],
            "generator_param_head": count_parameters(student.refiner.refinement_block.generator.parameter_head)[0],
            "query": count_parameters(student.refiner.refinement_block.query)[0],
            "key": count_parameters(student.refiner.refinement_block.key)[0],
            "value": count_parameters(student.refiner.refinement_block.value)[0],
            "output": count_parameters(student.refiner.refinement_block.output)[0],
        },
        "refinement_residual": count_parameters(student.refiner.refinement_residual)[0],
        "delta_head": count_parameters(student.refiner.delta_head)[0],
        "gate_head": count_parameters(student.refiner.gate_head)[0],
    }
    breakdown["refiner"] = ref_breakdown
    
    # Compact active vs unused
    breakdown["compact_active_params"] = enc_tot + a0_tot
    breakdown["compact_inactive_params"] = ref_tot
    breakdown["compact_active_fraction"] = (enc_tot + a0_tot) / tot
    
    return breakdown

def measure_flops(student, resolutions=[128, 224]):
    results = {}
    for res in resolutions:
        results[res] = {}
        x = torch.randn(1, 3, res, res)
        for profile in ["compact", "balanced", "accurate"]:
            # Custom forward wrapper for thop
            class ProfileWrapper(nn.Module):
                def __init__(self, model, prof):
                    super().__init__()
                    self.m = model
                    self.p = prof
                def forward(self, x):
                    return self.m(x, profile=self.p)
            
            wrapper = ProfileWrapper(student, profile)
            wrapper.eval()
            with torch.no_grad():
                macs, params = thop.profile(wrapper, inputs=(x,), verbose=False)
            results[res][profile] = {
                "macs": macs,
                "flops": macs * 2, # standard convention: 1 MAC = 2 FLOPs
                "params": params
            }
    return results

def benchmark_latency(student, resolutions=[128, 224], repeats=30, warmups=10, num_threads=4):
    torch.set_num_threads(num_threads)
    results = {}
    process = psutil.Process()
    
    for res in resolutions:
        results[res] = {}
        for profile in ["compact", "balanced", "accurate"]:
            # Cold forward: fresh input, first call
            # Measure RSS before
            rss_before = process.memory_info().rss / (1024 * 1024)
            
            cold_x = torch.randn(1, 3, res, res)
            t0 = time.perf_counter()
            _ = student(cold_x, profile=profile)
            cold_latency_ms = (time.perf_counter() - t0) * 1000.0
            
            # Warmup
            for _ in range(warmups):
                wx = torch.randn(1, 3, res, res)
                _ = student(wx, profile=profile)
            
            # Timed repeats
            timings = []
            for _ in range(repeats):
                rx = torch.randn(1, 3, res, res)
                t_start = time.perf_counter()
                _ = student(rx, profile=profile)
                t_end = time.perf_counter()
                timings.append((t_end - t_start) * 1000.0)
            
            rss_after = process.memory_info().rss / (1024 * 1024)
            
            timings = np.array(timings)
            results[res][profile] = {
                "cold_ms": float(cold_latency_ms),
                "mean_ms": float(np.mean(timings)),
                "std_ms": float(np.std(timings)),
                "p50_ms": float(np.percentile(timings, 50)),
                "p90_ms": float(np.percentile(timings, 90)),
                "p95_ms": float(np.percentile(timings, 95)),
                "min_ms": float(np.min(timings)),
                "max_ms": float(np.max(timings)),
                "repeats": repeats,
                "warmups": warmups,
                "rss_before_mb": float(rss_before),
                "rss_after_mb": float(rss_after),
            }
    return results

def get_system_info():
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "total_ram_gb": psutil.virtual_memory().total / (1024**3),
        "available_ram_gb": psutil.virtual_memory().available / (1024**3),
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
        "active_processes": len(psutil.pids()),
    }
    return info

def count_dw_and_encoder_calls():
    trace = {}
    for profile, prof_obj in PROFILES.items():
        turns = prof_obj.turns
        enc_calls = 1
        a0_calls = 1
        refiner_calls = turns
        dw_calls = 0
        depths_per_turn = []
        for t in range(turns):
            depth = min(t + 1, 3)
            depth = max(depth, 1)
            depths_per_turn.append(depth)
            dw_calls += depth
        trace[profile] = {
            "turns": turns,
            "encoder_calls": enc_calls,
            "a0_calls": a0_calls,
            "refiner_calls": refiner_calls,
            "depths_per_turn": depths_per_turn,
            "total_dw_attention_calls": dw_calls,
            "total_grid_sample_calls": dw_calls * 2, # key and value
        }
    return trace

if __name__ == "__main__":
    print("=== Starting Resource & Adaptation Audit ===")
    student = AdaptiveAnnotationStudent(width=32, window_k=8).eval()
    teacher = CinePseudoTeacher(width=24, appearance_dim=48, motion_dim=16, fused_dim=64, k=12).eval()
    
    teacher_params, _ = count_parameters(teacher)
    student_params, _ = count_parameters(student)
    print(f"Teacher total params: {teacher_params:,}")
    print(f"Student total params: {student_params:,}")
    
    breakdown = detailed_param_breakdown(student)
    call_trace = count_dw_and_encoder_calls()
    flops = measure_flops(student, [128, 224])
    print("FLOPs measured.")
    
    sys_info = get_system_info()
    print("System info:", sys_info)
    
    print("Benchmarking latency on CPU (threads=4)...")
    cpu_latency = benchmark_latency(student, resolutions=[128, 224], repeats=30, warmups=10, num_threads=4)
    print("CPU latency benchmark completed.")
    
    # Check MPS latency if available
    mps_latency = None
    if torch.backends.mps.is_available():
        print("Benchmarking MPS latency...")
        try:
            student_mps = student.to("mps")
            mps_results = {}
            for res in [128, 224]:
                mps_results[res] = {}
                for profile in ["compact", "balanced", "accurate"]:
                    # Warmup
                    for _ in range(10):
                        wx = torch.randn(1, 3, res, res, device="mps")
                        _ = student_mps(wx, profile=profile)
                        torch.mps.synchronize()
                    timings = []
                    for _ in range(30):
                        rx = torch.randn(1, 3, res, res, device="mps")
                        torch.mps.synchronize()
                        t0 = time.perf_counter()
                        _ = student_mps(rx, profile=profile)
                        torch.mps.synchronize()
                        t1 = time.perf_counter()
                        timings.append((t1 - t0) * 1000.0)
                    timings = np.array(timings)
                    mps_results[res][profile] = {
                        "mean_ms": float(np.mean(timings)),
                        "p50_ms": float(np.percentile(timings, 50)),
                        "p90_ms": float(np.percentile(timings, 90)),
                        "p95_ms": float(np.percentile(timings, 95)),
                        "min_ms": float(np.min(timings)),
                        "max_ms": float(np.max(timings)),
                    }
            mps_latency = mps_results
            print("MPS latency benchmark completed.")
        except Exception as e:
            print("MPS benchmark failed:", e)
            mps_latency = {"error": str(e)}

    data = {
        "teacher_params": teacher_params,
        "student_params": student_params,
        "param_breakdown": breakdown,
        "call_trace": call_trace,
        "flops_macs": flops,
        "cpu_latency": cpu_latency,
        "mps_latency": mps_latency,
        "system_info": sys_info,
    }
    
    out_path = "reports/astra_v3/workers/F_evidence/resource_audit_data.json"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Audit data written to {out_path}")
