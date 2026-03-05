#!/usr/bin/env python3
"""Benchmark CoreML model prediction latency.

Usage:
    python scripts/bench_coreml.py --dir models/coreml_bench

Tests:
    1. Fused FFN with LUT6 quantization
    2. Fused FFN with FP16 (no quantization)
    3. Simple o_proj matmul with LUT6
"""
import argparse
import os
import time

import numpy as np
import coremltools as ct


def benchmark_model(model_path: str, input_shape: tuple, n_warmup: int = 10, n_iter: int = 100):
    """Load and benchmark a CoreML model."""
    print(f"\n{'=' * 60}")
    print(f"Model: {os.path.basename(model_path)}")
    print(f"{'=' * 60}")

    # Load model
    t0 = time.time()
    model = ct.models.MLModel(model_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    load_time = time.time() - t0
    print(f"  Load time: {load_time:.2f}s")

    # Get input/output names from spec
    spec = model.get_spec()
    input_name = spec.description.input[0].name
    output_name = spec.description.output[0].name
    print(f"  Input: {input_name} {input_shape}")
    print(f"  Output: {output_name}")

    # Create input
    input_data = {input_name: np.random.randn(*input_shape).astype(np.float32)}

    # Warmup
    print(f"  Warming up ({n_warmup} iterations)...")
    for _ in range(n_warmup):
        model.predict(input_data)

    # Benchmark
    print(f"  Benchmarking ({n_iter} iterations)...")
    latencies = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        result = model.predict(input_data)
        latencies.append((time.perf_counter() - t0) * 1000)  # ms

    latencies.sort()
    mean = sum(latencies) / len(latencies)
    p50 = latencies[len(latencies) // 2]
    p99 = latencies[int(len(latencies) * 0.99)]
    p_min = latencies[0]
    p_max = latencies[-1]

    print(f"\n  Results (ms/call):")
    print(f"    Mean:  {mean:.3f}")
    print(f"    P50:   {p50:.3f}")
    print(f"    P99:   {p99:.3f}")
    print(f"    Min:   {p_min:.3f}")
    print(f"    Max:   {p_max:.3f}")

    # Throughput estimate
    calls_per_token_08b = 72  # 24 layers × 3 ops
    calls_per_token_4b = 96   # 32 layers × 3 ops
    est_08b = 1000.0 / (mean * calls_per_token_08b)
    est_4b = 1000.0 / (mean * calls_per_token_4b)
    print(f"\n  Throughput estimate (if all calls at this latency):")
    print(f"    0.8B ({calls_per_token_08b} calls/tok): {est_08b:.1f} tok/s")
    print(f"    4B   ({calls_per_token_4b} calls/tok):  {est_4b:.1f} tok/s")

    return mean, p50, p99


def main():
    parser = argparse.ArgumentParser(description="Benchmark CoreML models")
    parser.add_argument("--dir", required=True, help="Directory containing .mlpackage models")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iter", type=int, default=100, help="Benchmark iterations")
    args = parser.parse_args()

    results = {}

    # FFN LUT6
    ffn_lut6_path = os.path.join(args.dir, "fused_ffn_layer0.mlpackage")
    if os.path.exists(ffn_lut6_path):
        mean, p50, p99 = benchmark_model(ffn_lut6_path, (1, 1024), args.warmup, args.iter)
        results["FFN LUT6"] = mean

    # FFN FP16
    ffn_fp16_path = os.path.join(args.dir, "fused_ffn_layer0_fp16.mlpackage")
    if os.path.exists(ffn_fp16_path):
        mean, p50, p99 = benchmark_model(ffn_fp16_path, (1, 1024), args.warmup, args.iter)
        results["FFN FP16"] = mean

    # O_proj LUT6
    oproj_path = os.path.join(args.dir, "o_proj_layer0.mlpackage")
    if os.path.exists(oproj_path):
        mean, p50, p99 = benchmark_model(oproj_path, (1, 2048), args.warmup, args.iter)
        results["o_proj LUT6"] = mean

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for name, mean in results.items():
        print(f"  {name:15s}: {mean:.3f} ms/call")

    if "FFN LUT6" in results and "FFN FP16" in results:
        ratio = results["FFN LUT6"] / results["FFN FP16"]
        print(f"\n  LUT6 vs FP16 ratio: {ratio:.2f}x")

    # Decision gate
    print(f"\n{'=' * 60}")
    print("DECISION GATE")
    print(f"{'=' * 60}")
    if "FFN LUT6" in results:
        mean = results["FFN LUT6"]
        if mean < 1.5:
            print(f"  {mean:.3f}ms < 1.5ms → PROCEED (comparable to private API)")
        elif mean < 3.0:
            print(f"  {mean:.3f}ms < 3.0ms → PROCEED WITH CAUTION (slower but viable for 27B)")
        else:
            print(f"  {mean:.3f}ms > 3.0ms → ABORT (overhead too high)")


if __name__ == "__main__":
    main()
