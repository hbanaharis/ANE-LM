#!/usr/bin/env python3
"""Benchmark ANE utility model throughput."""
import sys, time
sys.path.insert(0, '/Users/hb/projects/deepseek_web/python')
from ane_model import ANEUtilityModel

model = ANEUtilityModel('/Volumes/LLM Store/models/coreml/gemma-3-4b-ane')
model.load()

# Warmup
print("Warming up...")
for _ in range(3):
    model.generate('Hi', max_tokens=5, raw=True)
print("Warmup done.\n")

# Test 1: 100-token generation
print("=== 100-token generation ===")
prompt = 'Explain the mechanism of action of metformin in type 2 diabetes.'
t0 = time.time()
result = model.generate(prompt, max_tokens=100, raw=True)
elapsed = time.time() - t0
print(f"Time: {elapsed:.2f}s")
print(f"Output: {result[:200]}")

# Test 2: Short utility calls (typical for classify/route)
print("\n=== Short calls (5 tokens, 5 runs) ===")
times = []
for i in range(5):
    t0 = time.time()
    r = model.generate(
        'SEARCH or KNOWLEDGE? Question: What causes cancer? Answer:',
        max_tokens=5, raw=True
    )
    t = time.time() - t0
    times.append(t)
    print(f"  Run {i+1}: {t:.3f}s -> {r.strip()!r}")
avg = sum(times) / len(times)
print(f"  Average: {avg:.3f}s")

# Test 3: Medium utility call (like analyze_prompt)
print("\n=== Medium call (60 tokens) ===")
t0 = time.time()
r = model.generate(
    'Extract entities from: "How does GLP-1 receptor agonist affect cardiovascular outcomes in patients with type 2 diabetes?" '
    'Return JSON: {"entities": [...], "domain": "..."}',
    max_tokens=60, raw=True
)
elapsed = time.time() - t0
print(f"Time: {elapsed:.2f}s")
print(f"Output: {r.strip()}")

print("\n=== Done ===")
