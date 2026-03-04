#!/usr/bin/env python3
"""Quick test of the ANE utility model via CoreML.

Usage:
    python scripts/test_ane_utility.py --model /Volumes/LLM\ Store/models/coreml/gemma-3-4b-ane
"""
import sys
import time
import argparse

sys.path.insert(0, '/Users/hb/projects/deepseek_web/python')

from ane_model import ANEUtilityModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to ANEMLL model dir')
    parser.add_argument('--prompt', default='What is the capital of France?',
                        help='Test prompt')
    parser.add_argument('--max-tokens', type=int, default=50)
    parser.add_argument('--raw', action='store_true',
                        help='Use raw tokenization (no chat template)')
    args = parser.parse_args()

    print(f"Loading ANE model from: {args.model}")
    model = ANEUtilityModel(args.model)
    model.load()

    print(f"\n--- Prompt: {args.prompt}")
    print(f"--- Max tokens: {args.max_tokens}")
    print(f"--- Raw mode: {args.raw}\n")

    # Test 1: Basic generation
    t0 = time.time()
    result = model.generate(args.prompt, max_tokens=args.max_tokens, raw=args.raw)
    elapsed = time.time() - t0
    print(f"\n=== Result ({elapsed:.2f}s) ===")
    print(result)

    # Test 2: Utility-style prompt (classification)
    print("\n\n--- Test 2: Classification task ---")
    classify_prompt = (
        "Classify this question as SEARCH or KNOWLEDGE.\n"
        "SEARCH = needs external data. KNOWLEDGE = can answer from training.\n"
        "Question: What are the side effects of metformin?\n"
        "Answer (one word):"
    )
    t0 = time.time()
    result2 = model.generate(classify_prompt, max_tokens=5, raw=True)
    elapsed = time.time() - t0
    print(f"Result ({elapsed:.2f}s): {result2!r}")

    # Test 3: JSON extraction (like analyze_prompt)
    print("\n\n--- Test 3: JSON extraction ---")
    json_prompt = (
        "Extract entities from this question as JSON.\n"
        "Question: How does metformin affect gut microbiome in type 2 diabetes?\n"
        'Return: {"entities": ["entity1", "entity2"], "domain": "domain_name"}\n'
        "JSON:"
    )
    t0 = time.time()
    result3 = model.generate(json_prompt, max_tokens=60, raw=True)
    elapsed = time.time() - t0
    print(f"Result ({elapsed:.2f}s): {result3!r}")

    # Test 4: PubMed query optimization
    print("\n\n--- Test 4: Query optimization ---")
    query_prompt = (
        "Convert this question into 2-3 PubMed search terms.\n"
        "Question: What are the benefits of GHK-Cu copper peptide for skin?\n"
        "Query:"
    )
    t0 = time.time()
    result4 = model.generate(query_prompt, max_tokens=20, raw=True)
    elapsed = time.time() - t0
    print(f"Result ({elapsed:.2f}s): {result4!r}")

    print("\n=== All tests complete ===")


if __name__ == '__main__':
    main()
