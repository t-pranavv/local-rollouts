"""
Sample a fixed number of JSONL records from a large file.

Usage:
  python sample_seed.py --input data.jsonl --output sample.jsonl --n 100 --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import math
import re
from typing import List, Iterable, Dict, Any


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample N examples from a JSONL file")
    parser.add_argument("--input", "-i", required=True, help="Path to source JSONL file")
    parser.add_argument("--output", "-o", required=True, help="Path to write sampled JSONL file")
    parser.add_argument("--n", type=int, default=100, help="Number of samples to draw (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--with-replacement",
        action="store_true",
        help="Sample with replacement (default: without replacement via reservoir)",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=None,
        help="Optional cap on number of lines to read from input (useful for quick trials).",
    )
    # Diversity-related arguments
    parser.add_argument(
        "--diverse",
        action="store_true",
        help="Enable diversity-aware sampling over a candidate pool (overrides reservoir / replacement logic).",
    )
    parser.add_argument(
        "--text-field",
        default="prompt",
        help="Primary JSON field containing the text/prompt (default: prompt).",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=50000,
        help="Maximum number of candidate lines to load for diversity selection (default: 50000).",
    )
    parser.add_argument(
        "--token-regex",
        default=r"[A-Za-z0-9_]+",
        help="Regex for tokenization when computing diversity (default: '[A-Za-z0-9_]+').",
    )
    parser.add_argument(
        "--diversity-metric",
        choices=["jaccard"],
        default="jaccard",
        help="Metric for diversity distance (currently only jaccard).",
    )
    parser.add_argument(
        "--first-pick",
        choices=["longest", "random"],
        default="longest",
        help="Strategy for the first selected item in diversity sampling (default: longest).",
    )
    return parser.parse_args(argv)


def iter_lines(path: str, max_lines: int | None = None) -> Iterable[str]:
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_lines is not None and idx >= max_lines:
                break
            if line.strip():  # skip pure blank lines
                yield line.rstrip("\n")


def sample_with_replacement(lines: List[str], n: int, rng: random.Random) -> List[str]:
    return [rng.choice(lines) for _ in range(min(n, len(lines)))] if lines else []


def reservoir_sample(path: str, n: int, rng: random.Random, max_lines: int | None = None) -> List[str]:
    reservoir: List[str] = []
    for i, line in enumerate(iter_lines(path, max_lines=max_lines)):
        if i < n:
            reservoir.append(line)
        else:
            # Generate random index between 0 and i (inclusive)
            j = rng.randint(0, i)
            if j < n:
                reservoir[j] = line
    return reservoir


def _extract_text(obj: Dict[str, Any], primary: str) -> str | None:
    # Return primary field if present else first non-empty string value.
    if primary in obj and isinstance(obj[primary], str) and obj[primary].strip():
        return obj[primary]
    for value in obj.values():
        if isinstance(value, str) and value.strip():
            return value
    return None


def _tokenize(text: str, pattern: re.Pattern) -> set:
    return {t.lower() for t in pattern.findall(text)}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return 1.0 - (len(a & b) / max(1, len(a | b)))


def diverse_sample(
    path: str,
    n: int,
    rng: random.Random,
    primary_field: str,
    max_candidates: int,
    token_pattern: str,
    metric: str,
    first_pick: str,
    max_lines: int | None = None,
) -> List[str]:
    token_re = re.compile(token_pattern)
    candidates: List[Dict[str, Any]] = []
    raw_lines: List[str] = []

    for idx, line in enumerate(iter_lines(path, max_lines=max_lines)):
        if idx >= max_candidates:
            break
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            print(f"[WARN] Skipping invalid JSON line {idx+1}", file=sys.stderr)
            continue  # skip invalid
        txt = _extract_text(obj, primary_field)
        if not txt:
            continue
        tokens = _tokenize(txt, token_re)
        if not tokens:
            continue
        candidates.append({"tokens": tokens, "len": len(txt), "line": line})
        raw_lines.append(line)

    if not candidates:
        return []

    if len(candidates) <= n:
        return [c["line"] for c in candidates]

    # Initialize selection
    if first_pick == "longest":
        first_idx = max(range(len(candidates)), key=lambda i: candidates[i]["len"])
    else:
        first_idx = rng.randrange(len(candidates))

    selected_indices = [first_idx]
    # Precompute token sets list for speed
    token_sets = [c["tokens"] for c in candidates]
    distances = [math.inf] * len(candidates)

    while len(selected_indices) < n:
        last_idx = selected_indices[-1]
        last_tokens = token_sets[last_idx]
        for i in range(len(candidates)):
            if i in selected_indices:
                continue
            if metric == "jaccard":
                d = _jaccard(last_tokens, token_sets[i])
            else:
                raise ValueError("Unsupported metric")
            if d < distances[i]:
                distances[i] = d
        # pick farthest (max distance)
        best_idx = None
        best_dist = -1.0
        for i, dist in enumerate(distances):
            if i in selected_indices:
                continue
            if dist > best_dist:
                best_dist = dist
                best_idx = i
        if best_idx is None:
            break
        selected_indices.append(best_idx)

    return [candidates[i]["line"] for i in selected_indices]


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    rng = random.Random(args.seed)

    if not os.path.isfile(args.input):
        print(f"[ERROR] Input file not found: {args.input}", file=sys.stderr)
        return 1

    if args.n <= 0:
        print("[ERROR] --n must be > 0", file=sys.stderr)
        return 1

    if args.diverse:
        sampled = diverse_sample(
            path=args.input,
            n=args.n,
            rng=rng,
            primary_field=args.text_field,
            max_candidates=args.max_candidates,
            token_pattern=args.token_regex,
            metric=args.diversity_metric,
            first_pick=args.first_pick,
            max_lines=args.max_lines,
        )
    elif args.with_replacement:
        # Need all lines (bounded by max_lines if provided)
        lines = list(iter_lines(args.input, max_lines=args.max_lines))
        sampled = sample_with_replacement(lines, args.n, rng)
    else:
        sampled = reservoir_sample(args.input, args.n, rng, max_lines=args.max_lines)

    if not sampled:
        print("[WARN] No lines sampled (input may be empty). Output will be empty.")

    # If file has fewer lines than requested (no replacement), that's fine.
    if not args.with_replacement and len(sampled) < args.n:
        print(
            f"[INFO] Input had only {len(sampled)} lines (fewer than requested {args.n}); wrote all available.",
            file=sys.stderr,
        )

    # Write output
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as out_f:
        for line in sampled:
            out_f.write(line + "\n")

    print(f"[OK] Wrote {len(sampled)} sampled lines to {args.output}")
    return 0


if __name__ == "__main__":
    main()
