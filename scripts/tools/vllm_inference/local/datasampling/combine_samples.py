#!/usr/bin/env python
"""Combine all sampled *.jsonl files under a data root into a single jsonl.

The previous version hard‑coded a list of sampled files. This version discovers
them automatically so it works with the updated sampling bash script.

Rules for the `source` field (mirrors legacy behavior):
  * Base is the relative path with extension stripped and the first 'sampled/'
    segment removed.
  * If a corresponding '.../filtered/...' file exists (e.g. math/filtered/XYZ.jsonl)
    we set source to that filtered path (without .jsonl) to reflect the original
    dataset origin. This reproduces the special-casing that prepended
    'math/filtered/' earlier.
"""

from __future__ import annotations

import os
import sys
import json
import argparse
from typing import Iterable, List


def discover_sampled_files(root: str, pattern: str = "*.jsonl") -> List[str]:
    sampled_files: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "/sampled" not in dirpath and not dirpath.endswith("/sampled"):
            # Only include files inside a sampled directory (to avoid mixing raw inputs)
            if os.path.basename(dirpath) != "sampled":
                continue
        if os.path.basename(dirpath) == "sampled":
            for fn in filenames:
                if fn.endswith(".jsonl"):
                    sampled_files.append(os.path.join(dirpath, fn))
    sampled_files.sort()
    return sampled_files


def compute_source(root: str, full_path: str) -> str:
    rel = os.path.relpath(full_path, root)
    base_no_ext = rel[:-6] if rel.endswith(".jsonl") else rel
    # Remove the first occurrence of 'sampled/'
    parts = base_no_ext.split("/")
    if "sampled" in parts:
        # remove only first sampled segment
        sampled_index = parts.index("sampled")
        parts.pop(sampled_index)
    candidate_no_filtered = "/".join(parts)
    # Try inserting 'filtered' after the first directory (legacy math case) if a file exists.
    if len(parts) >= 2:
        with_filtered_parts = parts.copy()
        # Insert 'filtered' after first segment if not already present
        if "filtered" not in parts:
            with_filtered_parts.insert(1, "filtered")
            filtered_rel = "/".join(with_filtered_parts) + ".jsonl"
            if os.path.exists(os.path.join(root, filtered_rel)):
                return "/".join(with_filtered_parts)
    return candidate_no_filtered


def iter_items(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping malformed JSON line in {path}: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Combine sampled jsonl files")
    parser.add_argument("--dataroot", default=os.environ.get("DATAROOT", "/datadisk/datasets/seeds/phi/phi4_reasoning_prompts"), help="Root directory containing sampled subdirectories")
    parser.add_argument("--output", default="combined_samples.jsonl", help="Output jsonl file path")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of records (global) to write")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging")
    args = parser.parse_args()

    root = os.path.abspath(args.dataroot)
    files = discover_sampled_files(root)
    if not files:
        print(f"No sampled jsonl files found under {root}", file=sys.stderr)
        return 1
    if not args.quiet:
        print(f"Discovered {len(files)} sampled files", file=sys.stderr)

    total_written = 0
    with open(args.output, "w", encoding="utf-8") as out_f:
        for fp in files:
            source = compute_source(root, fp)
            if not args.quiet:
                print(f"[+] Adding {fp} (source={source})", file=sys.stderr)
            for item in iter_items(fp):
                item["source"] = source
                out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                total_written += 1
                if args.limit and total_written >= args.limit:
                    break
            if args.limit and total_written >= args.limit:
                break

    if not args.quiet:
        print(f"Wrote {total_written} records from {len(files)} files to {args.output}", file=sys.stderr)
    return 0

if __name__ == "__main__":  # pragma: no cover
    main()