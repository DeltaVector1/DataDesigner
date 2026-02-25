#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Clean meta-references from a critic-revision ShareGPT dataset.

Two-pass pipeline:
  Pass 1 — LLM cleaning: fires all candidate turns concurrently at an
           OpenAI-compatible endpoint, collects cleaned responses.
  Pass 2 — Quality gate: runs deterministic checks on every result and
           auto-repairs common failure modes (wrong empties, residual
           meta, invented tags) without needing a second LLM call.

Usage:
    python scripts/clean_meta_references.py --endpoint http://host:port/v1
    python scripts/clean_meta_references.py --dry-run
    python scripts/clean_meta_references.py --limit 50 --concurrency 32
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
from openai import AsyncOpenAI, OpenAI

ENDPOINT_URL = "http://202.214.223.66:10056/v1"
API_KEY = "not-needed"
DEFAULT_CONCURRENCY = 64

DATASET_ID = "NewEden/CAI-critic-revision-8k-opus-22k-subset-sharegpt"
OUTPUT_DIR = Path("artifacts/cleaned_sharegpt")

# Regex pre-filters — a turn matching any of these gets sent to the LLM.
META_INDICATORS: list[str] = [
    r"^<revision>",
    r"^The assistant['\u2019]s",
    r"^No revision needed",
    r"^The original response",
    r"^Actually, the response",
    r"^The response is",
    r"^Your answer is",
    r"^Revised response",
    r"^The reply",
    r"^Critique:",
    r"Here['\u2019]s a concise revision",
    r"Here['\u2019]s a (?:lighter|revised|cleaner|corrected)",
    r"revision that keeps",
    r"so no revision is needed",
    r"no revision is necessary",
]

# Patterns that identify text which is STILL meta-commentary after cleaning.
# Used by the quality gate to catch residual meta the LLM missed.
RESIDUAL_META_PATTERNS: list[str] = [
    r"^The assistant['\u2019]s (?:reply|response|answer|final sentence|poem)",
    r"^The original response",
    r"^The response is (?:accurate|well|clear|appropriate)",
    r"^The reply is",
    r"^Your answer is",
    r"^No revision (?:needed|necessary|is needed|is necessary|required)",
    r"^No changes (?:needed|necessary|are needed)",
    r"^Actually, the response",
    r"^Critique:",
    r"^I checked the response",
    r"^No contrarian",
    r"the assistant['\u2019]s (?:duty|response|reply) was",
    r"no revision is (?:needed|necessary|justified)",
    r"without any contrarian",
    r"performative (?:disagreement|self-deprecation)",
]

SYSTEM_PROMPT = (
    "You are a precise text editor. Remove meta-commentary about revisions/rewrites "
    "from assistant responses. Preserve ALL actual user-facing content exactly as-is. "
    "Do NOT add any XML tags, wrappers, or formatting that wasn't in the original. "
    "Do NOT invent structure. Just remove the meta parts and return the rest unchanged."
)

CLEANING_PROMPT = """\
<task>
The text below is an assistant response from a conversation dataset.
During dataset creation, a critic-revision process left behind meta-commentary
that talks ABOUT the response being revised, rather than actually responding to
the user. Your job: remove that meta-commentary and return ONLY the actual
response content, unchanged.
</task>

<meta_reference_examples>
These are examples of meta-references you MUST remove:

- "The assistant's reply is clear, friendly, and accurate, but it contains one factual slip..."
- "Here's a concise revision that keeps the warmth while fixing the issue:"
- "No revision needed. The original response is appropriate..."
- "<revision>" or "</revision>" XML tags
- "Critique: The original response already uses..."
- "Your answer is well-structured, enthusiastic, and fact-rich, so I'll keep the core content but..."
- "The reply is accurate and well-structured, so..."
- Horizontal rule separators (---) used to divide meta-commentary from the actual content
- Any sentence that evaluates, critiques, or discusses "the assistant's response", "the original response", "the reply", or "your answer"
- Preambles like "Revised response:" or "H: ... A: ..." formatting added by the critic
</meta_reference_examples>

<rules>
1. Remove ALL meta-references about the revision/critique process
2. Remove horizontal rules (---) that separate meta-commentary from actual content
3. Remove <revision> and </revision> tags
4. Do NOT add any new XML tags (no <poem>, <list>, etc.) — output plain content only
5. Keep the actual response content EXACTLY as-is — do not rephrase, summarize, add to it, or change its formatting
6. Preserve ALL code, math, stories, checklists, explanations, and inline comments verbatim
7. If a sentence mixes meta-framing with useful info, keep the useful part
8. If the ENTIRE text is meta-commentary with no actual user-facing content, output exactly: EMPTY_RESPONSE
9. Do NOT remove content that merely uses words like "revision" or "response" in normal conversational context
</rules>

<assistant_response>
{text}
</assistant_response>

Put your cleaned output inside <cleaned> tags. Nothing else outside the tags."""


def detect_model_name(endpoint: str) -> str:
    """Auto-detect the served model name from the endpoint."""
    client = OpenAI(base_url=endpoint, api_key=API_KEY)
    models = client.models.list()
    model_ids = [m.id for m in models.data]
    if not model_ids:
        print("ERROR: No models found at endpoint.", file=sys.stderr)
        sys.exit(1)
    if len(model_ids) > 1:
        print(f"  Found multiple models: {model_ids}, using first.", file=sys.stderr)
    return model_ids[0]


def has_meta_reference(text: str) -> bool:
    """Pre-filter: does this text likely contain meta-references?"""
    return any(re.search(p, text.strip(), re.IGNORECASE) for p in META_INDICATORS)


def parse_model_output(raw: str) -> str:
    """Extract cleaned text, stripping <think> and <cleaned> wrappers."""
    without_think = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    match = re.search(r"<cleaned>(.*?)</cleaned>", without_think, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return without_think


def regex_strip_revision_tags(text: str) -> str:
    r"""Strip <revision>/<\/revision> tags and meta-preamble above a --- separator."""
    cleaned = re.sub(r"</?revision>", "", text).strip()

    # If the first block before --- is meta-commentary, drop it
    parts = re.split(r"\n---+\n", cleaned, maxsplit=1)
    if len(parts) == 2:
        first = parts[0].strip()
        if any(re.search(p, first, re.IGNORECASE) for p in RESIDUAL_META_PATTERNS):
            cleaned = parts[1].strip()

    return cleaned


def is_residual_meta(text: str) -> bool:
    """Check if text is still meta-commentary after cleaning."""
    stripped = text.strip()
    return any(re.search(p, stripped, re.IGNORECASE) for p in RESIDUAL_META_PATTERNS)


def has_invented_tags(original: str, cleaned: str) -> str | None:
    """Check if the model invented XML tags not present in the original."""
    invented = re.findall(r"</?(\w+)>", cleaned)
    original_tags = set(re.findall(r"</?(\w+)>", original))
    allowed = {"revision", "cleaned", "think"}
    for tag in invented:
        if tag not in original_tags and tag not in allowed:
            return tag
    return None


def sanity_check(original: str, cleaned: str) -> dict[str, Any]:
    """Run all deterministic quality checks on a cleaned result."""
    result: dict[str, Any] = {
        "original_len": len(original),
        "cleaned_len": len(cleaned),
        "is_empty": cleaned == "EMPTY_RESPONSE",
        "len_ratio": len(cleaned) / len(original) if original else 0,
        "passed": True,
        "warnings": [],
        "auto_action": None,
    }

    if result["is_empty"]:
        # Verify the EMPTY verdict — if the original had substantial content
        # after stripping <revision> tags, it was probably wrong.
        stripped = regex_strip_revision_tags(original)
        if len(stripped) > 100 and not is_residual_meta(stripped):
            result["warnings"].append("WRONG_EMPTY: real content exists after tag stripping")
            result["auto_action"] = "regex_restore"
            result["passed"] = False
        return result

    if len(cleaned) > len(original) + 20:
        result["warnings"].append("ADDED_CONTENT: cleaned is longer than original")
        result["passed"] = False

    invented = has_invented_tags(original, cleaned)
    if invented:
        result["warnings"].append(f"INVENTED_TAG: <{invented}> not in original")
        result["auto_action"] = "strip_invented_tags"
        result["passed"] = False

    if is_residual_meta(cleaned):
        result["warnings"].append("RESIDUAL_META: cleaned text is still meta-commentary")
        result["auto_action"] = "drop"
        result["passed"] = False

    if result["len_ratio"] < 0.2:
        result["warnings"].append(f"LARGE_REDUCTION: only {result['len_ratio']:.0%} retained")

    # Spot-check: chunks from cleaned text should appear verbatim in original
    if len(cleaned) > 60:
        for pos in [0, len(cleaned) // 3, 2 * len(cleaned) // 3]:
            chunk = cleaned[pos : pos + 40]
            if chunk and chunk not in original:
                result["warnings"].append(f"NEW_TEXT_AT_{pos}: '{chunk[:50]}' not in original")
                result["passed"] = False
                break

    return result


def auto_repair(original: str, cleaned: str, check: dict[str, Any]) -> str | None:
    """Apply automatic repairs based on sanity check findings.

    Returns the repaired text, or None if the row should be dropped.
    """
    action = check.get("auto_action")

    if action == "regex_restore":
        return regex_strip_revision_tags(original)

    if action == "strip_invented_tags":
        original_tags = set(re.findall(r"</?(\w+)>", original))
        allowed = original_tags | {"revision", "cleaned", "think"}

        def strip_tag(m: re.Match) -> str:
            tag = m.group(1)
            return "" if tag not in allowed else m.group(0)

        return re.sub(r"</?(\w+)>", strip_tag, cleaned).strip()

    if action == "drop":
        return None

    return cleaned


def collect_candidates(ds: Dataset, limit: int | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Scan dataset and collect all GPT turns that need cleaning."""
    candidates: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []

    total = min(len(ds), limit) if limit else len(ds)
    for idx in range(total):
        row = ds[idx]
        conversations = row["conversations"]
        row_turns: list[dict[str, Any]] = []

        for turn_idx, turn in enumerate(conversations):
            if turn["from"] == "gpt" and has_meta_reference(turn["value"]):
                candidates.append({"row_idx": idx, "turn_idx": turn_idx, "text": turn["value"]})
                row_turns.append(
                    {
                        "from": turn["from"],
                        "value": None,
                        "_candidate_idx": len(candidates) - 1,
                    }
                )
            else:
                row_turns.append({"from": turn["from"], "value": turn["value"]})

        all_rows.append({"conversations": row_turns})

    return candidates, all_rows


async def clean_one(
    client: AsyncOpenAI,
    model: str,
    sem: asyncio.Semaphore,
    candidate: dict[str, Any],
    progress: dict[str, int],
) -> dict[str, Any]:
    """Clean a single candidate turn via the LLM."""
    prompt = CLEANING_PROMPT.replace("{text}", candidate["text"])

    async with sem:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=16384,
        )

    raw_output = response.choices[0].message.content or ""
    cleaned = parse_model_output(raw_output)
    check = sanity_check(candidate["text"], cleaned)

    progress["done"] += 1
    n = progress["done"]
    total = progress["total"]
    status = "OK" if check["passed"] else "WARN"
    empty_tag = " [EMPTY]" if check["is_empty"] else ""
    ratio = f"{check['len_ratio']:.0%}"
    action = f" -> {check['auto_action']}" if check.get("auto_action") else ""
    warns = "; ".join(check["warnings"]) if check["warnings"] else ""
    print(
        f"  [{n}/{total}] Row {candidate['row_idx']} turn {candidate['turn_idx']}: "
        f"{status} (kept {ratio}){empty_tag}{action}"
        f"{' -- ' + warns if warns else ''}",
        flush=True,
    )

    return {
        "row_idx": candidate["row_idx"],
        "turn_idx": candidate["turn_idx"],
        "original_text": candidate["text"],
        "cleaned_text": cleaned,
        "sanity": check,
    }


async def run_pass1(
    endpoint: str,
    model: str,
    candidates: list[dict[str, Any]],
    concurrency: int,
) -> list[dict[str, Any]]:
    """Fire all candidates at the endpoint concurrently."""
    client = AsyncOpenAI(base_url=endpoint, api_key=API_KEY)
    sem = asyncio.Semaphore(concurrency)
    progress = {"done": 0, "total": len(candidates)}

    tasks = [clean_one(client, model, sem, c, progress) for c in candidates]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    processed: list[dict[str, Any]] = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            print(
                f"  ERROR on candidate {i} (row {candidates[i]['row_idx']}): {r}",
                file=sys.stderr,
            )
            processed.append(
                {
                    "row_idx": candidates[i]["row_idx"],
                    "turn_idx": candidates[i]["turn_idx"],
                    "original_text": candidates[i]["text"],
                    "cleaned_text": candidates[i]["text"],
                    "sanity": {
                        "passed": False,
                        "warnings": [f"API_ERROR: {r}"],
                        "is_empty": False,
                        "auto_action": None,
                    },
                    "error": str(r),
                }
            )
        else:
            processed.append(r)

    return processed


def run_pass2(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply auto-repairs and produce final results."""
    repaired: list[dict[str, Any]] = []
    stats = {"repaired": 0, "dropped": 0, "kept": 0}

    for r in results:
        check = r["sanity"]
        original = r["original_text"]
        cleaned = r["cleaned_text"]

        if check.get("auto_action"):
            fixed = auto_repair(original, cleaned, check)
            if fixed is None:
                r["final_text"] = None
                r["final_action"] = "dropped"
                stats["dropped"] += 1
            else:
                r["final_text"] = fixed
                r["final_action"] = f"auto_repaired:{check['auto_action']}"
                stats["repaired"] += 1
        elif check["is_empty"]:
            r["final_text"] = None
            r["final_action"] = "empty"
            stats["dropped"] += 1
        else:
            r["final_text"] = cleaned
            r["final_action"] = "kept"
            stats["kept"] += 1

        repaired.append(r)

    print(
        f"\nPass 2 — Quality gate: {stats['kept']} kept, {stats['repaired']} auto-repaired, {stats['dropped']} dropped"
    )
    return repaired


def reassemble_and_save(
    all_rows: list[dict[str, Any]],
    results: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    """Plug results back into rows and save outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for r in results:
        row = all_rows[r["row_idx"]]
        turn = row["conversations"][r["turn_idx"]]
        final = r.get("final_text")
        if final is None:
            turn["value"] = ""
            turn["_drop"] = True
        else:
            turn["value"] = final

    dataset_path = output_dir / "cleaned_dataset.jsonl"
    kept_count = 0
    with open(dataset_path, "w") as f:
        for row in all_rows:
            convs = [
                {"from": t["from"], "value": t["value"]}
                for t in row["conversations"]
                if not t.get("_drop") and t.get("value")
            ]
            roles = {t["from"] for t in convs}
            if "human" in roles and "gpt" in roles:
                f.write(json.dumps({"conversations": convs}, ensure_ascii=False) + "\n")
                kept_count += 1
    print(f"Saved {kept_count} rows to {dataset_path}")

    log_path = output_dir / "change_log.jsonl"
    with open(log_path, "w") as f:
        for r in results:
            entry = {
                "row_idx": r["row_idx"],
                "turn_idx": r["turn_idx"],
                "original_preview": r["original_text"][:400],
                "cleaned_preview": (r.get("final_text") or "")[:400],
                "final_action": r.get("final_action", ""),
                "sanity": r["sanity"],
            }
            if r.get("error"):
                entry["error"] = r["error"]
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    print(f"Saved change log ({len(results)} entries) to {log_path}")

    report_path = output_dir / "diff_report.txt"
    with open(report_path, "w") as f:
        for r in results:
            sanity = r["sanity"]
            action = r.get("final_action", "")
            f.write(f"{'=' * 80}\n")
            f.write(f"Row {r['row_idx']}, Turn {r['turn_idx']}  [{action}]\n")
            if sanity.get("warnings"):
                f.write(f"Warnings: {', '.join(sanity['warnings'])}\n")
            f.write(f"\n--- ORIGINAL ---\n{r['original_text'][:500]}\n")
            final = r.get("final_text") or "[DROPPED]"
            f.write(f"\n--- FINAL ---\n{final[:500]}\n\n")
    print(f"Saved diff report to {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean meta-references from a critic-revision ShareGPT dataset")
    parser.add_argument("--dry-run", action="store_true", help="Identify candidates only, no API calls")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N rows")
    parser.add_argument("--model", type=str, default=None, help="Model name (auto-detected if omitted)")
    parser.add_argument("--endpoint", type=str, default=ENDPOINT_URL, help="OpenAI-compatible endpoint URL")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR), help="Output directory")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="Max concurrent requests")
    parser.add_argument("--dataset", type=str, default=DATASET_ID, help="HuggingFace dataset ID")
    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset}")
    ds = load_dataset(args.dataset, split="train")
    print(f"Loaded {len(ds)} rows")

    print("Collecting candidates...")
    candidates, all_rows = collect_candidates(ds, limit=args.limit)
    print(f"Found {len(candidates)} candidates out of {min(len(ds), args.limit) if args.limit else len(ds)} rows")

    if args.dry_run:
        print("\n=== DRY RUN ===")
        for c in candidates:
            print(f"  Row {c['row_idx']} turn {c['turn_idx']}: {c['text'][:120]}...")
        print(f"\nWould process {len(candidates)} turns.")
        return

    model = args.model or detect_model_name(args.endpoint)
    print(f"Using model: {model}")
    print(f"Concurrency: {args.concurrency}")
    print(f"\n--- Pass 1: LLM cleaning ({len(candidates)} requests) ---\n")

    start = time.time()
    results = asyncio.run(run_pass1(args.endpoint, model, candidates, args.concurrency))
    p1_elapsed = time.time() - start
    print(f"\nPass 1 done: {len(results)} results in {p1_elapsed:.1f}s ({len(results) / p1_elapsed:.1f} req/s)")

    print("\n--- Pass 2: Quality gate & auto-repair ---")
    final_results = run_pass2(results)

    reassemble_and_save(all_rows, final_results, Path(args.output))

    elapsed = time.time() - start
    warnings_list = [r for r in final_results if r["sanity"].get("warnings")]
    errors = [r for r in final_results if r.get("error")]
    print(f"\nTotal time: {elapsed:.1f}s")
    if errors:
        print(f"  {len(errors)} API errors (kept originals)")
    if warnings_list:
        print(f"  {len(warnings_list)} entries had warnings -- review diff_report.txt")
    if not warnings_list and not errors:
        print("  All checks passed.")


if __name__ == "__main__":
    main()
