#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Clean meta-references from a critic-revision ShareGPT dataset.

Uses DataDesigner's CustomColumnConfig + ModelFacade to handle all LLM
concurrency, progress tracking, and error handling. The domain logic
(regex patterns, quality checks, repair heuristics) is preserved.

Pipeline:
  1. Pre-process  — Load HF dataset, flatten candidate turns into a DataFrame
  2. DataDesigner — Custom column generator cleans each turn via ModelFacade
                    and runs the quality gate + auto-repair
  3. Post-process — Merge cleaned turns back into conversations, save JSONL

Usage:
    python scripts/clean_meta_references.py --endpoint http://host:port/v1
    python scripts/clean_meta_references.py --dry-run
    python scripts/clean_meta_references.py --limit 50
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset
from openai import OpenAI

import data_designer.config as dd
from data_designer.config.custom_column import custom_column_generator
from data_designer.config.seed_source_dataframe import DataFrameSeedSource
from data_designer.interface import DataDesigner

ENDPOINT_URL = "http://202.214.223.66:10056/v1"
API_KEY = "not-needed"
DEFAULT_CONCURRENCY = 64

DATASET_ID = "NewEden/CAI-critic-revision-8k-opus-22k-subset-sharegpt"
OUTPUT_DIR = Path("artifacts/cleaned_sharegpt")

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------

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
- Any sentence that evaluates, critiques, or discusses "the assistant's response", \
"the original response", "the reply", or "your answer"
- Preambles like "Revised response:" or "H: ... A: ..." formatting added by the critic
</meta_reference_examples>

<rules>
1. Remove ALL meta-references about the revision/critique process
2. Remove horizontal rules (---) that separate meta-commentary from actual content
3. Remove <revision> and </revision> tags
4. Do NOT add any new XML tags (no <poem>, <list>, etc.) — output plain content only
5. Keep the actual response content EXACTLY as-is — do not rephrase, summarize, add to it, \
or change its formatting
6. Preserve ALL code, math, stories, checklists, explanations, and inline comments verbatim
7. If a sentence mixes meta-framing with useful info, keep the useful part
8. If the ENTIRE text is meta-commentary with no actual user-facing content, output exactly: \
EMPTY_RESPONSE
9. Do NOT remove content that merely uses words like "revision" or "response" in normal \
conversational context
</rules>

<assistant_response>
{text}
</assistant_response>

Put your cleaned output inside <cleaned> tags. Nothing else outside the tags."""


# ---------------------------------------------------------------------------
# Domain functions (unchanged)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Pre-processing: flatten candidates into a DataFrame
# ---------------------------------------------------------------------------


def flatten_candidates(dataset_id: str, limit: int | None = None) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Load HF dataset and extract candidate turns into a flat DataFrame.

    Returns:
        candidates_df: DataFrame with columns [row_idx, turn_idx, original_text]
        all_rows: list of row skeletons for reassembly
    """
    print(f"Loading dataset: {dataset_id}")
    ds = load_dataset(dataset_id, split="train")
    print(f"Loaded {len(ds)} rows")

    candidates: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []

    total = min(len(ds), limit) if limit else len(ds)
    for idx in range(total):
        row = ds[idx]
        conversations = row["conversations"]
        row_turns: list[dict[str, Any]] = []

        for turn_idx, turn in enumerate(conversations):
            if turn["from"] == "gpt" and has_meta_reference(turn["value"]):
                candidates.append(
                    {
                        "row_idx": idx,
                        "turn_idx": turn_idx,
                        "original_text": turn["value"],
                    }
                )
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

    print(f"Found {len(candidates)} candidates out of {total} rows")

    candidates_df = (
        pd.DataFrame(candidates) if candidates else pd.DataFrame(columns=["row_idx", "turn_idx", "original_text"])
    )
    return candidates_df, all_rows


# ---------------------------------------------------------------------------
# Custom column generator: LLM cleaning + quality gate
# ---------------------------------------------------------------------------


@custom_column_generator(
    required_columns=["original_text"],
    side_effect_columns=["action", "warnings"],
    model_aliases=["cleaner"],
)
def clean_and_validate(row: dict, generator_params: None, models: dict) -> dict:
    """Clean a single turn via ModelFacade, then run quality gate + auto-repair."""
    original = row["original_text"]

    prompt = CLEANING_PROMPT.replace("{text}", original)
    raw_output, _ = models["cleaner"].generate(
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
        parser=parse_model_output,
    )
    cleaned = raw_output

    check = sanity_check(original, cleaned)

    if check.get("auto_action"):
        fixed = auto_repair(original, cleaned, check)
        if fixed is None:
            row["cleaned_text"] = ""
            row["action"] = "dropped"
        else:
            row["cleaned_text"] = fixed
            row["action"] = f"auto_repaired:{check['auto_action']}"
    elif check["is_empty"]:
        row["cleaned_text"] = ""
        row["action"] = "empty"
    else:
        row["cleaned_text"] = cleaned
        row["action"] = "kept"

    row["warnings"] = "; ".join(check["warnings"]) if check["warnings"] else ""
    return row


# ---------------------------------------------------------------------------
# Post-processing: reassemble and save
# ---------------------------------------------------------------------------


def reassemble_and_save(
    all_rows: list[dict[str, Any]],
    result_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plug cleaned results back into rows and save outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    stats: dict[str, int] = {"kept": 0, "repaired": 0, "dropped": 0, "errors": 0}

    for _, r in result_df.iterrows():
        row_idx = int(r["row_idx"])
        turn_idx = int(r["turn_idx"])
        action = r.get("action", "kept")
        row = all_rows[row_idx]
        turn = row["conversations"][turn_idx]

        if action in ("dropped", "empty"):
            turn["value"] = ""
            turn["_drop"] = True
            stats["dropped"] += 1
        elif "auto_repaired" in str(action):
            turn["value"] = r["cleaned_text"]
            stats["repaired"] += 1
        else:
            turn["value"] = r["cleaned_text"]
            stats["kept"] += 1

    print(
        f"\nResults: {stats['kept']} kept, {stats['repaired']} auto-repaired, "
        f"{stats['dropped']} dropped, {stats['errors']} errors"
    )

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
        for _, r in result_df.iterrows():
            entry = {
                "row_idx": int(r["row_idx"]),
                "turn_idx": int(r["turn_idx"]),
                "original_preview": str(r["original_text"])[:400],
                "cleaned_preview": str(r.get("cleaned_text", ""))[:400],
                "action": r.get("action", ""),
                "warnings": r.get("warnings", ""),
            }
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    print(f"Saved change log ({len(result_df)} entries) to {log_path}")

    report_path = output_dir / "diff_report.txt"
    with open(report_path, "w") as f:
        for _, r in result_df.iterrows():
            action = r.get("action", "")
            warns = r.get("warnings", "")
            f.write(f"{'=' * 80}\n")
            f.write(f"Row {int(r['row_idx'])}, Turn {int(r['turn_idx'])}  [{action}]\n")
            if warns:
                f.write(f"Warnings: {warns}\n")
            f.write(f"\n--- ORIGINAL ---\n{str(r['original_text'])[:500]}\n")
            final = r.get("cleaned_text", "") or "[DROPPED]"
            f.write(f"\n--- FINAL ---\n{str(final)[:500]}\n\n")
    print(f"Saved diff report to {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


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

    candidates_df, all_rows = flatten_candidates(args.dataset, limit=args.limit)

    if len(candidates_df) == 0:
        print("No candidates found. Nothing to do.")
        return

    if args.dry_run:
        print("\n=== DRY RUN ===")
        for _, c in candidates_df.iterrows():
            print(f"  Row {int(c['row_idx'])} turn {int(c['turn_idx'])}: {str(c['original_text'])[:120]}...")
        print(f"\nWould process {len(candidates_df)} turns.")
        return

    model_name = args.model or detect_model_name(args.endpoint)
    print(f"Using model: {model_name}")
    print(f"Concurrency: {args.concurrency}")

    # --- Build DataDesigner pipeline ---

    data_designer = DataDesigner(
        model_providers=[
            dd.ModelProvider(
                name="cleaning-endpoint",
                endpoint=args.endpoint,
                api_key=API_KEY,
            ),
        ],
    )

    builder = dd.DataDesignerConfigBuilder(
        model_configs=[
            dd.ModelConfig(
                alias="cleaner",
                model=model_name,
                provider="cleaning-endpoint",
                inference_parameters=dd.ChatCompletionInferenceParams(
                    temperature=0.0,
                    max_tokens=16384,
                    max_parallel_requests=args.concurrency,
                ),
                skip_health_check=True,
            ),
        ],
    )

    builder.with_seed_dataset(DataFrameSeedSource(df=candidates_df))

    builder.add_column(
        dd.CustomColumnConfig(
            name="cleaned_text",
            generator_function=clean_and_validate,
        ),
    )

    # --- Run pipeline ---

    print(f"\n--- Cleaning {len(candidates_df)} candidates via DataDesigner ---\n")
    start = time.time()

    results = data_designer.create(builder, num_records=len(candidates_df), dataset_name="meta_cleaning")
    result_df = results.load()

    elapsed = time.time() - start
    print(f"\nDone: {len(result_df)} results in {elapsed:.1f}s ({len(result_df) / elapsed:.1f} req/s)")

    # --- Reassemble and save ---

    reassemble_and_save(all_rows, result_df, Path(args.output))

    warnings_list = result_df[result_df["warnings"].astype(str).str.len() > 0]
    print(f"\nTotal time: {elapsed:.1f}s")
    if len(warnings_list) > 0:
        print(f"  {len(warnings_list)} entries had warnings -- review diff_report.txt")
    else:
        print("  All checks passed.")


if __name__ == "__main__":
    main()
