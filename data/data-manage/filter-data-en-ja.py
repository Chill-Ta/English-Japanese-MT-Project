"""
filter-data-en-ja.py
════════════════════
Data quality filter for EN-JA parallel corpora.

Enhanced with CLI arguments, integrated semantic filtering from
semantic_filter.py, and improved logging. Maintains backward
compatibility with the original filtering rules.

Usage:
    # Basic filtering (original behavior)
    python data/data-manage/filter-data-en-ja.py

    # Custom paths and thresholds
    python data/data-manage/filter-data-en-ja.py --input data.txt --output filtered.txt --min_score 0.80

    # Full semantic filtering with near-dedup
    python data/data-manage/filter-data-en-ja.py --input data.txt --output filtered.txt --semantic
"""

import re
import os
import sys
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default paths (backward compatible)
DEFAULT_INPUT  = "data/dataset/train-set/en-ja/resources/en-ja.bicleaner05.txt"
DEFAULT_OUTPUT = "data/dataset/train-set/en-ja/resources/en-ja_extracted_filtered.txt"


def clean_text(text):
    """
    Apply regex formatting to remove HTML tags, normalize whitespaces, 
    and clean up punctuation artifacts.
    """
    text = re.sub(r'<[^>]+>', '', text)          # Remove HTML tags
    text = re.sub(r'\s+', ' ', text).strip()     # Normalize multiple spaces to single space
    return text


def run_basic_filter(input_path, output_path, min_score=0.70, max_en_words=100,
                     max_ja_chars=120, max_length_ratio=3.0):
    """Run the original basic filtering pipeline."""
    kept = 0
    total = 0
    seen_sentences = set()

    logger.info("Starting data cleaning process...")
    logger.info("  Input:     %s", input_path)
    logger.info("  Output:    %s", output_path)
    logger.info("  Min score: %.2f", min_score)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            total += 1
            parts = line.strip('\n').split('\t')

            # Rule 1: Structure verification
            if len(parts) < 5:
                continue

            try:
                score = float(parts[2])
            except ValueError:
                continue

            # Rule 2: Minimum quality score
            if score < min_score:
                continue

            # Rule 3: Text cleaning and formatting
            en = clean_text(parts[3])
            ja = clean_text(parts[4])

            en_words = len(en.split())
            ja_chars = len(ja)

            # Rule 4: Empty string or length limits check
            if en_words == 0 or ja_chars == 0:
                continue
            if en_words > max_en_words or ja_chars > max_ja_chars:
                continue

            # Rule 5: Length Ratio Filtering
            # Approximating that 1 English word usually translates to 2-3 Japanese characters
            if (ja_chars / en_words) > max_length_ratio * 3 or (en_words / ja_chars) > max_length_ratio:
                continue

            # Rule 6: Exact Deduplication (Removes repetitive loops)
            unique_key = f"{en}|||{ja}"
            if unique_key in seen_sentences:
                continue
            seen_sentences.add(unique_key)

            # Write cleaned and verified data to output
            parts[3] = en
            parts[4] = ja
            fout.write('\t'.join(parts) + '\n')
            kept += 1

            if total % 500_000 == 0:
                logger.info("  Progress: %s lines processed, %s kept", f"{total:,}", f"{kept:,}")

    logger.info("Filtering completed. Total: %s → Kept: %s (%.1f%%)",
                f"{total:,}", f"{kept:,}", kept / max(1, total) * 100)
    return kept


def parse_args():
    p = argparse.ArgumentParser(description="Filter EN-JA parallel data for quality.")
    p.add_argument("--input", default=DEFAULT_INPUT, help="Input file path")
    p.add_argument("--output", default=DEFAULT_OUTPUT, help="Output file path")
    p.add_argument("--min_score", type=float, default=0.70,
                   help="Minimum BiCleaner score (default: 0.70)")
    p.add_argument("--max_en_words", type=int, default=100,
                   help="Maximum English words per sentence (default: 100)")
    p.add_argument("--max_ja_chars", type=int, default=120,
                   help="Maximum Japanese characters per sentence (default: 120)")
    p.add_argument("--max_length_ratio", type=float, default=3.0,
                   help="Maximum length ratio between EN and JA (default: 3.0)")
    p.add_argument("--semantic", action="store_true",
                   help="Use the advanced semantic filter pipeline instead of basic filtering")
    p.add_argument("--strict", action="store_true",
                   help="Enable strict mode for semantic filter (includes negation checks)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.semantic:
        # Use the advanced semantic filter pipeline
        try:
            from semantic_filter import run_filter, DEFAULT_CONFIG
            cfg = {**DEFAULT_CONFIG, "min_score": args.min_score, "strict": args.strict}
            run_filter(args.input, args.output, cfg)
        except ImportError:
            logger.error("Could not import semantic_filter.py. Make sure it's in the same directory.")
            logger.error("Falling back to basic filter...")
            run_basic_filter(args.input, args.output, args.min_score,
                           args.max_en_words, args.max_ja_chars, args.max_length_ratio)
    else:
        run_basic_filter(args.input, args.output, args.min_score,
                        args.max_en_words, args.max_ja_chars, args.max_length_ratio)