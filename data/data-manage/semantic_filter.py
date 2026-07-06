"""
semantic_filter.py
══════════════════
Advanced data quality filtering pipeline for EN-JA parallel corpora.

Applies multiple filtering stages beyond the basic length/score filtering
in filter-data-en-ja.py:
  1. Language detection (script-ratio heuristic)
  2. Polarity/negation consistency check
  3. Near-deduplication via SimHash
  4. Elevated BiCleaner threshold for noisy web-crawled data

Usage:
    python data/data-manage/semantic_filter.py --input data.txt --output filtered.txt
    python data/data-manage/semantic_filter.py --input data.txt --output filtered.txt --min_score 0.80 --strict
"""

import re
import os
import sys
import argparse
import hashlib
import logging
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Filter Configuration
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "min_score": 0.75,          # BiCleaner score threshold (raised from 0.70)
    "max_en_words": 100,
    "max_ja_chars": 150,
    "min_en_words": 3,          # NEW: minimum English sentence length
    "min_ja_chars": 5,          # NEW: minimum Japanese sentence length
    "max_length_ratio": 3.0,
    "ja_script_min_ratio": 0.30,  # Minimum ratio of JA script chars in JA column
    "en_latin_min_ratio": 0.50,   # Minimum ratio of Latin chars in EN column
    "simhash_threshold": 3,       # Hamming distance threshold for near-dedup
}


# ──────────────────────────────────────────────────────────────────────────────
# Text Cleaning
# ──────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Remove HTML tags, normalize whitespace, strip."""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ──────────────────────────────────────────────────────────────────────────────
# Filter Functions
# ──────────────────────────────────────────────────────────────────────────────

def check_length(en: str, ja: str, cfg: dict) -> tuple[bool, str]:
    """Check sentence length constraints."""
    en_words = len(en.split())
    ja_chars = len(ja)

    if en_words < cfg["min_en_words"]:
        return False, "en_too_short"
    if ja_chars < cfg["min_ja_chars"]:
        return False, "ja_too_short"
    if en_words > cfg["max_en_words"]:
        return False, "en_too_long"
    if ja_chars > cfg["max_ja_chars"]:
        return False, "ja_too_long"

    # Length ratio check (EN word count vs JA char count)
    if en_words > 0 and ja_chars > 0:
        ratio_ja_en = ja_chars / en_words
        ratio_en_ja = en_words / ja_chars
        if ratio_ja_en > cfg["max_length_ratio"] * 3:
            return False, "length_ratio_ja_too_long"
        if ratio_en_ja > cfg["max_length_ratio"]:
            return False, "length_ratio_en_too_long"

    return True, ""


def check_language_script(en: str, ja: str, cfg: dict) -> tuple[bool, str]:
    """
    Verify that EN column is actually English and JA column is actually Japanese
    using character-script ratios. This catches web-crawled noise where both
    columns might be in the same language, or contain mostly non-target script.
    """
    # Check EN column: should be predominantly Latin script
    en_non_ws = [ch for ch in en if not ch.isspace()]
    if en_non_ws:
        latin_count = sum(1 for ch in en_non_ws if ch.isascii() and (ch.isalpha() or ch in ".,!?;:'-\"()"))
        latin_ratio = latin_count / len(en_non_ws)
        if latin_ratio < cfg["en_latin_min_ratio"]:
            return False, "en_not_latin"

    # Check JA column: should contain meaningful Japanese characters
    ja_non_ws = [ch for ch in ja if not ch.isspace()]
    if ja_non_ws:
        ja_count = sum(1 for ch in ja_non_ws if (
            '\u3040' <= ch <= '\u309F' or   # Hiragana
            '\u30A0' <= ch <= '\u30FF' or   # Katakana
            '\u4E00' <= ch <= '\u9FFF' or   # CJK
            ch in '。、！？「」『』（）ー～'
        ))
        ja_ratio = ja_count / len(ja_non_ws)
        if ja_ratio < cfg["ja_script_min_ratio"]:
            return False, "ja_not_japanese"

    return True, ""


def check_negation_consistency(en: str, ja: str) -> tuple[bool, str]:
    """
    Heuristic check for polarity/negation alignment.
    Flags pairs where one side has negation but the other doesn't.
    Only triggers if the evidence is strong (to avoid false positives).
    """
    # English negation markers
    en_neg_patterns = re.compile(
        r"\b(?:not|n't|never|no one|nobody|nothing|nowhere|neither|nor|cannot|can't|won't|don't|doesn't|didn't|isn't|aren't|wasn't|weren't|haven't|hasn't|hadn't)\b",
        re.IGNORECASE
    )
    # Japanese negation particles/endings
    ja_neg_patterns = re.compile(
        r'(?:ない|なかった|ません|ず|ぬ|なく|ないで|なければ|なきゃ|じゃない|ではない|ありません)'
    )

    en_has_neg = bool(en_neg_patterns.search(en))
    ja_has_neg = bool(ja_neg_patterns.search(ja))

    # Only flag if one side has negation but not the other
    # Be conservative: only flag when EN has strong negation but JA has none
    en_neg_count = len(en_neg_patterns.findall(en))
    if en_neg_count >= 2 and not ja_has_neg:
        return False, "negation_mismatch_en_has_ja_lacks"

    # Reverse check (JA has negation but EN doesn't) — less reliable, use weaker threshold
    ja_neg_count = len(ja_neg_patterns.findall(ja))
    if ja_neg_count >= 2 and not en_has_neg:
        return False, "negation_mismatch_ja_has_en_lacks"

    return True, ""


def check_encoding_artifacts(en: str, ja: str) -> tuple[bool, str]:
    """Reject pairs with encoding artifacts or garbage text."""
    # Unicode replacement character
    if '\uFFFD' in en or '\uFFFD' in ja:
        return False, "replacement_char"

    # Excessive punctuation (spam indicator)
    if re.search(r'[!?]{4,}', en) or re.search(r'[！？]{4,}', ja):
        return False, "excessive_punctuation"

    # URL-heavy text (not useful for general translation)
    url_pattern = re.compile(r'https?://\S+', re.IGNORECASE)
    en_urls = len(url_pattern.findall(en))
    en_words = len(en.split())
    if en_words > 0 and en_urls / max(1, en_words) > 0.3:
        return False, "url_heavy"

    return True, ""


# ──────────────────────────────────────────────────────────────────────────────
# Near-Deduplication via SimHash
# ──────────────────────────────────────────────────────────────────────────────

def compute_simhash(text: str, ngram_size: int = 3) -> int:
    """
    Compute a 64-bit SimHash fingerprint for near-duplicate detection.
    Uses character n-grams for language-agnostic hashing.
    """
    v = [0] * 64
    text_clean = text.lower().replace(" ", "")

    for i in range(max(1, len(text_clean) - ngram_size + 1)):
        ngram = text_clean[i:i + ngram_size]
        h = int(hashlib.md5(ngram.encode('utf-8')).hexdigest(), 16)
        for j in range(64):
            if h & (1 << j):
                v[j] += 1
            else:
                v[j] -= 1

    fingerprint = 0
    for j in range(64):
        if v[j] > 0:
            fingerprint |= (1 << j)
    return fingerprint


def hamming_distance(a: int, b: int) -> int:
    """Count differing bits between two 64-bit integers."""
    return bin(a ^ b).count('1')


# ──────────────────────────────────────────────────────────────────────────────
# Main Filtering Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_filter(input_path: str, output_path: str, cfg: dict, enable_simhash: bool = True):
    """
    Run the full semantic filtering pipeline on a TSV dataset.

    Expected format: col[0-1] = metadata, col[2] = score, col[3] = EN, col[4] = JA
    """
    if not os.path.exists(input_path):
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    # Statistics counters
    stats = defaultdict(int)
    stats["total_lines"] = 0

    # Deduplication sets
    seen_exact = set()
    seen_simhash: dict[int, int] = {}  # simhash -> line index (for near-dedup)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    logger.info("Starting semantic filtering pipeline...")
    logger.info("  Input:  %s", input_path)
    logger.info("  Output: %s", output_path)
    logger.info("  Config: %s", cfg)

    with open(input_path, 'r', encoding='utf-8', errors='replace') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:

        for line in fin:
            stats["total_lines"] += 1

            parts = line.strip('\n').split('\t')
            if len(parts) < 5:
                stats["rejected_malformed"] += 1
                continue

            # Score check
            try:
                score = float(parts[2])
                if score < cfg["min_score"]:
                    stats["rejected_low_score"] += 1
                    continue
            except ValueError:
                stats["rejected_malformed"] += 1
                continue

            # Clean text
            en = clean_text(parts[3])
            ja = clean_text(parts[4])

            if not en or not ja:
                stats["rejected_empty"] += 1
                continue

            # Filter 1: Length constraints
            ok, reason = check_length(en, ja, cfg)
            if not ok:
                stats[f"rejected_{reason}"] += 1
                continue

            # Filter 2: Language/script verification
            ok, reason = check_language_script(en, ja, cfg)
            if not ok:
                stats[f"rejected_{reason}"] += 1
                continue

            # Filter 3: Encoding artifacts
            ok, reason = check_encoding_artifacts(en, ja)
            if not ok:
                stats[f"rejected_{reason}"] += 1
                continue

            # Filter 4: Negation consistency (only in strict mode)
            if cfg.get("strict", False):
                ok, reason = check_negation_consistency(en, ja)
                if not ok:
                    stats[f"rejected_{reason}"] += 1
                    continue

            # Filter 5: Exact deduplication
            dedup_key = f"{en}|||{ja}"
            if dedup_key in seen_exact:
                stats["rejected_exact_dup"] += 1
                continue
            seen_exact.add(dedup_key)

            # Filter 6: Near-deduplication via SimHash (EN side)
            if enable_simhash:
                en_hash = compute_simhash(en)
                is_near_dup = False
                for existing_hash in list(seen_simhash.keys()):
                    if hamming_distance(en_hash, existing_hash) <= cfg["simhash_threshold"]:
                        is_near_dup = True
                        break
                if is_near_dup:
                    stats["rejected_near_dup"] += 1
                    continue
                seen_simhash[en_hash] = stats["total_lines"]

                # Limit simhash memory: keep only last 500K hashes
                if len(seen_simhash) > 500_000:
                    keys = list(seen_simhash.keys())
                    for k in keys[:100_000]:
                        del seen_simhash[k]

            # All filters passed — write to output
            parts[3] = en
            parts[4] = ja
            fout.write('\t'.join(parts) + '\n')
            stats["kept"] += 1

            # Progress logging
            if stats["total_lines"] % 500_000 == 0:
                logger.info(
                    "  Progress: %s lines processed, %s kept (%.1f%%)",
                    f"{stats['total_lines']:,}",
                    f"{stats['kept']:,}",
                    stats['kept'] / max(1, stats['total_lines']) * 100,
                )

    # Final report
    logger.info("")
    logger.info("=" * 60)
    logger.info("SEMANTIC FILTERING REPORT")
    logger.info("=" * 60)
    logger.info("Total lines processed: %s", f"{stats['total_lines']:,}")
    logger.info("Lines kept:           %s (%.1f%%)",
                f"{stats['kept']:,}",
                stats['kept'] / max(1, stats['total_lines']) * 100)
    logger.info("")
    logger.info("Rejection breakdown:")
    for key, count in sorted(stats.items()):
        if key.startswith("rejected_"):
            reason = key.replace("rejected_", "")
            logger.info("  %-35s %s", reason, f"{count:,}")
    logger.info("=" * 60)

    return stats


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Semantic data quality filter for EN-JA parallel corpora.")
    p.add_argument("--input", required=True, help="Input TSV file path")
    p.add_argument("--output", required=True, help="Output filtered TSV file path")
    p.add_argument("--min_score", type=float, default=DEFAULT_CONFIG["min_score"],
                   help=f"Minimum BiCleaner score (default: {DEFAULT_CONFIG['min_score']})")
    p.add_argument("--strict", action="store_true",
                   help="Enable strict mode (includes negation consistency check)")
    p.add_argument("--no_simhash", action="store_true",
                   help="Disable SimHash near-deduplication (faster, less thorough)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = {**DEFAULT_CONFIG, "min_score": args.min_score, "strict": args.strict}
    run_filter(
        input_path=args.input,
        output_path=args.output,
        cfg=cfg,
        enable_simhash=not args.no_simhash,
    )
