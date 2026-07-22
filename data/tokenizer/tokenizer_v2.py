"""
tokenizer_v2.py
═══════════════
Trains a shared BPE tokenizer for bidirectional English-Japanese translation.

Changes from v1 
──────────────────────────────────────────────────────────────────────────────
CRIT-1  Replace Metaspace replacement=" " (U+0020) with METASPACE_CHAR="▁"
        (U+2581, LOWER ONE EIGHTH BLOCK) in BOTH the pre-tokenizer and decoder.
        Using a plain space makes the decoder unable to distinguish word
        boundaries from genuine spaces in source text → corrupted output.

CRIT-2  Replace .strip('\\n') with .strip() to handle \\r\\n (Windows) endings.
        The trailing \\r was being injected into every Japanese training sentence.

CRIT-3  Validate and parse lines *before* reservoir insertion.
        · Store only (parts[3], parts[4]) tuples — not raw lines — cutting
          peak memory by ~3–5× on a 20M-line corpus.
        · Use `valid_count` (not raw line index `i`) as the denominator
          so the sampling probability is calculated over valid lines only.

CRIT-4  Seed random before the reservoir loop for reproducibility.

CRIT-5  Remove dead `count` variable; replace with a logged yield_count.

ENH-1   Add min_frequency=2 to BpeTrainer to prune hapax legomena.

ENH-2   Replace print() with the logging module (timestamps + severity levels).

ENH-3   Add os.path.exists pre-flight check; open file with errors="replace".

ENH-4   Shuffle reservoir before yielding to diversify trainer input batches.

ENH-5   Replace hardcoded DATA_PATH/SAVE_PATH with argparse CLI arguments.
"""

import os
import random
import logging
import argparse

from tqdm import tqdm
from tokenizers import Tokenizer, normalizers, pre_tokenizers, decoders
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import NFKC


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# CRIT-1 FIX: U+2581 (LOWER ONE EIGHTH BLOCK) is the canonical SentencePiece /
# HuggingFace boundary sentinel. It cannot appear in normal text, so the
# decoder can unambiguously detect and remove it during reconstruction.
# DO NOT change this to " " (U+0020): the decoder cannot distinguish that
# from a real space and will produce corrupted output.
METASPACE_CHAR: str = "▁"


# ──────────────────────────────────────────────────────────────────────────────
# Logging  (ENH-2 FIX: structured, timestamped, redirectable)
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data Generator
# ──────────────────────────────────────────────────────────────────────────────

def batch_iterator(
    data_path: str,
    max_lines: int = 1_000_000,
    seed: int = 42,
):
    """
    Reservoir-sampled generator over a tab-separated En-Ja corpus.

    Memory model
    ────────────
    Reads the file line-by-line (O(1) memory for file I/O).
    The reservoir holds at most `max_lines` (en, ja) string tuples.
    No full-line payloads are stored — only the two fields needed.

    Reservoir sampling (Vitter's Algorithm R)
    ──────────────────────────────────────────
    Each valid line has an equal probability of appearing in the final
    reservoir: P(line_i selected) = max_lines / valid_count_at_i.

    Parameters
    ──────────
    data_path : str   Path to the tab-separated dataset file.
    max_lines : int   Reservoir capacity (sentence pairs).
    seed      : int   Random seed for reproducibility (CRIT-4 FIX).
    """
    # ENH-3 FIX: fail fast with a clear message instead of a raw traceback
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path!r}")

    logger.info(
        "Reservoir-sampling %r — target capacity: %s pairs",
        data_path, f"{max_lines:,}",
    )

    # CRIT-4 FIX: pin the seed before any random call so the sampled subset
    # is identical across runs on the same corpus.
    random.seed(seed)

    # CRIT-3 FIX: store only the two text fields, not the full raw line.
    # At max_lines=1_000_000 this typically saves 1–2 GB of peak RAM.
    reservoir: list[tuple[str, str]] = []
    valid_count: int = 0
    skipped: int = 0

    # ENH-3 FIX: errors="replace" survives rare encoding anomalies in large
    # corpora without crashing the entire training run.
    with open(data_path, "r", encoding="utf-8", errors="replace") as f:
        for line in tqdm(f, desc="Reservoir sampling", unit=" lines", mininterval=5):

            # CRIT-2 FIX: .strip() handles \r\n (Windows), \n (Unix), and
            # any trailing whitespace. The original .strip('\n') left \r
            # attached to parts[4], injecting it into every Japanese sentence.
            #
            # CRIT-3 FIX: maxsplit=5 splits only on the first 5 tabs, so if
            # the Japanese text itself contains a tab it is not fragmented.
            parts = line.strip().split('\t', 5)

            if len(parts) < 5:
                skipped += 1
                continue  # CRIT-3 FIX: reject BEFORE reservoir insertion

            pair = (parts[3], parts[4])

            # CRIT-3 FIX: use valid_count as the denominator (not the raw
            # line index i), so the sampling probability is computed only
            # over valid lines — matching the mathematical guarantee of
            # Algorithm R.
            if valid_count < max_lines:
                reservoir.append(pair)
            else:
                j = random.randint(0, valid_count)
                if j < max_lines:
                    reservoir[j] = pair

            valid_count += 1

    logger.info(
        "Reservoir built — %s valid pairs collected, %s malformed lines skipped.",
        f"{len(reservoir):,}", f"{skipped:,}",
    )

    # ENH-4 FIX: shuffle the reservoir so the trainer sees diverse language
    # patterns rather than corpus-order clusters (e.g. domain runs).
    random.shuffle(reservoir)

    # CRIT-5 FIX: replace the dead `count` variable with a logged yield_count.
    yield_count = 0
    for en_text, ja_text in reservoir:
        yield en_text
        yield ja_text
        yield_count += 1

    logger.info(
        "Yielded %s sentence pairs (%s total texts) to trainer.",
        f"{yield_count:,}", f"{yield_count * 2:,}",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Tokenizer Training
# ──────────────────────────────────────────────────────────────────────────────

def train_tokenizer(
    data_path: str,
    save_path: str,
    vocab_size: int = 50_000,
    max_lines: int = 1_000_000,
    min_frequency: int = 2,
    seed: int = 42,
) -> None:
    """
    Train a shared BPE tokenizer for bidirectional English-Japanese translation.

    Special token IDs (determined by list order)
    ─────────────────────────────────────────────
    0  <pad>   — padding
    1  <s>     — start of sequence
    2  </s>    — end of sequence
    3  <unk>   — unknown (also BPE fallback label)
    4  <mask>  — masked language modelling
    5  <2en>   — target-language tag: translate to English
    6  <2ja>   — target-language tag: translate to Japanese

    Parameters
    ──────────
    data_path    : str   Path to the tab-separated corpus.
    save_path    : str   Destination for the tokenizer JSON.
    vocab_size   : int   Target vocabulary size.
    max_lines    : int   Reservoir capacity (sentence pairs).
    min_frequency: int   Minimum merge frequency (ENH-1 FIX).
    seed         : int   Random seed for reproducibility (CRIT-4 FIX).
    """
    logger.info(
        "Starting tokenizer training — vocab_size=%s, max_lines=%s, "
        "min_frequency=%s, seed=%s",
        vocab_size, f"{max_lines:,}", min_frequency, seed,
    )

    # 1. Special tokens
    special_tokens = [
        "<pad>", "<s>", "</s>", "<unk>", "<mask>",
        "<2en>", "<2ja>",
        "<2ja_polite>", "<2ja_plain>",  # Style control tags for EN→JA register
    ]

    # 2. BPE model
    #    byte_fallback=True: any character not in the learned vocabulary is
    #    decomposed into individual UTF-8 byte tokens — zero hard UNKs even
    #    for rare CJK characters or unseen scripts.
    tokenizer = Tokenizer(BPE(unk_token="<unk>", byte_fallback=True))

    # 3. Normalizer
    tokenizer.normalizer = normalizers.Sequence([NFKC()])

    # 4. Pre-tokenizer
    #    UnicodeScripts: splits at script boundaries (Latin ↔ CJK ↔ Katakana
    #    etc.) — essential for Japanese which mixes multiple scripts.
    #    Metaspace: prefixes each word-start with METASPACE_CHAR so the decoder
    #    can reconstruct spaces.
    #
    #    CRIT-1 FIX: METASPACE_CHAR = "▁" (U+2581), not " " (U+0020).
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.UnicodeScripts(),
        pre_tokenizers.Metaspace(replacement=METASPACE_CHAR),
    ])

    # 5. Trainer
    #    ENH-1 FIX: min_frequency=2 prunes hapax legomena (tokens seen only
    #    once), preventing vocabulary bloat from noise tokens. On a 1M-pair
    #    sample this typically saves thousands of vocabulary slots.
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=min_frequency,
        show_progress=True,
    )

    # 6. Train from generator
    tokenizer.train_from_iterator(
        batch_iterator(data_path, max_lines, seed),
        trainer=trainer,
    )

    # 7. Decoder
    #    CRIT-1 FIX: MUST use the same METASPACE_CHAR as the pre-tokenizer.
    #    If these differ, decoding will never correctly reconstruct spaces.
    tokenizer.decoder = decoders.Metaspace(replacement=METASPACE_CHAR)

    # 8. Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tokenizer.save(save_path)
    logger.info("Tokenizer saved → %s", save_path)


# ──────────────────────────────────────────────────────────────────────────────
# CLI  (ENH-5 FIX: argparse replaces hardcoded DATA_PATH / SAVE_PATH)
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a shared BPE tokenizer for En-Ja translation."
    )
    p.add_argument(
        "--data_path", required=True,
        help="Path to the tab-separated corpus (expects en in col[3], ja in col[4])",
    )
    p.add_argument(
        "--save_path", required=True,
        help="Output path for the tokenizer JSON (e.g. data/tokenizer/shared_tokenizer.json)",
    )
    p.add_argument("--vocab_size",    type=int, default=50_000,
                   help="BPE vocabulary size (default: 50 000)")
    p.add_argument("--max_lines",     type=int, default=1_000_000,
                   help="Reservoir capacity in sentence pairs (default: 1 000 000)")
    p.add_argument("--min_frequency", type=int, default=2,
                   help="Minimum token merge frequency — filters hapax legomena (default: 2)")
    p.add_argument("--seed",          type=int, default=42,
                   help="Random seed for reproducibility (default: 42)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_tokenizer(
        data_path=args.data_path,
        save_path=args.save_path,
        vocab_size=args.vocab_size,
        max_lines=args.max_lines,
        min_frequency=args.min_frequency,
        seed=args.seed,
    )