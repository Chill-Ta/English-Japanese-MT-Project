"""
pronoun_augment.py
══════════════════
Generates augmentation data targeting the Japanese pro-drop problem in JA→EN.

Japanese frequently omits subjects (pro-drop), but English requires explicit
pronouns. The model currently fails on these cases, producing errors like:
    JA: "戻らなければならない" → EN: "It must be returned." (wrong!)
    Should be: "I/You/He must go back."

This script creates template-based parallel pairs that explicitly teach
pronoun restoration from zero-pronoun Japanese to English.

Usage:
    python data/data-manage/pronoun_augment.py
    python data/data-manage/pronoun_augment.py --output data/dataset/train-set/pronoun_pairs.txt --oversample 3
"""

import os
import random
import argparse
import logging
from itertools import product

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Pronoun Restoration Templates
# ──────────────────────────────────────────────────────────────────────────────
# Format: (ja_template, en_template, pronoun_variants)
# {subj_ja} and {subj_en} are placeholders for the subject.
# Templates cover common pro-drop patterns.

# Subject pronoun mappings: (ja_subject_or_empty, en_pronoun)
SUBJECT_PRONOUNS = [
    ("", "I"),
    ("", "you"),
    ("", "he"),
    ("", "she"),
    ("", "we"),
    ("", "they"),
    ("私は", "I"),
    ("あなたは", "you"),
    ("彼は", "he"),
    ("彼女は", "she"),
    ("私たちは", "we"),
    ("彼らは", "they"),
]

# Pro-drop pairs: (ja_without_explicit_subject, en_with_explicit_subject)
# These are structured so the JA side may or may not include the subject.
PRODROP_TEMPLATES = [
    # ── Must / Have to patterns ──
    # The "戻らなければならない" → "It must be returned" error class
    ("{subj_ja}戻らなければならない。", "{subj_en} must go back."),
    ("{subj_ja}行かなければならない。", "{subj_en} must go."),
    ("{subj_ja}食べなければならない。", "{subj_en} must eat."),
    ("{subj_ja}帰らなければならない。", "{subj_en} must go home."),
    ("{subj_ja}働かなければならない。", "{subj_en} must work."),
    ("{subj_ja}勉強しなければならない。", "{subj_en} must study."),
    ("{subj_ja}急がなければならない。", "{subj_en} must hurry."),
    ("{subj_ja}待たなければならない。", "{subj_en} must wait."),
    ("{subj_ja}謝らなければならない。", "{subj_en} must apologize."),
    ("{subj_ja}準備しなければならない。", "{subj_en} must prepare."),

    # ── Want to patterns ──
    ("{subj_ja}行きたい。", "{subj_en} want to go."),
    ("{subj_ja}食べたい。", "{subj_en} want to eat."),
    ("{subj_ja}帰りたい。", "{subj_en} want to go home."),
    ("{subj_ja}休みたい。", "{subj_en} want to rest."),
    ("{subj_ja}寝たい。", "{subj_en} want to sleep."),
    ("{subj_ja}遊びたい。", "{subj_en} want to play."),
    ("{subj_ja}買いたい。", "{subj_en} want to buy it."),
    ("{subj_ja}知りたい。", "{subj_en} want to know."),
    ("{subj_ja}試したい。", "{subj_en} want to try it."),
    ("{subj_ja}会いたい。", "{subj_en} want to meet."),

    # ── Past tense patterns (た form) ──
    ("{subj_ja}行った。", "{subj_en} went."),
    ("{subj_ja}食べた。", "{subj_en} ate."),
    ("{subj_ja}帰った。", "{subj_en} went home."),
    ("{subj_ja}見た。", "{subj_en} saw it."),
    ("{subj_ja}聞いた。", "{subj_en} heard it."),
    ("{subj_ja}書いた。", "{subj_en} wrote it."),
    ("{subj_ja}読んだ。", "{subj_en} read it."),
    ("{subj_ja}買った。", "{subj_en} bought it."),
    ("{subj_ja}忘れた。", "{subj_en} forgot."),
    ("{subj_ja}思い出した。", "{subj_en} remembered."),

    # ── Can / Able to patterns ──
    ("{subj_ja}泳げる。", "{subj_en} can swim."),
    ("{subj_ja}話せる。", "{subj_en} can speak."),
    ("{subj_ja}走れる。", "{subj_en} can run."),
    ("{subj_ja}できる。", "{subj_en} can do it."),
    ("{subj_ja}理解できる。", "{subj_en} can understand."),
    ("{subj_ja}信じられない。", "{subj_en} can't believe it."),

    # ── Negative patterns ──
    ("{subj_ja}行かない。", "{subj_en} won't go."),
    ("{subj_ja}食べない。", "{subj_en} won't eat."),
    ("{subj_ja}分からない。", "{subj_en} don't understand."),
    ("{subj_ja}知らない。", "{subj_en} don't know."),
    ("{subj_ja}覚えていない。", "{subj_en} don't remember."),
    ("{subj_ja}行きたくない。", "{subj_en} don't want to go."),

    # ── Thinking / Feeling patterns ──
    ("{subj_ja}嬉しい。", "{subj_en} am happy."),
    ("{subj_ja}悲しい。", "{subj_en} am sad."),
    ("{subj_ja}疲れた。", "{subj_en} am tired."),
    ("{subj_ja}怒っている。", "{subj_en} am angry."),
    ("{subj_ja}心配している。", "{subj_en} am worried."),
    ("{subj_ja}驚いた。", "{subj_en} was surprised."),
    ("{subj_ja}そう思う。", "{subj_en} think so."),
    ("{subj_ja}間違っていると思う。", "{subj_en} think it's wrong."),

    # ── Progressive / Ongoing ──
    ("{subj_ja}走っている。", "{subj_en} am running."),
    ("{subj_ja}待っている。", "{subj_en} am waiting."),
    ("{subj_ja}考えている。", "{subj_en} am thinking."),
    ("{subj_ja}探している。", "{subj_en} am looking for it."),
    ("{subj_ja}働いている。", "{subj_en} am working."),

    # ── Complex sentences with pro-drop ──
    ("{subj_ja}明日東京に行く。", "{subj_en} will go to Tokyo tomorrow."),
    ("{subj_ja}毎日運動している。", "{subj_en} exercise every day."),
    ("{subj_ja}来週までに終わらせる。", "{subj_en} will finish it by next week."),
    ("{subj_ja}昨日の夜遅くまで起きていた。", "{subj_en} stayed up late last night."),
    ("{subj_ja}最近忙しい。", "{subj_en} have been busy lately."),
    ("{subj_ja}日本語を勉強している。", "{subj_en} am studying Japanese."),
    ("{subj_ja}ここに住んでいる。", "{subj_en} live here."),
    ("{subj_ja}そこに行ったことがある。", "{subj_en} have been there before."),
    ("{subj_ja}すぐに戻る。", "{subj_en} will be right back."),
    ("{subj_ja}もう一度やり直す。", "{subj_en} will try again."),
]

# Verb conjugation fixups for third person singular
_THIRD_PERSON_FIXUPS = {
    # want/don't patterns
    " want ": " wants ",
    " don't ": " doesn't ",
    # am/is patterns
    " am ": " is ",
    # have patterns
    " have ": " has ",
    # exercise/live patterns — handled by regex
}


def fix_third_person(en: str, pronoun: str) -> str:
    """Apply third-person singular conjugation rules."""
    if pronoun.lower() not in ("he", "she", "it"):
        return en

    for old, new in _THIRD_PERSON_FIXUPS.items():
        en = en.replace(old, new)

    return en


def fix_plural(en: str, pronoun: str) -> str:
    """Fix conjugation for plural pronouns."""
    if pronoun.lower() not in ("we", "they"):
        return en

    # 'am' → 'are' for plural
    en = en.replace(" am ", " are ")
    return en


# ──────────────────────────────────────────────────────────────────────────────
# Generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_pronoun_corpus(
    output_path: str,
    oversample: int = 3,
    seed: int = 42,
    synthetic_score: float = 0.95,
    include_explicit_subjects: bool = True,
):
    """
    Generate pronoun restoration augmentation data.

    Creates pairs with both zero-pronoun JA and explicit-subject JA,
    all mapping to explicit-pronoun EN.

    Parameters
    ----------
    output_path : str
        Output TSV file path.
    oversample : int
        Repetition factor for each generated pair.
    seed : int
        Random seed.
    synthetic_score : float
        Synthetic BiCleaner score.
    include_explicit_subjects : bool
        If True, also generates pairs where JA has explicit subjects.
    """
    random.seed(seed)
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    all_pairs = []

    for ja_template, en_template in PRODROP_TEMPLATES:
        for subj_ja, subj_en in SUBJECT_PRONOUNS:
            # Skip if template expects explicit subject but we have none, and vice versa
            if not include_explicit_subjects and subj_ja:
                continue

            ja = ja_template.replace("{subj_ja}", subj_ja)
            en = en_template.replace("{subj_en}", subj_en)

            # Fix verb conjugation
            en = fix_third_person(en, subj_en)
            en = fix_plural(en, subj_en)

            all_pairs.append((en, ja))

    # Deduplicate
    all_pairs = list(set(all_pairs))

    # Oversample
    expanded = all_pairs * oversample
    random.shuffle(expanded)

    # Write in training data format
    total_written = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for en, ja in expanded:
            line = f"pronoun_aug\tpronoun_aug\t{synthetic_score:.3f}\t{en}\t{ja}\n"
            f.write(line)
            total_written += 1

    logger.info("Generated %d pronoun augmentation pairs → %s", total_written, output_path)
    logger.info("  Unique pairs: %d", len(all_pairs))
    logger.info("  Oversample factor: %d×", oversample)

    return total_written


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate pronoun restoration augmentation data for JA→EN.")
    p.add_argument("--output", default="data/dataset/train-set/pronoun_pairs.txt",
                   help="Output TSV file path")
    p.add_argument("--oversample", type=int, default=3,
                   help="Repetition factor (default: 3)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--score", type=float, default=0.95,
                   help="Synthetic BiCleaner score (default: 0.95)")
    p.add_argument("--no_explicit", action="store_true",
                   help="Only generate zero-pronoun JA pairs (skip explicit subjects)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_pronoun_corpus(
        output_path=args.output,
        oversample=args.oversample,
        seed=args.seed,
        synthetic_score=args.score,
        include_explicit_subjects=not args.no_explicit,
    )
