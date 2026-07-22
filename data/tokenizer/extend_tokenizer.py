"""
extend_tokenizer.py
═══════════════════
One-time migration utility that extends shared_tokenizerv2.json with style tags
and provides checkpoint embedding resize for seamless model migration.

Usage:
    # Extend tokenizer only
    python data/tokenizer/extend_tokenizer.py

    # Extend tokenizer + migrate a checkpoint
    python data/tokenizer/extend_tokenizer.py --checkpoint checkpoints/transformer_epoch_10_chunk_4.pt
"""

import os
import sys
import json
import copy
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

INPUT_TOKENIZER  = "data/tokenizer/shared_tokenizerv2.json"
OUTPUT_TOKENIZER = "data/tokenizer/shared_tokenizerv3.json"

# New style tokens to add
NEW_STYLE_TOKENS = ["<2ja_polite>", "<2ja_plain>"]

# Existing special tokens (for reference — IDs 0–6)
# 0: <pad>, 1: <s>, 2: </s>, 3: <unk>, 4: <mask>, 5: <2en>, 6: <2ja>


# ──────────────────────────────────────────────────────────────────────────────
# Tokenizer Extension
# ──────────────────────────────────────────────────────────────────────────────

def extend_tokenizer(input_path: str, output_path: str, new_tokens: list[str]) -> dict:
    """
    Add new special tokens to an existing HuggingFace tokenizer JSON file.

    The tokens are appended after the last existing token, preserving all
    existing token IDs (backward compatible).

    Returns a dict with the new token → ID mapping.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input tokenizer not found: {input_path!r}")

    logger.info("Loading tokenizer from %s", input_path)
    with open(input_path, "r", encoding="utf-8") as f:
        tok_data = json.load(f)

    vocab = tok_data["model"]["vocab"]
    current_size = len(vocab)
    logger.info("Current vocabulary size: %d", current_size)

    # Check which tokens already exist
    new_mapping = {}
    tokens_to_add = []
    for token in new_tokens:
        if token in vocab:
            logger.info("Token %r already exists at ID %d — skipping", token, vocab[token])
            new_mapping[token] = vocab[token]
        else:
            tokens_to_add.append(token)

    if not tokens_to_add:
        logger.info("All requested tokens already exist. No changes needed.")
        if input_path != output_path:
            # Still copy the file so v3 exists
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(tok_data, f, ensure_ascii=False)
            logger.info("Copied tokenizer to %s (no changes)", output_path)
        return new_mapping

    # Add new tokens to vocabulary
    next_id = current_size
    for token in tokens_to_add:
        vocab[token] = next_id
        new_mapping[token] = next_id
        logger.info("Added token %r → ID %d", token, next_id)
        next_id += 1

    # Also add to the added_tokens list if it exists
    if "added_tokens" in tok_data:
        for token in tokens_to_add:
            tok_data["added_tokens"].append({
                "id": new_mapping[token],
                "content": token,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            })

    # Save the extended tokenizer
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tok_data, f, ensure_ascii=False)

    new_size = len(vocab)
    logger.info("Extended vocabulary: %d → %d tokens", current_size, new_size)
    logger.info("Saved extended tokenizer → %s", output_path)

    return new_mapping


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint Embedding Resize
# ──────────────────────────────────────────────────────────────────────────────

def resize_checkpoint_embeddings(
    checkpoint_path: str,
    old_vocab_size: int,
    new_vocab_size: int,
    output_path: str = None,
) -> None:
    """
    Resize embedding and output projection layers in a saved checkpoint
    to accommodate new tokens. New embeddings are initialized with the
    mean of existing embeddings (better than random for quick fine-tuning).

    Parameters
    ----------
    checkpoint_path : str
        Path to the .pt checkpoint file.
    old_vocab_size : int
        The original vocabulary size (should match the checkpoint).
    new_vocab_size : int
        The target vocabulary size after extension.
    output_path : str, optional
        Where to save the migrated checkpoint. If None, overwrites in-place.
    """
    import torch

    if output_path is None:
        output_path = checkpoint_path

    logger.info("Loading checkpoint from %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint["model_state_dict"]

    # Keys that need resizing (shared embeddings architecture):
    # - encoder.embedding.weight  (shared with decoder.embedding.weight)
    # - decoder.embedding.weight  (shared — same tensor)
    # - decoder.fc_out.weight     (shared — same tensor)
    # Because of weight sharing, we only need to resize one and the others
    # will reference the same tensor. But since state_dict stores separate
    # copies, we resize all three.

    embedding_keys = [
        "encoder.embedding.weight",
        "decoder.embedding.weight",
        "decoder.fc_out.weight",
    ]

    for key in embedding_keys:
        if key not in state_dict:
            logger.warning("Key %r not found in state_dict — skipping", key)
            continue

        old_weight = state_dict[key]
        if old_weight.shape[0] == new_vocab_size:
            logger.info("Key %r already has size %d — no resize needed", key, new_vocab_size)
            continue

        if old_weight.shape[0] != old_vocab_size:
            logger.warning(
                "Key %r has shape %s (expected [%d, *]). Skipping.",
                key, old_weight.shape, old_vocab_size
            )
            continue

        d_model = old_weight.shape[1]
        # Initialize new embeddings with the mean of existing ones
        # (closer to the embedding distribution than random init)
        mean_embedding = old_weight.mean(dim=0, keepdim=True)
        num_new = new_vocab_size - old_vocab_size
        new_rows = mean_embedding.expand(num_new, d_model).clone()

        new_weight = torch.cat([old_weight, new_rows], dim=0)
        state_dict[key] = new_weight
        logger.info(
            "Resized %r: [%d, %d] → [%d, %d]",
            key, old_vocab_size, d_model, new_vocab_size, d_model
        )

    # Save the migrated checkpoint
    checkpoint["model_state_dict"] = state_dict
    checkpoint["vocab_size_migrated_from"] = old_vocab_size
    checkpoint["vocab_size_migrated_to"] = new_vocab_size
    torch.save(checkpoint, output_path)
    logger.info("Saved migrated checkpoint → %s", output_path)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extend tokenizer with style tags and optionally migrate a checkpoint."
    )
    p.add_argument(
        "--input_tokenizer", default=INPUT_TOKENIZER,
        help=f"Path to the source tokenizer JSON (default: {INPUT_TOKENIZER})"
    )
    p.add_argument(
        "--output_tokenizer", default=OUTPUT_TOKENIZER,
        help=f"Path for the extended tokenizer JSON (default: {OUTPUT_TOKENIZER})"
    )
    p.add_argument(
        "--checkpoint", default=None,
        help="Path to a .pt checkpoint to resize embeddings (optional)"
    )
    p.add_argument(
        "--checkpoint_output", default=None,
        help="Output path for the migrated checkpoint (default: overwrite in-place)"
    )
    p.add_argument(
        "--old_vocab_size", type=int, default=50000,
        help="Original vocabulary size (default: 50000)"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Step 1: Extend tokenizer
    new_mapping = extend_tokenizer(
        input_path=args.input_tokenizer,
        output_path=args.output_tokenizer,
        new_tokens=NEW_STYLE_TOKENS,
    )

    new_vocab_size = args.old_vocab_size + len([t for t in NEW_STYLE_TOKENS if t in new_mapping and new_mapping[t] >= args.old_vocab_size])
    logger.info("New vocab size: %d", new_vocab_size)

    # Step 2: Optionally migrate checkpoint
    if args.checkpoint:
        resize_checkpoint_embeddings(
            checkpoint_path=args.checkpoint,
            old_vocab_size=args.old_vocab_size,
            new_vocab_size=new_vocab_size,
            output_path=args.checkpoint_output,
        )

    logger.info("Done! Summary:")
    for token, tid in new_mapping.items():
        logger.info("  %s → ID %d", token, tid)
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Update train.py VOCAB_SIZE from %d to %d", args.old_vocab_size, new_vocab_size)
    logger.info("  2. Update SRC_TOKENIZER / TGT_TOKENIZER to point to %s", args.output_tokenizer)
    logger.info("  3. If migrating a checkpoint, use --checkpoint flag")
