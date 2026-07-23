# Bidirectional English-Japanese Neural Machine Translation (EN ↔ JA)

An end-to-end Neural Machine Translation (NMT) system built from scratch using **PyTorch**, optimized for bidirectional translation between English (EN) and Japanese (JA). The project features a Pre-LN Transformer architecture with FlashAttention, dynamic Japanese register style control (`<2ja_polite>`, `<2ja_plain>`), a custom Metaspace BPE tokenizer, and dynamic memory-aware chunk training for constrained hardware environments.

---

## Key Features

* **Pre-LN Transformer Architecture:** Implements a 6-layer Encoder and 6-layer Decoder with Pre-LayerNorm (Pre-LN) for improved gradient stability during deep network training.
* **FlashAttention (SDPA):** Leverages PyTorch's native `torch.nn.functional.scaled_dot_product_attention` for fast, memory-efficient multi-head attention.
* **3-Way Weight Sharing:** Shares weights between the Encoder embedding, Decoder embedding, and Decoder linear output head (`fc_out`) to drastically reduce total parameters and memory footprint.
* **Japanese Register Control:** Supports target direction indicator tokens (`<2en>`, `<2ja>`) as well as dynamic Japanese honorific/register style tags (`<2ja_polite>` for *desu/masu* style and `<2ja_plain>` for *da/daru* style).
* **Metaspace BPE Tokenizer (v3):** Employs a single unified vocabulary (`shared_tokenizerv3.json`, size **50,002**) using BPE with a Metaspace sentinel (`▁` U+2581) to preserve exact word boundaries without text corruption.
* **6GB VRAM Optimization:** Designed for lightweight training budgets (e.g., NVIDIA RTX 2060 ~2.0 GB active VRAM) using Automatic Mixed Precision (AMP), bucket-batched `IterableDataset`, and explicit CUDA cache cleanup between corpus chunks.
* **Data Processing & Augmentation Pipeline:** Includes corpus line-slicing, quality threshold filtering, semantic similarity filtering, and targeted pronoun/idiom augmentation scripts.
* **Industrial Evaluation:** Built-in evaluation supporting SacreBLEU (utilizing `Fugashi`/MeCab for Japanese morphological segmentation), chrF2, TER, COMET (`wmt22-comet-da`), and training loss visualization.

---

## Architectural Specifications & Hyperparameters

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Architecture** | Pre-LN Transformer | Encoder-Decoder Stack |
| **`d_model`** | 512 | Model hidden dimension |
| **`num_layers`** | 6 Encoder / 6 Decoder | Total layer depth |
| **`num_heads`** | 8 | Attention heads (64 dims/head) |
| **`d_ff`** | 2048 | Feed-Forward inner dimension |
| **`max_length`** | 128 | Maximum sequence length |
| **`vocab_size`** | 50,002 | Extended BPE Tokenizer v3 vocabulary |
| **Batch Size** | 32 | Gradient accumulation steps: 4 (effective batch size: 128) |
| **Optimizer** | AdamW | `lr=5e-4`, `betas=(0.9, 0.98)`, `eps=1e-9`, `weight_decay=1e-4` |
| **Scheduler** | Cosine Warmup | `LambdaLR` with linear warmup and cosine decay |
| **Mixed Precision** | PyTorch 2.0 AMP | `torch.amp.autocast('cuda')` & `torch.amp.GradScaler('cuda')` |

---

## Project Structure

*(Note: Internal data outputs, trained weights, log reports, and local environment files are excluded per `.gitignore`)*

```text
.
├── AGENTS.md                           # Machine Translation Engineer Agent rules & architecture guidelines
├── README.md                           # Project documentation
├── train.py                            # Entrypoint for chunked multi-epoch training & checkpointing
├── data/
│   ├── data-manage/                    # Data preprocessing, filtering & augmentation suite
│   │   ├── chunk-split.py              # Splits parallel corpora into manageable chunk files
│   │   ├── data-extract-and-shuffle.py # Dataset line-slicing and randomizing utility
│   │   ├── filter-data-en-ja.py        # Noise elimination filter based on quality score thresholding
│   │   ├── idiom_augmentation.py       # Idiom-focused dataset augmentation utility
│   │   ├── merge-data-en-ja.py         # Parallel file merging and structural alignment utility
│   │   ├── pronoun_augment.py          # Pronoun-focused dataset augmentation utility
│   │   └── semantic_filter.py          # Embedding-based semantic similarity alignment filter
│   └── tokenizer/                      # BPE Tokenizer training & migration tools
│       ├── extend_tokenizer.py         # Vocabulary migration tool (adds Japanese register tags)
│       ├── shared_tokenizerv3.json     # Active 50,002 vocabulary tokenizer configuration
│       ├── tokenizer.py                # Base BPE tokenizer training script
│       ├── tokenizer_data_check.py     # Tokenizer decoding verification and roundtrip test
│       └── tokenizer_v2.py             # BPE Trainer utilizing Metaspace sentinel (U+2581)
├── evaluate/                           # Model evaluation & visualization tools
│   ├── evaluate.py                     # Evaluation script (SacreBLEU with Fugashi JA, chrF2, TER, COMET)
│   └── loss_check.py                   # Chronological training loss curve visualizer
└── model/                              # Core PyTorch Transformer architecture
    ├── attention.py                    # Positional Encoding, FlashAttention (SDPA), and FeedForward
    ├── dataloader.py                   # Bucket-batched IterableDataset with automatic style detection
    ├── decoder.py                      # Pre-LN Decoder Block Stack
    ├── encoder.py                      # Pre-LN Encoder Block Stack
    └── transformer.py                  # Full Transformer module with 3-way weight sharing
```

---

## Vocabulary & Special Tokens

The active tokenizer configuration (`data/tokenizer/shared_tokenizerv3.json`) maps the following special tokens:

* `<pad>` (0): Padding token
* `<s>` (1): Beginning of sequence token
* `</s>` (2): End of sequence token
* `<unk>` (3): Unknown token
* `<mask>` (4): Mask token
* `<2en>` (5): Target English translation indicator
* `<2ja>` (6): Target Japanese translation indicator (generic)
* `<2ja_polite>` (7): Target Japanese polite style indicator (*Desu/Masu*)
* `<2ja_plain>` (8): Target Japanese plain style indicator (*Da/Daru*)

---

## Workflow & Usage

### 1. Data Preparation & Tokenization
To train or update the BPE Tokenizer using Metaspace pre-tokenization:
```bash
python data/tokenizer/tokenizer_v2.py
```
To extend an existing tokenizer config with Japanese register style tokens:
```bash
python data/tokenizer/extend_tokenizer.py
```
To verify tokenizer encoding/decoding:
```bash
python data/tokenizer/tokenizer_data_check.py
```

### 2. Model Training
Run chunk-based training across training sets. Checkpoints are automatically saved to `checkpoints/` every 1,000 batches and at chunk completion:
```bash
python train.py
```

### 3. Model Evaluation
To evaluate model checkpoints using SacreBLEU, chrF2, TER, and COMET metrics:
```bash
python evaluate/evaluate.py
```

### 4. Loss Curve Visualization
To plot and inspect loss curves from saved checkpoints:
```bash
python evaluate/loss_check.py
```

---

## Future Roadmap: Multilingual Extension (TH ↔ EN ↔ JA)

Planned future development includes scaling this NMT architecture into a **Trilingual Multilingual Neural Machine Translation** system supporting **Thai (TH)**, **English (EN)**, and **Japanese (JA)**.

* **Trilingual Tokenizer Expansion:** Extending the shared Metaspace BPE vocabulary to incorporate Thai subword units alongside special direction indicator tokens (e.g., `<2th>`).
* **Any-to-Any Direct Translation:** Enabling direct supervised and zero-shot translations across 6 translation directions:
  - $\text{EN} \leftrightarrow \text{JA}$
  - $\text{TH} \leftrightarrow \text{EN}$
  - $\text{TH} \leftrightarrow \text{JA}$
* **Data & Segmentation Pipeline Integration:** Integrating specialized Thai morphological/word segmentation (e.g., PyThaiNLP), parallel corpus cleaning, and semantic alignment filtering for Thai parallel datasets.

