## Bidirectional English-Japanese Transformer Machine Translation
An end-to-end Machine Translation (MT) ecosystem built from scratch using PyTorch. 
This project implements a vanilla Transformer architecture optimized for bidirectional translation between English (EN) and Japanese (JA), featuring an advanced data preprocessing pipeline, customized tokenization, and multi-metric evaluation capabilities.

**Project Status: Under Development**
> This project is currently in its development phase.
> The core architecture and data pipeline are fully operational, but ongoing optimizations include extending hyperparameter tuning, scaling dataset sizes, and refining decoding strategies.

---

## Key Features
* **Custom Transformer from Scratch:** Implements a full Encoder-Decoder Transformer utilizing a **Pre-LayerNorm (Pre-LN)** architecture for smoother gradient flows in deep networks.
* **High-Performance Attention:** Built-in support for **FlashAttention** via PyTorch's `scaled_dot_product_attention` to drastically boost training efficiency.
* **Bidirectional Capability:** Translates both EN $\rightarrow$ JA and JA $\rightarrow$ EN using a single target-language indicator token system (`<2en>`, `<2ja>`).
* **Robust Data Pipeline:** Features an optimized `IterableDataset` with dynamic tokenization, chunk filtering based on corpus quality alignment scores, and cross-platform multiprocessing safeguards (Windows spawn vs Linux/Mac fork handling).
* **Advanced Tokenization:** Utilizes Hugging Face's BPE Tokenizer, customized with `UnicodeScripts` and `Whitespace` splitting to correctly handle complex Japanese boundaries alongside English syntax without turning text into byte garbage.
* **Industrial Evaluation Metrics:** Integrates multiple translation metrics including **BLEU** (with auto-fallback tokenizers), **chrF2**, **TER** via `SacreBLEU`, and neural evaluation using **COMET** (`wmt22-comet-da`).

---

## Project Structure
```text
├── data/                              # Data preparation & Tokenizer tools
│   ├── data_management/
│      ├── data_extract_and_shuffle.py # Dataset line-slicing and randomizing utility
│      ├── filter_data.py              # Noise elimination script based on quality thresholding
│      └── merge_dataset.py            # Multi-file matching and structural alignment utility
│   ├── tokenizer/               
│      ├── tokenizer_data_check.py    # Tokenizer verification and decoding tests
│      ├── tokenizer.py               # BPE Training script with custom pre-tokenization
├── evaluate/                       # Model evaluation scripts
│   ├── evaluate.py                 # Main validation script (BLEU, chrF2, TER, COMET)
│   └── loss_check.py               # Epoch-wise training curve visualizer
├── model/                          # Transformer architecture components
│   ├── attention.py                # Positional Encoding, Flash Multi-Head Attention, FeedForward
│   ├── dataloader.py               # Iterable Dataset and custom parallel DataLoader
│   ├── encoder.py                  # Pre-LN Encoder Layers and Block Stack
│   ├── decoder.py                  # Pre-LN Decoder Layers and Block Stack
│   └── transformer.py              # Core Transformer module with weight/embedding sharing
├── checkpoints/                    # Saved model checkpoints (not uploaded)
└── train.py                        # Main training execution script
```
