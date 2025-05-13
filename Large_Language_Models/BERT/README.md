# BERT 110M — Implementation from Scratch

This repository provides a **from-scratch implementation of the BERT (Bidirectional Encoder Representations from Transformers) 110M** parameter model using PyTorch. The goal is to replicate and understand the internal workings of the original BERT-base model, released by Google in 2018, which has since become foundational in modern NLP.

---

## 📌 What is BERT?

**BERT** is a transformer-based language representation model designed to pre-train deep bidirectional representations by jointly conditioning on both left and right context in all layers. This is different from traditional language models which are unidirectional.

The BERT-110M model specifically refers to the **BERT-Base** architecture, which has:

| Model        | Hidden Size | Layers | Attention Heads | Parameters |
|--------------|-------------|--------|------------------|------------|
| **BERT-Base** (this repo) | 768         | 12     | 12               | 110M       |

---

## 🧠 BERT Architecture — Detailed Overview

### 1. **Input Embeddings**
- Token Embeddings (WordPiece tokenizer)
- Segment Embeddings (for sentence pair tasks)
- Positional Embeddings

Final embedding = sum of all three

### 2. **Transformer Encoder Stack**
- **12 Encoder Layers (Transformer blocks)**:
  - Each has:
    - Multi-Head Self Attention
    - Add & LayerNorm
    - Feedforward Network (FFN)
    - Add & LayerNorm again
- **Multi-Head Self Attention**:
  - 12 heads, each with dimension 64 (768/12)
- **Feedforward Network**:
  - 2 Linear layers with GELU activation in between
  - Hidden dimension = 3072

### 3. **Pretraining Objectives**
- **Masked Language Modeling (MLM)**:
  - Randomly mask 15% of input tokens and predict them
- **Next Sentence Prediction (NSP)**:
  - Predict if two segments are contiguous

---

## 🧪 Implementation Features

- ✅ WordPiece tokenizer
- ✅ Configurable model dimensions (base config = 110M)
- ✅ Training loop with pretraining objectives (MLM + NSP)
- ✅ Evaluation pipeline
- ✅ Easy extension for finetuning tasks (QA, classification, etc.)
- ✅ HuggingFace compatibility for loading/saving weights

---

## 🧬 Variants of BERT

| Model             | Layers | Hidden Size | Heads | Parameters | Notable Features |
|------------------|--------|-------------|-------|------------|------------------|
| **BERT-Base**     | 12     | 768         | 12    | 110M       | Original base model |
| **BERT-Large**    | 24     | 1024        | 16    | 340M       | More powerful, but slower |
| **DistilBERT**    | 6      | 768         | 12    | 66M        | Smaller, faster, 97% performance |
| **TinyBERT**      | 4      | 312         | 12    | ~15M       | Designed for edge devices |
| **ALBERT**        | 12     | 768 (shared) | 12    | ~12M       | Parameter sharing + factorized embeddings |
| **RoBERTa**       | 12     | 768         | 12    | 125M       | Trained on more data, no NSP |
| **SpanBERT**      | 12     | 768         | 12    | 110M       | Span-level masking |
| **BioBERT**       | 12     | 768         | 12    | 110M       | Trained on biomedical corpora |

---

## 🔍 BERT vs Other Famous LLMs

| Model        | Type         | Parameters | Context Size | Training Objective     | Notes                        |
|--------------|--------------|------------|---------------|-------------------------|------------------------------|
| **BERT-Base**| Encoder-only | 110M       | 512 tokens    | MLM + NSP               | Bidirectional pretraining    |
| **GPT-2**    | Decoder-only | 117M+      | 1024 tokens   | Left-to-right LM        | Autoregressive generation    |
| **GPT-3**    | Decoder-only | 175B       | 2048 tokens   | Left-to-right LM        | Few-shot learning            |
| **T5**       | Encoder-Decoder | 220M+    | 512 tokens    | Text-to-text (span denoising) | Unified multitask framework |
| **BART**     | Encoder-Decoder | 140M+    | 1024 tokens   | Denoising autoencoding  | Seq2seq, good for generation |
| **RoBERTa**  | Encoder-only | 125M       | 512 tokens    | MLM (no NSP)            | More data, better performance |

---

## 🚀 Quickstart

```bash
# Clone the repo
git clone https://github.com/your-username/bert-110m-from-scratch.git
cd bert-110m-from-scratch

# Install dependencies
pip install -r requirements.txt

# Run pretraining (example)
python train.py --config configs/bert_base.json
```

---

## 📁 Repository Structure

```
bert-110m/
├── model/                  # BERT model implementation
│   ├── embedding.py
│   ├── transformer.py
│   └── bert.py
├── tokenizer/              # WordPiece tokenizer
├── data/                   # Dataset preprocessing
├── train.py                # Pretraining pipeline
├── utils.py
└── configs/
    └── bert_base.json
```

---

## 📖 References

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [The Illustrated BERT](https://jalammar.github.io/illustrated-bert/)
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)

---

## 🛠️ TODO

- [ ] Add finetuning scripts (QA, classification)
- [ ] Add support for mixed-precision training
- [ ] Implement BERT-Large variant
- [ ] Integrate with HuggingFace Model Hub

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## 🧑‍💻 License

This project is licensed under the MIT License.
