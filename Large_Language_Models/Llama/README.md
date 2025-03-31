
# Llama 3 - 8B

LLaMA (Large Language Model Meta AI) is a series of autoregressive transformer-based models developed by Meta AI. Designed for efficiency, LLaMA models use grouped-query attention, SwiGLU activation, and rotary embeddings to achieve strong performance with lower computational requirements.

## Comparison of LLaMA, GPT, BERT, and the Original Transformer  

| Feature                 | **LLaMA**                     | **GPT (Generative Pre-trained Transformer)** | **BERT (Bidirectional Encoder Representations from Transformers)** | **"Attention Is All You Need" (Original Transformer)** |  
|-------------------------|------------------------------|----------------------------------------------|----------------------------------------------|-------------------------------------|  
| **Architecture**        | Decoder-only Transformer    | Decoder-only Transformer                   | Encoder-only Transformer                   | Encoder-Decoder Transformer        |  
| **Pretraining**        | Causal LM (left-to-right)   | Causal LM (left-to-right)                  | Masked LM (MLM) + Next Sentence Prediction | Supervised Translation Task       |  
| **Training Data**      | Diverse web corpus          | WebText, BooksCorpus, Common Crawl         | BooksCorpus, Wikipedia                    | WMT English-German dataset         |  
| **Tokenization Scheme**| Sentence Piece Byte-level BPE | Optimized BPE (tiktoken)                     | 	WordPiece                     | Word/Character-Level            |
| **Attention Mechanism**| Grouped-Query Attention, Rotary Embeddings | Standard Self-Attention                     | Standard Self-Attention                     | Standard Self-Attention            |  
| **Efficiency Features**| SwiGLU activation, Optimized memory use | Standard FFN                               | Standard FFN                               | Standard FFN                       |  
| **Context Length**     | 2K-128K tokens (LLaMA 3)    | 2K-32K tokens (GPT-4 Turbo)                | 512-1024 tokens                           | ~512 tokens                         |  
| **Directionality**     | Autoregressive (unidirectional) | Autoregressive (unidirectional)            | Bidirectional                             | Encoder-Decoder                    |  
| **Multimodal Capabilities** | LLaVA (Vision-Language Models) | GPT-4V (Vision-Language)                   | No direct multimodal support               | No multimodal support               |  
| **Fine-tuning Approaches** | LoRA, Sparse MoE, Instruction Tuning | Instruction Tuning, RLHF                   | Fine-tuning, Knowledge Distillation       | Supervised Learning                 |  
| **Primary Use Cases**  | NLP, RAG, Multimodal AI, Code | Text generation, Chatbots, Summarization  | Text understanding, QA, Classification    | Machine Translation                 |  


## Key Theoretical Features
### 1. Transformer Architecture
LLaMA 3 follows the **decoder-only transformer** model, similar to GPT architectures. Key components include:
   - **Multi-head Self-Attention (MHSA):** Allows the model to focus on different parts of the input simultaneously.
   - **Query-Key-Value (QKV) Attention Mechanism:** It calculates attention scores by first computing the dot product of the query and key matrices, scaling the result by the square root of the key dimension, applying a softmax function to obtain attention weights, and then using these weights to compute a weighted sum of the value matrix.
   - **Feedforward Networks (FFN):** A two-layer MLP with activation functions (such as ReLU or SwiGLU) applied independently to each token position.
   - **Layer Normalization & Residual Connections:** Stabilizes training and prevents gradient issues by normalizing activations before applying non-linear transformations.

### 2. RMS Norm
   - LLaMA 3 replaces traditional Layer Normalization with **Root Mean Square Normalization (RMS Norm)**.
   - Computes the norm of the input vector and normalizes it, avoiding dependency on mean subtraction.
   - More stable than LayerNorm in large-scale training and avoids reliance on batch statistics.

### 3. Rotary Positional Embeddings (RoPE)
LLaMA 3 employs **Rotary Positional Embeddings (RoPE)** instead of traditional absolute positional encodings. Key advantages of RoPE:
   - **Better Long-Range Dependency Handling:** Unlike absolute position encodings, RoPE enables smooth token interactions over long sequences.
   - **Rotational Encoding in Complex Space:** Applies a rotation matrix to query and key embeddings to encode relative positional information.
   - **Scalability:** More efficient for long-context models compared to learned positional embeddings.

### 4. Tokenization Strategy
   - Uses **SentencePiece Byte-Pair Encoding** tokenizer
   - Tokenized input sequences are mapped to **embedding vectors** before being fed into the transformer layers.

### 5. Training Configuration
   - **Mixed Precision (`bfloat16`):** Optimizes memory usage without sacrificing much accuracy.
   - **Batch Size = 64:** Set for balancing memory consumption and training efficiency.
   - **Sequence Length = 2048:** Defines the maximum input context length for training.
   - **Weight Tying:** Shares embedding weights between the input and output layers to reduce the number of parameters.
   
### 6. Memory Management & Efficient Computation
   - **Activation Checkpointing:** Reduces memory consumption during training by recomputing activations on-the-fly.
   - **Flash Attention:** Implements an optimized attention computation technique that reduces memory overhead.

### 7. SwiGLU Activation
   - Uses **SwiGLU (Swish-Gated Linear Units)** instead of ReLU in feedforward networks.
   - Enhances expressiveness and enables smoother optimization paths.

### 9. Grouped Query Attention (GQA) with KV Cache
   - Implements **Grouped Query Attention (GQA)** to reduce memory overhead in self-attention computation.
   - Instead of computing a separate key-value pair for each query, queries are grouped, reducing redundant operations.
   - Benefits:
     - Improves efficiency, especially for long sequences.
     - Reduces the number of KV cache storage requirements during inference.
   - KV cache stores precomputed key-value pairs to speed up autoregressive generation, avoiding recomputation of previous tokens.


## Major Tensor Dimensions Summary
Below is a table outlining the dimensions of key tensors in the implementation:

| Tensor Name   | Shape Description |
|--------------|------------------|
| `x`         | (batch_size, seq_len, hidden_dim) |
| `freqs_cis` | (seq_len, head_dim // 2) |
| `tokens`    | (batch_size, seq_len) |
| `xq`        | (batch_size, seq_len, n_heads, head_dim) |
| `xk`        | (batch_size, seq_len, n_heads, head_dim) |
| `xv`        | (batch_size, seq_len, n_heads, head_dim) |
| `attn_scores` | (batch_size, n_heads, seq_len, seq_len) |
| `attn_probs`  | (batch_size, n_heads, seq_len, seq_len) |
| `attn_output` | (batch_size, seq_len, n_heads, head_dim) |
| `xq with rotary embedding`      | (batch_size, seq_len, hidden_dim) |
| `xk with rotary embedding`      | (batch_size, seq_len, hidden_dim) |
| `mlp_hidden` | (batch_size, seq_len, ffn_dim) |
| `mlp_output` | (batch_size, seq_len, hidden_dim) |
| `logits`    | (batch_size, seq_len, vocab_size) |
| `output`    | (batch_size, seq_len, vocab_size) |


## Acknowledgements

 - [Grouped Query Attention](https://klu.ai/glossary/grouped-query-attention)
 - [Grouped Query Attention (GQA) explained with code](https://medium.com/@maxshapp/grouped-query-attention-gqa-explained-with-code-e56ee2a1df5a)
 - [LLaMA: Concepts Explained](https://akgeni.medium.com/llama-concepts-explained-summary-a87f0bd61964)
 - [Activation function and GLU variants for Transformer models](https://medium.com/@tariqanwarph/activation-function-and-glu-variants-for-transformer-models-a4fcbe85323f)
 - [Transformers KV Caching Explained](https://medium.com/@joaolages/kv-caching-explained-276520203249)

