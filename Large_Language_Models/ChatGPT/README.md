
# GPT 2

GPT-2 is a large transformer⁠-based unsupervised language model, with 1.5 billion parameters, which generates coherent paragraphs of text. It performs rudimentary reading comprehension, machine translation, question answering, and summarization without task-specific training.


The staged release of GPT-2 by OpenAI in 2019 involved a cautious, phased approach to making the model publicly available due to concerns about potential misuse. The GPT-2 miniseries is given below:
| Model | Number of Parameters     | Number of Layers | Number of Heads | Embedding Size |
| :-------- | :------- | :------- | :------- | :------- |
| Small	 | 124 M | 12 | 12 | 768 |
| Medium	 | 335 M| 24 | 16 | 1024 |
| Large | 774 M | 36 | 20 | 1280 |
| Extra Large	| 1,558 M | 48 | 25 | 1600 |

Comparison with basic transformer architecture as given in the paper "Attention is All You Need":
Comparison Parameter | Attention is All You Need | GPT  |
|:------| :-------- | :------- |
| Training Objective | Minimizing cross-entropy loss for sequence-to-sequence tasks | Minimizing cross-entropy loss for autoregressive token prediction |
| Training Approach | Supervised | Unsupervised |
| Task | Both input-output tasks (e.g., translation) and language understanding | Predicting the next token in a sequence (unidirectional language modeling) |
|Encoder-Decoder | Includes both encoder and decoder components. | Uses only the decoder component. |
| Self-Attention Masking	| Unmasked attention in the encoder; causal masking in the decoder. | Uses causal (unidirectional) masking throughout. |
| Positional Embedding | Initialized and fixed to sinusoids of diff frequencies	 |  They are learnable parameters|
| Layer Norm	 | Present after Multi-Head Attention and Feed Forward Blocks | Present before Multi-Head Attention and Feed Forward Blocks. Additional layer norm present before the final classifier layer. |
| Residual Stream | Has layer norms inside it	| Clean residual stream. Gradients from the top flow straight to the inputs through the residual pathways unchanged |
| Number of Layers	 | 6 encoder and 6 decoder layers | Much deeper (e.g., 12, 24, or more layers depending on model size). |
| Number of Heads	| 8 attention heads | Larger number of heads (e.g., 12, 16, or more) |

# Model Summary

Some key hyper-parameters have been given below:
| Hyper-parameter | Value  |
| :-------- | :------- |
| Embedding Size/ Number of channels	 | 768|
| Max Sequence Length/ Block_size | 1,024 |
| Vocab Size	| 50,304 (Originaly 50,257) |
| Number of Layers	 | 12 |
| Number of Heads	 | 12 |
| Batch Size	 | 5,24,288 (2**19 ~0.5M) |
| Minor Batch Size | 4 |
| Loss Function | Cross Entropy Loss |
| Activation Function in MLP block | GeLU (tanh approximation) |
| Optimizer | AdamW (β1 = 0.9, β2 = 0.95, eps=1e-8, weight_decay=0.1) |
| Leanrning Rate Schedule | Warm up Phase followed by Cosine Decay (max_lr = 6e-4, min_lr = 6e-5) |

Some key dimensionalities have been mentioned below:
| Vector/ Matrix/ Layer | Dimension  |
| :-------- | :------- |
| Look up table for tokens (wte)	 | (vocab_size, n_embd) |
| Look up table for positions (wpe)	 | (seq_len, n_embd) |
| Position Embedding (pos_emb) | (1, seq_len, n_embd) |
| Token Embedding (tok_emb) | (minor_batch_size, seq_len, n_embd) |
| Input to transformer (x) | (minor_batch_size, seq_len, n_embd) |
| Causal Self Attention| q, k, v: (minor_batch_size, n_head, seq_len, head_size) [n_head X head_size = n_embd]|
|| Attention Scores matrix:  (minor_batch_size, n_head, seq_len, seq_len)|
|| c_proj linear layer: (n_embd, n_embd)|
|| output (y): (minor_batch_size, seq_len, n_embd)|
| MLP | c_fc linear layer: (n_embd, 4\*n_embd) |
|| c_proj linear layer: (4\*n_embd, n_embd)|


# Key Points

- nn.Embedding: The nn.Embedding module is a high-level wrapper around an array of numbers, effectively mapping each input index to a dense vector of fixed size. It allows for efficient lookups of token representations.
- Attention is a communication operation that allows all tokens (e.g., 1024 tokens) to interact. It's a mechanism for aggregating information — a weighted sum of inputs — where each token contributes according to its relevance. This is different from a Multi-Layer Perceptron (MLP), where information is processed individually for each token without communication between them.  In essence, attention aggregates and reduces information across tokens, while an MLP works independently on each token.
- The GELU (Gaussian Error Linear Unit) activation function is designed to provide smoother approximations compared to ReLU, especially for gradients near zero. The ReLU function can suffer from the dying ReLU problem, where neurons get stuck during training with zero gradients, preventing updates. GELU mitigates this issue with its smoother curve.
- The vocabulary size used is 50,304 tokens. When initializing the model, it's important to ensure that all tokens are equally likely during the initial phase, as this prevents the model from favoring any token too strongly and ensures the loss is relatively uniform across the vocabulary. This is achieved by setting each token's initial probability to be roughly 1/50,304, resulting in an initial loss of about 11.046. This setup ensures that the model begins training in a balanced state, avoiding confident mistakes.
- GPT models, including GPT-2, rely on tokenization for converting text into numeric representations. Tiktoken, the tokenizer used in GPT models, has a compression ratio of approximately 3:1, meaning that 1000 characters are roughly equivalent to 300 tokens.
- Top-K sampling is a technique used to generate diverse and coherent outputs. Unlike greedy decoding, which always picks the most probable token, Top-K sampling selects from the top K most likely tokens according to the model's predictions. This introduces randomness, which helps to generate more varied and creative text. By limiting the sampling pool to the top K tokens, the method strikes a balance between randomness and coherence, allowing for more natural language generation. In this implementation, I have sampled the top 50 tokens.
- The weight sharing scheme, as described in the paper "Using the Output Embedding to Improve Language Models," aims to align the matrices `lm_head.weight` and `transformer.wte.weight` so that tokens with similar meanings, such as an all-lowercase version and its uppercase counterpart or the same token in different languages, share similar embeddings and output probabilities. By ensuring that semantically similar tokens have close representations both at the input and output layers of the transformer, the model benefits from improved efficiency. This approach reduces the parameter count by approximately 40M (50,257 * 768), leading to a 30% reduction in the overall parameter size (30% of 124M = 40M), making the model more efficient without sacrificing performance.
- Weight initialization follows a normal distribution with a standard deviation of 0.02 for the weights, 0 for the biases, 0.01 for the position embeddings (`wpe`), and 0.02 for the token embeddings (`wte`). The 0.02 value aligns with Xavier initialization, where the standard deviation is proportional to the inverse square root of the number of incoming features (e.g., 1/√768 = 0.036, 1/√1600 = 0.02). For bias terms, PyTorch uses a uniform distribution by default, while the LayerNorm layers are initialized with a scale of 1 and an offset of 0, as per PyTorch's default initialization.
- The key computational bottleneck lies in the matrix multiplication of the `lm_head` final classifier layer, which dominates the processing time compared to other layers, including linear and activation layers. This operation is efficiently accelerated through the use of tensor cores, enhancing overall performance.
- The Flash Attention paper introduces a memory-efficient attention mechanism through kernel fusion, ensuring the attention matrix is never fully materialized using the Online SoftMax trick. This approach highlights the importance of memory hierarchy, emphasizing that optimizing memory access patterns is more critical than raw computational FLOPs. Additionally, it demonstrates optimizations that go beyond the capabilities of torch.compile, making it a powerful technique for efficient attention computation.
- In the GPT-2 paper, the learning rate schedule consists of two phases: a warm-up phase where the learning rate linearly increases from 0 to the desired value (1e-4) over a set number of steps, followed by a decay phase where the learning rate decreases according to a cosine decay schedule.
- After computing gradients with `loss.backward()`, gradient norm clipping is applied to prevent excessively large updates. This involves calculating the global norm of the gradients by squaring each parameter's gradient, summing them, and taking the square root. The global norm is then scaled such that its magnitude does not exceed a predefined threshold, ensuring that parameter updates remain controlled and preventing instability during training.
- A weight decay of 0.1 is applied to regularize the model, encouraging more balanced weight distributions. Weight decay is selectively applied to parameters involved in matrix multiplications, such as weights in linear layers and embeddings, to prevent any individual weight from becoming excessively large. However, biases and 1-D tensors, like those in layer normalization, are excluded from weight decay. This approach ensures that the model distributes the learning effectively across all channels, avoiding overfitting and promoting better generalization.
- Using fused Adam accelerates training on CUDA by eliminating the need for iterating through parameter tensors and updating them individually. Instead, all kernels are merged into a single operation, reducing overhead and improving efficiency. This kernel fusion streamlines the AdamW update process, leading to faster training times.
- To maintain a batch size of 0.5M tokens, which is crucial for tuning other hyperparameters, gradient accumulation is used to simulate this batch size without overloading the GPU. Instead of processing the entire batch in one go, multiple smaller batches are processed sequentially. Gradients are accumulated across these smaller batches during the forward and backward passes without updating the model weights until the desired batch size is reached. This approach allows for a larger effective batch size while managing memory limitations.
- Typically, for MSE loss, the formula is 1/N ∑(y - ŷ)², where N is the number of samples, normalizing the squared differences. However, if the MSE is computed for each sample individually, the loss for each becomes (y - ŷ)² without normalization. Similarly, when calculating the loss for a batch of size B and sequence length T, the loss is averaged over the entire batch (1/(B*T) ∑ (individual losses). In the case of gradient accumulation, the effective batch size increases, requiring an additional normalization by 1/grad_accum_steps to correctly scale the gradients.

# Optimizations

Optimizations made and improvement in training speed:

| Optimization | Description | Time/ Step (in ms) | % Improvement in Speed |
| :-------- | :------- | :------- | :------- |
| float32 |  | 1000 | 
| TF32 precision | TF32 accelerates computations by using a reduced precision (10-bit mantissa) compared to float32's 23-bit mantissa, which lowers memory usage | 333 | 66.7% |
| bfloat16 precision | bfloat16 (bF16) accelerates computation by using 16 bits instead of 32. While TF32 retains a 10-bit mantissa for better precision, bF16 uses an 8-bit mantissa, which speeds up operations | 300 | 9.9% |
| torch.compile | It speeds up PyTorch models by applying runtime optimizations like operator fusion and graph-based enhancements, improving performance without changing the model code. | 130 | 56.67% |
| Flash Attention | FlashAttention speeds up attention by optimizing memory usage and computation, reducing memory overhead by processing smaller data chunks, leveraging efficient matrix multiplication using specialized kernels, and computing attention in-place. | 96 | 26.15% |
| Nice/ Ugly Numbers	 | The vocabulary size increased from 50,257 to 50,304. Tokens that never appear in the dataset are assigned `-inf` biases, reducing their probability to near zero. The model treats the added dimensions similarly.| 93 | 3.125% |

# Acknowledgements

 - [GPT-2 Blog: Better language models and their implications](https://openai.com/index/better-language-models/)
 - [GPT-2 Paper: Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
 - [GPT-3 Paper: Language Models are Few-Shot Learners](https://bulldogjob.com/news/449-how-to-write-a-good-readme-for-your-github-project)
 - [GPT-2 Hugging Face](https://huggingface.co/docs/transformers/en/model_doc/gpt2)
 - [Let's reproduce GPT-2 (124M) by Andrej Karpathy](https://www.youtube.com/watch?v=l8pRSuU81PU&t=212s)
