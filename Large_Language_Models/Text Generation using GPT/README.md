
# Text Generation

This project implements a **Decoder-Only Transformer** to generate text in the style of William Shakespeare. The model generates text in an **autoregressive manner**, predicting one token at a time based on previously generated tokens. It uses a triangular mask in the attention mechanism to ensure tokens attend only to prior context, omitting the cross-attention block typically present in encoder-decoder models.  

The dataset, **Tiny Shakespeare**, contains a collection of Shakespeare’s works in plain text format. It is preprocessed to extract unique characters as the vocabulary and convert the text into numerical sequences, enabling the model to learn Shakespeare's linguistic patterns and stylistic nuances effectively.  

The model is built using PyTorch with custom multi-head self-attention and feedforward layers, forming the backbone of the Transformer architecture. Trained using the AdamW optimizer to minimize cross-entropy loss, the training process includes periodic evaluations on training and validation sets to track performance.

# Summary

Below is a table, summarising the number of parameters and the BLEU scores achieved by each architecture.

| Parameter | Value     |
| :-------- | :------- |
| Training Set	 | 1115394 characters |
| Testing Set	 | 111539 characters|
| Validation Set| 111539 characters|
| Loss Function	| Cross Entropy Loss |
| Optimizer| AdamW |

## References

 - [GPT by Andrej Karpathy](https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py)
 - [Deep Dive into AI: Building a Bigram Language Model and Practicing Patience! by Ada Choudhry](https://medium.com/@adachoudhry26/deep-dive-into-ai-building-a-bigram-language-model-and-practicing-patience-9341838063f7)
 - [Understanding How ChatGPT Uses the Decoder-Only Transformer Architecture by Younes Dahami](https://medium.com/@dahami/understanding-how-chatgpt-uses-the-decoder-only-transformer-architecture-c247c872754a)

