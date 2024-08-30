# Neural Machine Translation

The machine translation task was traditionally performed using statistical methods. However, like most statistical paradigms, this Statistical Machine Translation (SMT) had large computational and memory overheads.

With the popularization of neural networks and deep learning, heavy research into Neural Machine Translation (NMT) and its successful employment has largely replaced SMT. This section contains implementations of papers that introduced some of those ground-breaking architectures in NMT.

All the models were trained on the [Multi30k](https://arxiv.org/abs/1605.00459) dataset which contains roughly 30 thousand English, German and French sentences, each sentence being 10-20 words long. We have trained and evaluated our models for translation from German to English.

Below is a table addressing some common data and optimization related parameters.

| Parameter      |       Value        |
| -------------- |:------------------:|
| Training Set   |    29000/31014     |
| Testing Set    |     1000/31014     |
| Validation Set |     1014/31014     |
| Loss Function  | Cross Entropy Loss |
| Optimizer      |       AdamW        |

## Architectures

### 1. [Neural Machine Translation by Jointly Learning to Align and Translate](https://github.com/IvLabs/Natural-Language-Processing/blob/master/neural_machine_translation/notebooks/Seq2Seq_with_Attention.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hOd2JFafWgOvdbeXWoSm1gIiMEQ64KbM?usp=sharing)
This paper presents a remarkable improvement in the Sequence to Sequence architecture by introducing a (soft)alignment metric called "attention". This metric induces a sense of similarity between tokens of the source and decoded sentences, which increases the BLEU score by almost 1.5 times and is much more robust with regards to the length of source and target sentences.\
**<ins>Note:</ins>** 
In this implementation, initializing the ```teacher_forcing_ratio``` at 0.6 and gradually decreasing it over iterations led to an improved BLEU score.

### 2. [Attention Is All You Need](https://github.com/IvLabs/Natural-Language-Processing/blob/master/neural_machine_translation/notebooks/Attention_Is_All_You_Need.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Aiden-Ross-Dsouza/Natural-Language-Processing/blob/main/Neural_Machine_Translation/notebooks/Seq2Seq_with_Attention.ipynb)
This paper can be said to have revolutionized almost all of Natural Language Processing, all owing to its tremendous simplicity and efficiency which makes it the backbone of almost all present State Of The Art architectures. The proposed *Transformer* architecture introduces a novel metric called *Self-Attention* which induces a sense of (soft)alignment amongst the source sentence tokens too. Additionally architecture uses simple feed-forward layers which makes it almost 4 times lighter than the Convolutional Sequence to Sequence architecture, while attaining the highest BLEU score amongst all the above mentioned architectures.

## Summary
Below is a table, summarising the number of parameters and the BLEU scores achieved by each architecture.

| Architecture                        | BLEU Score |
| ----------------------------------- |:----------:|
| Sequence to Sequence with Attention |   36    |
| Attention Is All You Need           |   37.50    |

<ins>**Note:**</ins>
1. The above BLEU scores may vary slightly upon training the models (even with fixed SEED).

## Plots
<p align="center">
  <img src = "https://github.com/IvLabs/Natural-Language-Processing/blob/master/neural_machine_translation/plots/Seq2Seq.jpeg?raw=true"/>
  <img src = "https://github.com/IvLabs/Natural-Language-Processing/blob/master/neural_machine_translation/plots/Seq2Seq_with_Attention.jpeg?raw=true"/> 
  <img src = "https://github.com/IvLabs/Natural-Language-Processing/blob/master/neural_machine_translation/plots/Conv_Seq2Seq.jpeg?raw=true"/>
  <img src = "https://github.com/IvLabs/Natural-Language-Processing/blob/master/neural_machine_translation/plots/Transformer.jpeg?raw=true"/>
</p>

### Reference(s):
* [PyTorch Seq2Seq by Aladdin Persson](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/558557c7989f0b10fee6e8d8f953d7269ae43d4f/ML/Pytorch/more_advanced/Seq2Seq_attention/seq2seq_attention.py)
