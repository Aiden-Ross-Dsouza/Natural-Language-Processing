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
**<ins>Note:</ins>** In this implementation, setting the ```teacher_forcing_ratio``` to 1 resulted in a better BLEU score (even with higher validation and test perplexities) than setting it to 0.5.

### 2. [Attention Is All You Need](https://github.com/IvLabs/Natural-Language-Processing/blob/master/neural_machine_translation/notebooks/Attention_Is_All_You_Need.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RlDhclIlJWzcFC0iPqwbiFEuiCbYIJBG?usp=sharing)
This paper can be said to have revolutionized almost all of Natural Language Processing, all owing to its tremendous simplicity and efficiency which makes it the backbone of almost all present State Of The Art architectures. The proposed *Transformer* architecture introduces a novel metric called *Self-Attention* which induces a sense of (soft)alignment amongst the source sentence tokens too. Additionally architecture uses simple feed-forward layers which makes it almost 4 times lighter than the Convolutional Sequence to Sequence architecture, while attaining the highest BLEU score amongst all the above mentioned architectures.

## Summary
Below is a table, summarising the number of parameters and the BLEU scores achieved by each architecture.

| Architecture                        | No. of Trainable Parameters | BLEU Score |
| ----------------------------------- |:---------------------------:|:----------:|
| Sequence to Sequence with Attention |         20,518,917          |   31.24    |
| Attention Is All You Need           |          9,038,853          |   37.50    |

<ins>**Note:**</ins>
1. The above BLEU scores may vary slightly upon training the models (even with fixed SEED).
2. The research paper notes for the above mentioned papers can be found [here](https://github.com/IvLabs/ResearchPaperNotes/tree/master/natural_language_processing).

## Plots
<p align="center">
  <img src = "https://github.com/IvLabs/Natural-Language-Processing/blob/master/neural_machine_translation/plots/Seq2Seq.jpeg?raw=true"/>
  <img src = "https://github.com/IvLabs/Natural-Language-Processing/blob/master/neural_machine_translation/plots/Seq2Seq_with_Attention.jpeg?raw=true"/> 
  <img src = "https://github.com/IvLabs/Natural-Language-Processing/blob/master/neural_machine_translation/plots/Conv_Seq2Seq.jpeg?raw=true"/>
  <img src = "https://github.com/IvLabs/Natural-Language-Processing/blob/master/neural_machine_translation/plots/Transformer.jpeg?raw=true"/>
</p>


### Reference(s):
* [PyTorch Seq2Seq by Ben Trevett](https://github.com/bentrevett/pytorch-seq2seq)
