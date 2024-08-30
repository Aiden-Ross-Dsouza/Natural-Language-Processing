# Neural Machine Translation

The machine translation task was traditionally performed using statistical methods. However, like most statistical paradigms, this Statistical Machine Translation (SMT) had large computational and memory overheads.

With the popularization of neural networks and deep learning, heavy research into Neural Machine Translation (NMT) and its successful employment has largely replaced SMT. This section contains implementations of papers that introduced some of those ground-breaking architectures in NMT.

All the models were trained on the [Multi30k](https://arxiv.org/abs/1605.00459) dataset and the [English to Hindi Machine Translation (EHMT)](https://www.kaggle.com/datasets/parvmodi/english-to-hindi-machine-translation-dataset?select=train.hi) dataset. We have trained and evaluated our models for translation from German to English & Hindi to English.

Below is a table addressing some common data and optimization related parameters.

| Parameter      |       Multi30k        |          EHMT         |
| -------------- |:---------------------:|:---------------------:|
| Training Set   |      29000/31014      |       10000/18000     |
| Testing Set    |      1000/31014       |        3000/18000     |
| Validation Set |      1014/31014       |        5000/18000     |
| Loss Function  |   Cross Entropy Loss  |   Cross Entropy Loss  |
| Optimizer      |         Adam          |         Adam          |

## Architectures

### 1. [Neural Machine Translation by Jointly Learning to Align and Translate](https://github.com/Aiden-Ross-Dsouza/Natural-Language-Processing/blob/5e63cb4fc44a543579f6d89195576637bbf02555/Neural_Machine_Translation/notebooks/Seq2Seq_with_Attention.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hOd2JFafWgOvdbeXWoSm1gIiMEQ64KbM?usp=sharing)
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
* [PyTorch Transformer by Umar Jamil](https://github.com/hkproj/pytorch-transformer)
