
# Skip-Gram Word2Vec Model

The Skip-Gram Word2Vec model is a popular technique for learning word embeddings from large text corpora. It aims to predict context words given a target (center) word, effectively capturing semantic relationships between words.

## Features

- Word Embeddings: Learns dense vector representations of words that effectively capture their meanings and relationships.
- Window Size: Context window size of 5 words to capture context around the target word.
 - Embedding Visualization: Tools for visualizing learned word embeddings in 2D and 3D space to analyze relationships and clustering using PCA.
- Analogical Reasoning: Evaluates the quality of embeddings using analogy tasks (e.g., "king" - "queen" ≈ "man" - "woman").

## Note

The quality of the embeddings depends on the size and quality of the training data. The corpus can be viewed [here](https://docs.google.com/document/d/19iBT9dReOPSoYh6juL6oaWelrlplaSaDkDqdMc8ZunA/edit?usp=sharing).
## Results

- Training Loss

- PCA plot of entire corpus

- Analogical Reasoning
## Acknowledgements

 - [Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient Estimation of Word Representations in Vector Space.](https://arxiv.org/abs/1301.3781)
 - [Implementing word2vec in PyTorch (skip-gram model)](https://towardsdatascience.com/implementing-word2vec-in-pytorch-skip-gram-model-e6bae040d2fb)
 - [Word2Vec and Visualization with PCA](https://www.kaggle.com/code/chmasgun/word2vec-and-visualization-with-pca)

