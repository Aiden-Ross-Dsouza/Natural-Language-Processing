
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
  
![Screenshot 2024-09-11 123919](https://github.com/user-attachments/assets/30feb953-39db-4a94-b28d-6c0dfe45fd4d)

- PCA plot of entire corpus
  
![Screenshot 2024-09-11 124015](https://github.com/user-attachments/assets/cb96c369-5c72-4ce0-b7ba-daee8848c15f)

- Analogical Reasoning: men - women ≈ phone - cell
  
![Screenshot 2024-09-11 124054](https://github.com/user-attachments/assets/d012985b-801b-491d-867e-03dd5515c580)

## Acknowledgements

 - [Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient Estimation of Word Representations in Vector Space.](https://arxiv.org/abs/1301.3781)
 - [Implementing word2vec in PyTorch (skip-gram model)](https://towardsdatascience.com/implementing-word2vec-in-pytorch-skip-gram-model-e6bae040d2fb)
 - [Word2Vec and Visualization with PCA](https://www.kaggle.com/code/chmasgun/word2vec-and-visualization-with-pca)

