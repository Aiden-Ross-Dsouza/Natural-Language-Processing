# Vision Transformer (ViT)

## Overview

Vision Transformers (ViT) are deep learning models that leverage self-attention mechanisms, originally designed for NLP, to process image data. Unlike traditional Convolutional Neural Networks (CNNs), which rely on spatial hierarchies through convolutional layers, ViTs divide images into fixed-size patches and process them similarly to tokenized text in NLP.

This repository explains the inference and training of Vision Transformers (ViTs) for image classification tasks, providing insights into their advantages over traditional CNNs and comparing them with regular Transformer architectures.

### Architecture

The Vision Transformer follows a transformer-based structure for image classification:
1. **Patch Embedding:** The input image is divided into fixed-size patches, which are flattened and linearly embedded.
2. **Positional Encoding:** Added to retain spatial information within the sequence of patches.
3. **Transformer Encoder:** Consists of multiple layers of self-attention and feedforward neural networks.
4. **Classification Head:** The final encoded representation is processed through a classifier to output predictions.

ViTs have demonstrated competitive performance compared to CNNs, especially when trained on large-scale datasets. However, they require substantial data and computational resources to generalize effectively.

The dataset used for training is the [Digit Recognizer Dataset](https://www.kaggle.com/competitions/digit-recognizer/data).

## Comparison: ViT vs. CNN

| Feature                       | Vision Transformer (ViT)                           | Convolutional Neural Network (CNN)                  |
| ----------------------------- | -------------------------------------------------- | --------------------------------------------------- |
| Feature Extraction            | Uses self-attention to capture global dependencies | Uses convolutional layers to extract local features |
| Inductive Bias                | Minimal; relies on large-scale training data       | Stronger due to hierarchical feature extraction     |
| Computational Cost            | Higher due to attention mechanisms                 | Lower; optimized with convolutions                  |
| Performance on Small Datasets | Requires large-scale pretraining                   | Performs well even on small datasets                |

## Comparison: ViT vs. Regular Transformer

| Feature                       | Vision Transformer (ViT)                           | Regular Transformer                             |
| ----------------------------- | -------------------------------------------------- | ----------------------------------------------- |
| Input Type                    | Image patches                                      | Tokenized text                                  |
| Positional Encoding           | Learned or fixed positional embeddings             | Absolute positional encoding                    |
| Patch Processing              | Linearly embedded patches + CLS token              | Token embeddings + CLS token                    |
| Computational Complexity      | Scales quadratically with image resolution         | Scales quadratically with sequence length       |

## Hyperparameters

The model was trained with the following hyperparameters:

| Parameter        | Value         |
| ---------------- | ------------- |
| Learning Rate    | 1e-4          |
| Batch Size       | 512           |
| Epochs           | 40            |
| Optimizer        | Adam          |
| Loss Function    | Cross-Entropy |
| Img size         | 28            |
| Patch size       | 4             |
| Number of heads  | 8             |
|Number of encoders| 4             |

## Results

After training, the model achieved the following results:

| Metric              | Value  |
| ------------------- | ------ |
| Training Accuracy   | 84.85% |
| Validation Accuracy | 88.31% |
| Precision           | 88.18% |
| Recall              | 88.22% |
| F1 score            | 88.15% |

---

This repository serves as an implementation of Vision Transformers for image classification tasks, providing insights into their advantages over traditional CNNs.


## Acknowledgements

 - [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
 - [Vision Transformer: What It Is & How It Works - 2024 Guide](https://www.v7labs.com/blog/vision-transformer-guide)
 - [Transformers for Vision D2L](https://d2l.ai/chapter_attention-mechanisms-and-transformers/vision-transformer.html)
 - [ViT: Vision Transformer Medium Blog by Shivani Junawane](https://medium.com/machine-intelligence-and-deep-learning-lab/vit-vision-transformer-cc56c8071a20)
 - [vit-pytorch by lucidrains](https://github.com/lucidrains/vit-pytorch)
 - [pytorch-image-models by huggingface](https://github.com/huggingface/pytorch-image-models)
 - [Vision Transformer in PyTorch](https://www.youtube.com/watch?v=ovB0ddFtzzA&t=5s)'
 - [Implement and Train ViT From Scratch for Image Recognition - PyTorch](https://www.youtube.com/watch?v=Vonyoz6Yt9c)
