---
title: "Building LLMs From Scratch (Part 4): The Embedding Layer"
date: "2024-09-22"
tags:
  [
    "AI",
    "Machine Learning",
    "LLM",
    "PyTorch",
    "Deep Learning",
    "NLP",
    "Embeddings",
  ]
series: "Building LLMs from Scratch"
---

# Building LLMs From Scratch (Part 4): The Embedding Layer

In the last three parts, we established our theoretical foundation, built a tokenizer, and crafted a data pipeline. Our `DataLoader` now serves up batches of token IDs, ready for training.

But there's a problem: token IDs like `40` and `367` are just arbitrary labels. They have no inherent meaning. The model can't know that the token for "cat" is semantically closer to "kitten" than it is to "car."

To solve this, we need to convert these IDs into a rich, numerical representation that captures meaning. This is the job of the **Embedding Layer**, the very first layer of our neural network.

![](../../images/token_embed_dia.png)

In this part, we will build the two critical components of this layer:

1.  **Token Embeddings**: To capture the _semantic meaning_ of each token.
2.  **Positional Embeddings**: To capture the _order_ of the tokens in the sequence.

---

## Chapter 1: The Quest for Meaning - Why Token IDs Aren't Enough

Let's start by understanding the problem more deeply. We have token IDs, which are unique integers. Why can't we just feed them into our model?

### Approach 1: Use Token IDs Directly (And Why It Fails)

Token IDs are categorical, not continuous. The numerical distance between them is meaningless. For example:

- "cat" -> ID `4512`
- "dog" -> ID `9821`

The model might incorrectly assume that "dog" is "more than" "cat," or that their relationship can be measured by the difference `9821 - 4512`. This introduces random, unhelpful biases.

### Approach 2: One-Hot Encoding (A Better, But Flawed, Idea)

A common technique in machine learning is to use **one-hot vectors**. For a vocabulary of size 50,000, we could represent a word as a vector of length 50,000 that is all zeros except for a single `1` at the index corresponding to that word's token ID.

- "cat" (ID 4512) -> `[0, 0, ..., 1, ..., 0, 0]` (a `1` at the 4512th position)
- "dog" (ID 9821) -> `[0, 0, ..., 1, ..., 0, 0]` (a `1` at the 9821st position)

This solves the arbitrary bias problem, but introduces two new, massive ones:

1.  **Extreme Sparsity & Dimensionality**: These vectors are enormous and mostly empty, making computation and storage highly inefficient.
2.  **No Notion of Similarity**: In this vector space, every word is equidistant from every other word. The model has no way to know that "cat" is more similar to "kitten" than it is to "car."

### The Solution: Vector Embeddings (Dense, Learned Representations)

Instead of a sparse, meaningless vector, we map each token ID to a **dense, lower-dimensional vector** filled with real numbers. This is called an **embedding**.

![](../../images/L10_vec_s5.png)

These vectors are not fixed; they are **learnable parameters**. During training, the model adjusts these vectors. The goal is to arrange them in a high-dimensional "semantic space" where:

- Words with similar meanings (e.g., "cat" and "kitten") have vectors that are numerically close to each other.
- Analogies and relationships can be captured with vector arithmetic (e.g., `vector("King") - vector("Man") + vector("Woman")` is famously close to `vector("Queen")`).

> In short: **Token IDs are just labels. Embeddings are meaning.**

For a deeper, visual intuition on how vectors can capture meaning, the video on Word Embeddings by 3Blue1Brown is an excellent resource.
**[Watch 3Blue1Brown on Word Embeddings](https://www.youtube.com/watch?v=gG-w_n9-1-A)**

---

## Chapter 2: The Embedding Matrix - A Learnable Lookup Table

In PyTorch, we implement this with `torch.nn.Embedding`. You can think of this layer as a giant **lookup table** or a matrix that the model learns over time.

- **Rows**: The number of rows equals the `vocab_size` (e.g., 50,257 for GPT-2).
- **Columns**: The number of columns is the `embedding_dim` (e.g., 768 for GPT-2). This is the size of the vector for each token.

![](../../images/L10_vec_s7.png)

When we pass a batch of token IDs to this layer, it simply performs a lookup, fetching the corresponding vector for each ID.

At the start of training, this giant matrix is filled with small, random numbers. As the model trains on predicting the next word, it uses backpropagation to constantly update these embedding vectors. If the model learns that "cat" and "kitten" often appear in similar contexts, it will gradually nudge their vectors closer together.

---

## Chapter 3: The Problem of Order

We've solved the meaning problem, but there's another one. A simple embedding layer is **order-agnostic**. The self-attention mechanism in a Transformer processes all tokens simultaneously, so without some positional information, it's like seeing a "bag of words."

Consider these two sentences:

- `The cat sat on the mat`
- `The mat sat on the cat`

Both sentences contain the exact same tokens, so their token embeddings would be identical. The model would have no way of knowing their fundamental meaning is different.

![](../../images/L11_posi_enc.png)

We need a way to inject the **order** of the tokens into our model.

---

## Chapter 4: Injecting Order with Positional Embeddings

The solution is to create a second embedding layer—a **positional embedding layer**. This layer doesn't care about the token's _meaning_, only its _position_ in the sequence (0, 1, 2, ...).

For our GPT-style model, we use **Absolute Positional Embeddings**. This is another learnable lookup table:

- **Rows**: The number of rows equals the `context_size` (the maximum sequence length our model can handle).
- **Columns**: The number of columns must match the `embedding_dim`, so we can add it to the token embedding.

The shape of this matrix is `[context_size, embedding_dim]`. The vector at row `0` represents the "first word" embedding, the vector at row `1` represents the "second word" embedding, and so on. These are also learned during training.

![](../../images/L11_abs_enc.png)

---

## Chapter 5: Putting It All Together - Code and Concepts

The final input to our transformer model is the **sum** of the token embedding and the positional embedding. This elegantly combines semantic meaning with sequential context in a single tensor.

`Input Embedding = Token Embedding + Positional Embedding`

Let's walk through the entire process with a concrete example, visualizing the tensor shapes.

![](../../images/L11_img1.png)

1.  Our `DataLoader` gives us a batch of token IDs of shape **`[8, 4]`** (`batch_size`, `context_size`).
    ![](../../images/L11_img2.png)

2.  The **token embedding layer** (`[vocab_size, 256]`) looks up these IDs, producing a tensor of shape **`[8, 4, 256]`**.
    ![](../../images/L11_img3.png)

3.  We create a sequence of position IDs: `[0, 1, 2, 3]`. The **positional embedding layer** (`[4, 256]`) looks these up, producing a tensor of shape **`[4, 256]`**.
    ![](../../images/L11_img4.png)

4.  We add the two. PyTorch uses **broadcasting** to automatically "stretch" or "repeat" the `[4, 256]` positional tensor across the batch dimension, effectively making it `[8, 4, 256]` so it can be added element-wise.

Here is the full implementation:

```python
import torch

# Hyperparameters
vocab_size = 50257
embedding_dim = 256
context_size = 4

# Create the layers
token_embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim)
pos_embedding_layer = torch.nn.Embedding(context_size, embedding_dim)

# --- Assume `inputs` is a batch of token IDs from our dataloader ---
# inputs shape: [8, 4]
# inputs = ...

# 1. Get token embeddings
# token_embeddings shape: [8, 4, 256]
# token_embeddings = token_embedding_layer(inputs)

# 2. Get positional embeddings
# First, create position IDs: tensor([0, 1, 2, 3])
pos_ids = torch.arange(context_size)
# pos_embeddings shape: [4, 256]
# pos_embeddings = pos_embedding_layer(pos_ids)

# 3. Add them together (broadcasting handles the batch dimension)
# input_embeddings shape: [8, 4, 256]
# input_embeddings = token_embeddings + pos_embeddings
```

This final `input_embeddings` tensor now contains vectors that are rich with both **semantic meaning** and **positional context**. This is the fully prepared input that will be fed into the first Transformer block.

---

## Conclusion & What's Next

We have now officially built the first neural network layer of our LLM!

We've learned not just _how_ to implement embeddings, but _why_ they are superior to simpler methods. We can now represent the meaning of tokens using a **token embedding** matrix and encode word order using a **positional embedding** matrix. By combining them, we create a powerful input representation for our model.

With our input data fully prepared, we are ready to tackle the core mechanism of the Transformer architecture.

**In Part 5**, we will implement the revolutionary **self-attention mechanism** from scratch. This is the engine that allows the model to understand context by weighing the importance of different words in a sequence, and it's where the magic truly begins.

---

## 🧑‍💻 Full Code

You can find the complete, interactive code for this part in our GitHub repository:

- **Jupyter Notebook**: [llm-from-scratch/notebooks/part04_embeddings.ipynb](https://github.com/soloeinsteinmit/llm-from-scratch/blob/main/notebooks/part04_embeddings.ipynb)
- **Python Script**: [llm-from-scratch/src/part04_embeddings.py](https://github.com/soloeinsteinmit/llm-from-scratch/blob/main/src/part04_embeddings.py)
