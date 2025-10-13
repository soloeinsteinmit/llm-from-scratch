---
title: "Building LLMs From Scratch (Part 6): The Attention Mechanism"
description: "A deep dive into the attention mechanism, the core concept that revolutionized sequence modeling and made modern LLMs like GPT possible. We start with the theory and build a simplified self-attention mechanism from scratch."
tags: [LLM, AI, Machine Learning, Python, PyTorch, Deep Learning]
thumbnail: "images/L14_atten.png"
---

# Building LLMs From Scratch (Part 6): The Attention Mechanism

Welcome to Part 6 of our "LLMs From Scratch" series! In our last chapter, we successfully built the entire data preprocessing pipeline, transforming raw text into model-ready tensors packed with token and positional information. But how does a model actually _use_ these embeddings to understand language?

The answer lies in the **attention mechanism**, arguably the most important concept behind the transformer architecture and modern LLMs. It's the "secret sauce" that allows models to understand context, handle long-range dependencies, and generate coherent text.

Today, we'll unravel the magic of attention. We'll explore the different types of attention, understand the problem it was designed to solve, and then build our very first, simplified self-attention mechanism from scratch.

### 🔗 Quick Links

- **GitHub Repository**: [llm-from-scratch](https://github.com/soloeinsteinmit/llm-from-scratch)
- **Previous Part**: [Part 5: The Complete Data Preprocessing Pipeline](https://soloshun.medium.com/building-llms-from-scratch-part-5-the-complete-data-preprocessing-pipeline-5247a8ee232a)

### 📋 What We'll Cover

- **Chapter 1: The Four Flavors of Attention** - A roadmap of the different attention mechanisms.
- **Chapter 2: The Problem with Long Sequences** - A deep dive into why RNNs struggle.
- **Chapter 3: The Breakthrough Idea of Attention** - How "dynamic focusing" solved the problem.
- **Chapter 4: From Attention to Self-Attention** - The core idea of the "Attention Is All You Need" paper.
- **Chapter 5: A Simplified Self-Attention Mechanism** - Building the logic from the ground up.
- **Chapter 6: Implementing Simplified Self-Attention in Code** - A clear PyTorch implementation.

---

## Chapter 1: The Four Flavors of Attention

Before we dive in, it's helpful to know that "attention" isn't a single thing but a family of mechanisms. Throughout this series, we will encounter four main types:

![](../../images/L13_types_att.png)

1.  **Simplified Self-Attention:** A clean, minimal version without trainable weights. It's perfect for building our initial intuition. _(This is our focus today!)_
2.  **Self-Attention with Trainable Weights:** The true foundation of LLMs, where the model _learns_ how to focus its attention using trainable Query, Key, and Value matrices.
3.  **Causal Attention:** A special type of self-attention used in decoder-style models like GPT. It ensures that when generating a new word, the model can only look at previous words and cannot see into the future.
4.  **Multi-Head Attention:** An extension that allows the model to perform self-attention multiple times in parallel, with each "head" focusing on different types of relationships in the text.

Today, we start with the simplest form to build a solid foundation.

---

## Chapter 2: The Problem with Long Sequences (A Recap)

Before we can appreciate attention, we need to understand the problem it solved. Let's start with a simple example: language translation. A naive approach would be to translate a sentence word-by-word. But as the image below shows, this often fails dramatically because languages have different grammatical structures.

![](../../images/L13_s1.png)

The German sentence "Kannst du mir helfen diesen Satz zu uebersetzen" translates word-for-word into the grammatically incorrect English "Can you me help this sentence to translate." The correct translation, "Can you help me translate this sentence," requires reordering the words, demonstrating the need for **contextual understanding** and **grammar alignment**.

![](../../images/L13_s3.png)

To solve this, before transformers, models relied on Recurrent Neural Networks (RNNs) in an **encoder-decoder** setup.

![](../../images/L13_s5.png)

The process was:

1.  The **encoder** RNN reads the input sentence one word at a time, updating its internal "hidden state."
2.  It compresses the _entire meaning_ of the input sentence into a single final hidden state vector (often called the "context vector").
3.  This single vector is then passed to the **decoder** RNN, which tries to generate the translated sentence word by word.

The issue? This single context vector creates an **information bottleneck**.

As one of my lecture notes wisely put it:

> “The decoder loses full context—it only sees a compressed summary from the encoder. Imagine writing an exam from just a 1-pager summary of a 6-month course... Too much gets lost!”

For a long, complex sentence like, "The cat that was sitting on the mat, which was next to the dog, **jumped**," the RNN might forget that "cat" was the subject by the time it needs to translate the verb "jumped." This struggle with **long-range dependencies** was a major limitation.

---

## Chapter 3: The Breakthrough Idea of Attention

In 2014, a paper titled "[Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)" by Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio introduced a revolutionary idea to solve this very problem: the **attention mechanism**.

**The core idea:** Instead of forcing the encoder to cram everything into one vector, what if we allow the decoder to "look back" at the _entire_ input sequence at every step of the generation process?

![](../../images/L13_s6.png)

With attention, the decoder can **dynamically focus** on the most relevant parts of the input for the specific word it's currently trying to generate. It does this by calculating **attention weights**, which are scores that determine how much importance to place on each input word.

When translating "jumped" to French ("sauté"), the attention mechanism would allow the model to pay high attention to the words "cat" and "jumped" in the input, no matter how far apart they are.

![](../../images/L13_s7.png)

This was a massive breakthrough. It allowed models to handle long-range dependencies effectively and dramatically improved performance on tasks like machine translation. An important insight here is that this "looking back" happens in the vector space of embeddings. The model isn't just looking at words; it's comparing their embedding vectors to see which ones are most semantically similar or relevant, regardless of their position in the sentence.

---

## Chapter 4: From Attention to Self-Attention

For a few years, attention was used to enhance RNN-based models. But in 2017, the seminal paper **"Attention Is All You Need"** proposed an even more radical idea: what if we get rid of the RNNs entirely and build a model using _only_ the attention mechanism?

This gave birth to the **Transformer architecture** and a new concept: **self-attention**.

- **Traditional Attention** connects two different sequences (an encoder sequence and a decoder sequence).
- **Self-Attention** operates on a _single sequence_. It allows each word in the sequence to look at all the other words in the _same sequence_ to compute a new representation of itself.

In essence, self-attention helps the model understand the context of a word by learning the relationships and dependencies between all the words within the same sentence.

![](../../images/L13_s8.png)

This is the fundamental mechanism that allows models like GPT to understand grammar, syntax, and complex relationships in text.

---

## Chapter 5: A Simplified Self-Attention Mechanism

Now, let's build our first attention mechanism. It's crucial to start with the main goal:

> **The primary goal of any attention mechanism is to convert input vectors into context vectors.**

A **context vector** is an enriched version of the original input vector. It contains not just the meaning of the word itself, but also contextual information learned from its relationship with all other words in the sequence.

![](../../images/L14_atten.png)

Let's build a simplified self-attention mechanism, without any trainable weights, to understand the core logic. Our goal is to calculate a context vector `z` for each input vector `x`.

We'll use the input sentence: `"your journey starts with one step"`

Let's say we want to compute the context vector `z₂` for the word "journey" (`x₂`).

### Step 1: Compute Attention Scores

First, we need to score how relevant every other word in the sequence is to our target word, "journey." A simple and effective way to do this is with the **dot product**. The dot product of two vectors measures their similarity or alignment. A higher dot product means the vectors are more aligned.

We compute the dot product of our query vector (`x₂`) with every other vector in the sequence (`x₁`, `x₂`, `x₃`, ..., `x₆`). These are our **attention scores**.

![](../../images/L14_s1.png)

### Step 2: Normalize with Softmax

These raw scores can be on any scale. We need to convert them into a probability distribution where all the scores are between 0 and 1 and sum up to 1. While a simple normalization (dividing each score by the sum of all scores) might seem intuitive, it doesn't handle extreme values well. For instance, with scores like `[1, 3, 4, 400]`, the largest value would dominate, but smaller values would still retain some weight. This can be problematic during training because even small, near-zero weights can receive undue attention during backpropagation.

A better approach is the **softmax function**. Softmax amplifies the highest scores and diminishes the smaller ones, making the highest score's weight approach 1 while the others become negligible. This helps the model focus on the most relevant tokens.

![](../../images/L14_s2_5.png)

However, a naive implementation of the softmax function can be numerically unstable. When dealing with very large numbers, the `exp(x)` term can lead to overflow errors. To address this, a common technique is to subtract the maximum value from each score before applying the exponentiation. This "stable softmax" is a more robust implementation, as used in libraries like PyTorch.

![](../../images/L14_s3.png)

After applying softmax, we get our final **attention weights**. These weights tell us exactly how much attention the word "journey" should pay to every other word in the sentence (including itself).

### Step 3: Compute the Context Vector

The final step is to create the context vector `z₂`. We do this by taking a **weighted sum** of all the input vectors `xᵢ`. The weights we use are the normalized attention weights we just calculated.

We multiply each input vector `xᵢ` by its corresponding attention weight and then sum up the results.

![](../../images/L14_s4_5.png)

The resulting vector `z₂` is the new, context-aware representation for "journey." It has absorbed information from the entire sequence, with the most relevant words (those with higher attention weights) contributing more to its final value.

We repeat this exact process for every single word in the input sequence to get a full set of context vectors.

---

## Chapter 6: Implementing Simplified Self-Attention in Code

Now, let's translate this logic into PyTorch code. To make it concrete, we'll use the exact values from the hands-on notebook that accompanies this series.

First, let's define our input embeddings for the sentence `"Your journey starts with one step"`.

```python
import torch

# Input sentence: "Your journey starts with one step"
# We'll use the specific embeddings from our notebook for reproducibility.
inputs = torch.tensor(
  [
   [0.43, 0.15, 0.89], # Your     (x_1)
   [0.55, 0.87, 0.66], # journey  (x_2)
   [0.57, 0.85, 0.64], # starts   (x_3)
   [0.22, 0.58, 0.33], # with     (x_4)
   [0.77, 0.25, 0.10], # one      (x_5)
   [0.05, 0.80, 0.55], # step     (x_6)
  ]
)
```

Now, let's walk through the three steps to calculate the context vector for our second word, "journey".

```python
# Our query is the embedding for "journey"
query = inputs[1] # x_2

# 1. Compute attention scores
# We compute the dot product of the query with all other input vectors.
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)

# These scores represent the similarity of each word to "journey".
# Notice the highest scores are for "journey" and "starts".
print("Attention scores:", attn_scores_2)
# Expected output: tensor([0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865])

# 2. Normalize with softmax
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)

# The weights now sum to 1, representing a probability distribution.
# "journey" (0.2379) and "starts" (0.2333) have the highest attention weights.
print("Attention weights:", attn_weights_2)
# Expected output: tensor([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])

# 3. Compute the context vector
# We compute the weighted sum of all input vectors using the attention weights.
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i

print("Context vector for 'journey':", context_vec_2)
# Expected output: tensor([0.4419, 0.6515, 0.5683])
```

The new `context_vec_2` is the enriched representation for "journey." It now contains contextual clues from "starts," "step," and the other words, proportional to their attention weights.

### Doing it for all inputs at once

Looping like this is great for understanding, but in practice, we use matrix multiplications for efficiency. We can compute the context vectors for all input words simultaneously.

```python
# 1. Compute all attention scores at once
# The @ operator in PyTorch performs matrix multiplication.
# (6, 3) @ (3, 6) -> (6, 6)
attn_scores = inputs @ inputs.T

# 2. Normalize all scores
# We apply softmax to each row of the attention score matrix.
attn_weights = torch.softmax(attn_scores, dim=-1)

# 3. Compute all context vectors
# (6, 6) @ (6, 3) -> (6, 3)
all_context_vectors = attn_weights @ inputs

print("Original input embeddings:\n", inputs)
print("\nFinal context vectors:\n", all_context_vectors)
```

This code precisely follows the three steps we outlined, but in a much more efficient, vectorized way. It calculates a new, context-rich vector for _every_ word in our sequence based on its relationship with all other words.

---

## Conclusion & What's Next

Congratulations! You now understand the fundamental mechanics of attention.

### Key Takeaways:

- Attention solves the **information bottleneck** of older RNN models.
- It allows a model to **dynamically focus** on relevant parts of the input.
- **Self-attention** enables a model to learn relationships between words in a single sequence.
- The process involves three steps: **scoring (dot product)**, **normalizing (softmax)**, and creating a **weighted sum**.

This simplified version is great for understanding the concept, but it's missing a crucial ingredient: **trainable weights**. Our current model can't _learn_ which words are important; it can only calculate similarity based on the initial embeddings.

In **Part 7**, we will introduce trainable weight matrices (the famous Query, Key, and Value) to create a full, trainable self-attention mechanism—the true powerhouse behind LLMs. Stay tuned!
