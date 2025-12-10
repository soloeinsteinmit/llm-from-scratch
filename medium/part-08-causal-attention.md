---
title: "Building LLMs From Scratch (Part 8): Causal Attention (Masked Self-Attention)"
description: "We solve the 'future leakage' problem by implementing Causal Attention. Learn how diagonal masking forces the model to respect the arrow of time, a critical requirement for autoregressive text generation."
tags: [LLM, AI, Machine Learning, Python, PyTorch, Deep Learning]
thumbnail: "images/L16_s1.png"
---

# Building LLMs From Scratch (Part 8): Causal Attention

Welcome back to the "LLMs From Scratch" series! In [Part 7](https://medium.com/@soloshun/building-llms-from-scratch-part-7-self-attention-with-trainable-weights-641035115cc6), we built a **trainable self-attention mechanism**. We gave our model the ability to learn relationships between words using Query, Key, and Value matrices.

However, our current implementation has a fatal flaw for text generation: **it cheats.**

When our model is trying to predict the word "starts" in the sentence "Your journey starts with one step", the self-attention mechanism allows it to "peek" at the words "with", "one", and "step". In a real-world generation scenario, those future words _wouldn't exist yet_.

Today, we are going to fix this by implementing **Causal Attention**, also known as **Masked Self-Attention**. This is the mechanism that forces the model to respect the arrow of time.

### 🔗 Quick Links

- **GitHub Repository**: [llm-from-scratch](https://github.com/soloeinsteinmit/llm-from-scratch)
- **Previous Part**: [Part 7: Self-Attention with Trainable Weights](https://medium.com/@soloshun/building-llms-from-scratch-part-7-self-attention-with-trainable-weights-641035115cc6)

### 📋 What We'll Cover

- **The Cheating Problem**: Why standard self-attention fails for generation.
- **The Autoregressive Property**: The mathematical rule we must follow.
- **Diagonal Masking**: How to blind the model to the future.
- **Implementation**: Why we mask _before_ softmax (crucial!).
- **The Code**: Building a production-ready `CausalAttention` class.

---

## The Problem: Future Leakage

Let's revisit our goal: we want to build an **autoregressive** model. This means the model generates text one token at a time, using only the tokens generated so far to predict the next one.

Mathematically, the probability of the next token $x_t$ depends only on the previous tokens $x_{<t}$:

$$P(x) = \prod_{t=1}^T P(x_t | x_{<t})$$

In standard self-attention, every token attends to _every other token_.

![](../../images/L16_s1.png)

If we are training the model on the sentence "Your journey starts...", and we are at the position for "journey", standard attention allows the model to see "starts". The model might learn to just copy the next word instead of understanding the language structure. This is called **data leakage**.

During inference (when we generate new text), we don't have the future tokens. So a model trained this way will fail miserably because it learned to rely on information that isn't there.

---

## The Solution: Causal Masking

To solve this, we need to modify our attention mechanism so that a token at position $t$ can only attend to positions $0$ through $t$. It effectively "masks out" all future positions ($t+1, t+2, ...$).

We achieve this by applying a **mask** to our attention scores matrix.

Recall that our attention scores matrix has the shape `(seq_len, seq_len)`.

- Rows represent **Queries** (current positions).
- Columns represent **Keys** (positions being attended to).

We want to keep the lower triangular part (past and present) and zero out the upper triangular part (future).

![](../../images/L16_s2.png)

### The Mechanics of Masking

We don't literally multiply by zero. Instead, we want the attention weights for future tokens to be **zero after the softmax**.

Recall the softmax function: $e^x / \sum e^{x_i}$.
To get a result of $0$ from softmax, the input must be $-\infty$ (negative infinity).

So, our strategy is:

1.  Compute **Attention Scores** ($QK^T / \sqrt{d_k}$).
2.  Create a mask where future positions are `1` (or `True`).
3.  Use `masked_fill` to replace scores at those positions with `-inf`.
4.  Apply **Softmax**. The `-inf` values become $0$.

---

## Implementation Details

Let's implement this in PyTorch. We'll start with the mask itself.

### Step 1: Creating the Mask

We can use `torch.triu` (triangular upper) to create a matrix with 1s above the diagonal and 0s below.

```python
context_length = 6
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
print(mask)
# tensor([[0., 1., 1., 1., 1., 1.],
#         [0., 0., 1., 1., 1., 1.],
#         [0., 0., 0., 1., 1., 1.],
#         [0., 0., 0., 0., 1., 1.],
#         [0., 0., 0., 0., 0., 1.],
#         [0., 0., 0., 0., 0., 0.]])
```

Here, `1` represents "future" (to be masked), and `0` represents "past/present" (to be kept).

### Step 2: Applying the Mask

We apply this mask _before_ the softmax.

```python
# Assume attn_scores is our (6, 6) matrix of dot products
masked_scores = attn_scores.masked_fill(mask.bool(), -torch.inf)
```

Now, if we print `masked_scores`, the upper triangle will be `-inf`. When we run `torch.softmax(masked_scores, dim=-1)`, those positions turn into clean zeros.

### Why Mask BEFORE Softmax?

You might wonder: "Why not compute softmax first and then zero out the future?"

If we apply softmax _first_, the probabilities will sum to 1 across _all_ tokens (including future ones). If we then zero out the future tokens, the remaining probabilities (past tokens) will sum to _less than 1_. We would have to re-normalize them.

More importantly, the **values** of the past tokens would have been influenced by the presence of future tokens in the softmax denominator. This is a subtle form of **leakage**. By masking with `-inf` _before_ softmax, we ensure future tokens effectively "don't exist" in the probability calculation.

---

## The Full `CausalAttention` Class

Let's put it all together into a robust PyTorch module. We'll also add **Dropout**, a regularization technique used in GPT models to prevent overfitting.

### Important: `register_buffer`

We use `self.register_buffer` for the mask. In PyTorch, a buffer is a tensor that is part of the model's state (it gets saved/loaded) but is **not a trainable parameter** (no gradients). This is perfect for our fixed mask.

```python
import torch
import torch.nn as nn

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)

        # Create the causal mask (upper triangular with 1s)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        # b: batch size, num_tokens: sequence length, d_in: embedding dim
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 1. Compute Attention Scores
        # Transpose keys for matrix multiplication: (b, num_tokens, d_out) @ (b, d_out, num_tokens)
        attn_scores = queries @ keys.transpose(1, 2)

        # 2. Apply Causal Mask
        # We slice the mask to match the current sequence length (num_tokens)
        # This handles cases where input sequence is shorter than max context_length
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens],
            -torch.inf
        )

        # 3. Scale and Softmax
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )

        # 4. Apply Dropout
        attn_weights = self.dropout(attn_weights)

        # 5. Compute Context Vectors
        context_vec = attn_weights @ values
        return context_vec
```

### Visualizing the Result

With this mechanism, if we look at the attention weights for the word "journey" (position 1), it will have non-zero weights for "Your" (position 0) and "journey" (position 1), but **zero** weights for "starts", "with", "one", "step".

The model is now forced to predict the next word based _only_ on what it has seen so far.

---

## Conclusion & What's Next

We have successfully implemented **Causal Attention**. By applying a diagonal mask, we've ensured our model adheres to the autoregressive property, making it suitable for text generation.

Our implementation now includes:

- ✅ Trainable Weights (Q, K, V)
- ✅ Scaled Dot-Product
- ✅ Causal Masking
- ✅ Dropout

However, modern LLMs like GPT-4 don't just use a single attention mechanism. They use **Multi-Head Attention**—running multiple causal attention mechanisms in parallel to capture different types of relationships (e.g., one head focuses on grammar, another on semantic meaning).

In **Part 9**, we will upgrade our class to implement **Multi-Head Attention**, the final piece of the attention puzzle!

See you there!
