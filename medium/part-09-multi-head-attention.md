---
title: "Building LLMs From Scratch (Part 9): Multi-Head Attention"
description: "We upgrade our attention mechanism to 'Multi-Head Attention', allowing the model to focus on multiple aspects of the input simultaneously. We'll implement the efficient 'Weight Split' method used in GPT."
tags: [LLM, AI, Machine Learning, Python, PyTorch, Deep Learning]
thumbnail: "images/L17_s1.png"
---

# Building LLMs From Scratch (Part 9): Multi-Head Attention

Welcome back to the "LLMs From Scratch" series! In [Part 8](https://medium.com/@soloshun/building-llms-from-scratch-part-8-causal-attention-6e4a0578c88c), we built **Causal Attention**, ensuring our model respects the flow of time by masking future tokens.

However, our current implementation has a limitation: it's a **single-head** attention mechanism. It can only focus on one set of relationships at a time.

Imagine reading a complex novel. You might need to track:

1.  **Grammar**: Which noun corresponds to this verb?
2.  **Sentiment**: Is this sentence angry or happy?
3.  **Facts**: What specific names and dates were mentioned?

Doing all this with a single "attention focus" is hard. You'd likely miss nuances. **Multi-Head Attention** solves this by giving the model multiple parallel "heads," each capable of learning different types of relationships independently.

Today, we will build the final, production-ready attention module used in modern Transformers (like GPT-4 and Llama).

### 🔗 Quick Links

- **GitHub Repository**: [llm-from-scratch](https://github.com/soloeinsteinmit/llm-from-scratch)
- **Previous Part**: [Part 8: Causal Attention](https://medium.com/@soloshun/building-llms-from-scratch-part-8-causal-attention-6e4a0578c88c)

### 📋 What We'll Cover

- **The Concept**: Why we need multiple heads.
- **Two Approaches**: The "Wrapper" method vs. the efficient "Weight Split" method.
- **The Dimensions**: Understanding `d_out`, `num_heads`, and `head_dim`.
- **The Implementation**: Building a highly efficient `MultiHeadAttention` class in PyTorch.
- **Shape Tracing**: A step-by-step walkthrough of tensor transformations.

---

## The Concept: Parallel Processing

The main goal of any attention mechanism is to convert input vectors (embeddings) into context vectors (enriched embeddings that contain information about neighbors).

In **Multi-Head Attention**, we simply run the attention mechanism multiple times in parallel.

![](../../images/L17_s1.png)
_(Image Concept: Multiple attention heads processing the same input independently)_

Each head has its own set of learnable weights (`W_query`, `W_key`, `W_value`). This allows Head 1 to specialize in finding rhymes, while Head 2 specializes in finding subject-verb agreement. Their outputs are then combined to form a rich final representation.

---

## Implementation Strategy: Wrapper vs. Weight Splits

There are two ways to code this.

### Approach 1: The "Wrapper" (Naive)

The simplest way to think about it is to just create a list of our `CausalAttention` classes from Part 8.

```python
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, num_heads, ...):
        # Create a list of independent attention instances
        self.heads = nn.ModuleList(
            [CausalAttention(...) for _ in range(num_heads)]
        )

    def forward(self, x):
        # Run each head and concatenate the results
        return torch.cat([head(x) for head in self.heads], dim=-1)
```

While this works and is easy to understand, it is **computationally inefficient**. Processing 12 separate small matrix multiplications is slower than processing one large matrix multiplication.

### Approach 2: Weight Splits (Efficient)

This is how PyTorch, TensorFlow, and FlashAttention implement it.

Instead of 12 separate layers, we create **one giant set of weight matrices** (`W_query`, `W_key`, `W_value`) with size `d_out`. We then logically "split" or "slice" the resulting tensors into multiple heads.

![](../../images/L17_s2.png)

This allows the GPU to perform a single massive matrix multiplication (which GPUs love), and then we just reshape the tensors to isolate the heads.

---

## Understanding the Dimensions

Before writing code, let's clarify the shapes. This is often the most confusing part.

Let's assume:

- `d_in` (Embedding Dimension) = **6**
- `d_out` (Total Output Dimension) = **6**
- `num_heads` = **2**

This implies that each individual head will have a dimension of:

$$
\text{head\_dim} = \frac{\text{d\_out}}{\text{num\_heads}} = \frac{6}{2} = 3
$$

So, instead of one context vector of size 6, we produce two context vectors of size 3, which are then concatenated back to size 6.

---

## Step-by-Step Implementation

Let's build the `MultiHeadAttention` class using the efficient **Weight Split** method.

### 1. Initialization

We initialize standard linear layers for Q, K, and V. Notice we check that `d_out` is divisible by `num_heads`.

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # e.g., 6 // 2 = 3

        # Linear projections for ALL heads combined
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # Output projection to combine heads at the end
        self.out_proj = nn.Linear(d_out, d_out)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
```

### 2. The Forward Pass: Splitting Heads

This is where the magic happens. We project the input `x` using our linear layers, then **reshape** to create the head dimension.

```python
    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # 1. Linear Projection
        # Shape: (b, num_tokens, d_out)
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 2. Split Heads (Reshape + Transpose)
        # We transform: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose to bring num_heads dimension forward
        # Final Shape: (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
```

**Why transpose?**
We want the dimensions to be `(batch, heads, tokens, features)`. This way, when we perform matrix multiplication, PyTorch automatically broadcasts over the `batch` and `heads` dimensions, performing the attention for each head in parallel.

### 3. Scaled Dot-Product Attention

Now we calculate attention scores. The math is identical to single-head attention, just with an extra dimension.

```python
        # 3. Compute Attention Scores
        # Matrix multiplication happens on the last two dimensions:
        # (num_tokens, head_dim) @ (head_dim, num_tokens) -> (num_tokens, num_tokens)
        attn_scores = queries @ keys.transpose(2, 3)

        # 4. Masking
        # Truncate mask to current sequence length
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 5. Softmax & Dropout
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
```

### 4. Combining Heads

Finally, we compute the context vectors and concatenate the heads back together.

```python
        # 6. Compute Context Vectors
        # Shape: (b, num_heads, num_tokens, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # 7. Concatenate Heads
        # Flatten the heads back into the d_out dimension
        # (b, num_tokens, num_heads, head_dim) -> (b, num_tokens, d_out)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        # 8. Output Projection
        context_vec = self.out_proj(context_vec)

        return context_vec
```

The `.contiguous()` call is necessary after `.transpose()` because `.view()` requires the tensor to be contiguous in memory.

---

## Trace Your Shapes!

Let's trace a concrete example to verify what happened.

**Input:** Batch size `b=1`, Sequence `num_tokens=3`, Input Dim `d_in=6`.
**Config:** `d_out=6`, `num_heads=2`. This implies `head_dim=3`.

1.  **Input**: `(1, 3, 6)`
2.  **Linear Proj**: `(1, 3, 6)` — (Projected to `d_out`)
3.  **View (Split)**: `(1, 3, 2, 3)` — (Split 6 into 2 heads of 3)
4.  **Transpose**: `(1, 2, 3, 3)` — (Heads moved to dim 1)
5.  **Attention Scores**: `(1, 2, 3, 3) @ (1, 2, 3, 3).T` $\rightarrow$ `(1, 2, 3, 3)` matrix mult.
    Result shape is `(1, 2, 3, 3)`. Wait, no!
    The dot product is between Query `(1, 2, 3, 3)` and Key `(1, 2, 3, 3)`.
    Specifically: `(tokens, head_dim) @ (head_dim, tokens)` $\rightarrow$ `(tokens, tokens)`.
    So Attention Scores shape is: **`(1, 2, 3, 3)`**?

    _Correction_: The dimensions for multiplication are `(num_tokens, head_dim)` and `(head_dim, num_tokens)`.
    So the result is `(num_tokens, num_tokens)`.
    Final Score Shape: **`(1, 2, 3, 3)`** where the last two dims are the 3x3 attention matrix.

6.  **Context Vector**: `(1, 2, 3, 3)` — (Weighted sum of values)
7.  **Transpose Back**: `(1, 3, 2, 3)` — (Heads moved back)
8.  **Flatten (Combine)**: `(1, 3, 6)` — (2 heads $\times$ 3 dims = 6)
9.  **Output Proj**: `(1, 3, 6)` — (Final linear layer)

The input was `(1, 3, 6)` and the output is `(1, 3, 6)`. The dimensions are preserved, but the information inside has been enriched by two independent attention processes!

---

## Conclusion & What's Next

We have successfully built **Multi-Head Attention** from scratch using the efficient weight-splitting technique. This is the exact mechanism used in GPT models to understand complex language.

Our LLM architecture is coming together piece by piece:

- ✅ Input Embeddings
- ✅ Positional Encodings
- ✅ Causal Multi-Head Attention
- ✅ Dropout & Layer Normalization (coming soon)
- ✅ Feed Forward Networks (coming soon)

In the **Next Part**, we will zoom out and take a **Bird's Eye View of the LLM Architecture**. We'll see how these blocks fit together to form the Transformer architecture (GPT).

See you there!
