---
title: "Building LLMs From Scratch (Part 7): Self-Attention with Trainable Weights"
description: "We take the next big step in building our LLM: upgrading our simple attention mechanism with trainable weights. We'll introduce the Query, Key, and Value matrices—the components that allow the model to actually 'learn' language relationships."
tags: [LLM, AI, Machine Learning, Python, PyTorch, Deep Learning]
thumbnail: "images/L15_s1.png"
---

# Building LLMs From Scratch (Part 7): Self-Attention with Trainable Weights

Welcome back to the "LLMs From Scratch" series! In [Part 6](https://medium.com/@soloshun/building-llms-from-scratch-part-6-the-attention-mechanism-123456), we built a **simplified self-attention mechanism**. It was a great starting point: it allowed us to take an input sentence like "Your journey starts with one step" and calculate a "context vector" for each word based on its similarity to other words.

But there was a catch: **it couldn't learn.**

Our simplified version relied solely on the fixed input embeddings to calculate similarity. If the embeddings said "cat" and "dog" were similar, the attention mechanism would _always_ focus on them equally, regardless of the context or the task. It had no way to adjust, to "pay attention" differently depending on whether it was translating a sentence, summarizing a text, or writing a poem.

Today, we fix that. We are going to introduce **trainable weights**—the "brain" of the attention mechanism.

### 🔗 Quick Links

- **GitHub Repository**: [llm-from-scratch](https://github.com/soloeinsteinmit/llm-from-scratch)
- **Previous Part**: [Part 6: The Attention Mechanism](https://medium.com/@soloshun/building-llms-from-scratch-part-6-the-attention-mechanism-123456)

### 📋 What We'll Cover

- **The Missing Piece**: Why we need trainable weights.
- **The Holy Trinity**: Queries (Q), Keys (K), and Values (V).
- **The Search Engine Analogy**: An intuitive way to understand Q, K, and V.
- **Step-by-Step Implementation**: Building the mathematical pipeline in PyTorch.
- **Scaled Dot-Product Attention**: Why we divide by square roots.
- **Refactoring**: Moving from raw parameters to `nn.Linear`.

---

## The Goal: Context Vectors

Let's ground ourselves in our main objective again.

> **The goal of self-attention is to convert `input vectors` into enriched `context vectors`.**

An input vector (embedding) represents the static meaning of a word. A **context vector** is an enriched version that captures the word's meaning _in the context of the specific sentence_.

In Part 6, we computed this context vector just by averaging input vectors based on similarity. Now, we want the model to **learn** how to construct these context vectors. To do that, we need **trainable weight matrices**.

---

## Enter Query, Key, and Value

In modern transformers (like GPT), we don't just use the input vectors directly to compute attention. Instead, for every input token `x`, we project it into three different vectors:

1.  **Query (Q)**
2.  **Key (K)**
3.  **Value (V)**

We do this by multiplying the input `x` by three separate weight matrices: $W_q$, $W_k$, and $W_v$. These matrices are initialized randomly and are **updated during training**. This is how the model learns!

### The Intuition: The Search Engine Analogy

Why do we need three vectors? Let's use a **Search Engine** analogy.

Imagine you are searching for something on Google.

1.  **Query (Q)**: This is the text you type into the search bar (e.g., "best Italian restaurants"). It represents **what you are looking for**.
2.  **Key (K)**: This is the metadata or index associated with every web page in the database. It represents **what the content is about**.
3.  **Value (V)**: This is the actual content of the web page itself. It represents **the information you get back**.

**The Process:**
The search engine compares your **Query** against the **Keys** of all the pages. If your Query matches a Key strongly (high attention score), the search engine serves you the corresponding **Value**.

In Self-Attention:

- Each token generates a **Query** ("What am I looking for?").
- Each token also generates a **Key** ("What do I contain?").
- Each token also generates a **Value** ("What information do I pass on?").

The model checks if Token A's **Query** matches Token B's **Key**. If it does, Token A pays attention to Token B and absorbs some of Token B's **Value**.

---

## Step-by-Step Implementation

Let's implement this in PyTorch. We'll use the same input sentence: "Your journey starts with one step".

### Step 1: Initialize Inputs and Weights

First, let's define our inputs and the three weight matrices ($W_q$, $W_k$, $W_v$).

```python
import torch
import torch.nn as nn

# Input sentence: "Your journey starts with one step" (6 words, embedding dim 3)
inputs = torch.tensor(
  [
   [0.43, 0.15, 0.89], # Your
   [0.55, 0.87, 0.66], # journey
   [0.57, 0.85, 0.64], # starts
   [0.22, 0.58, 0.33], # with
   [0.77, 0.25, 0.10], # one
   [0.05, 0.80, 0.55], # step
  ]
)

d_in = inputs.shape[1] # Input dimension = 3
d_out = 2              # Output dimension (we can project to a different size)

torch.manual_seed(123)

# Initialize the three weight matrices
# Note: In practice, we don't usually set requires_grad=False, but we do it here
# to keep the example static.
W_query = nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key   = nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
```

### Step 2: Compute Q, K, and V

Now, we project our inputs into Queries, Keys, and Values.

```python
# Compute Queries, Keys, and Values for all inputs
queries = inputs @ W_query
keys    = inputs @ W_key
values  = inputs @ W_value

print("Queries shape:", queries.shape)
# Output: torch.Size([6, 2]) -> 6 tokens, projected to 2 dimensions
```

We have now transformed our original embeddings into three specialized representations.

### Step 3: Compute Attention Scores

Just like in the simplified version, we calculate relevance using the **dot product**. We compare every **Query** with every **Key**.

```python
# Compute attention scores (omega)
# (6x2) @ (2x6) -> (6x6)
attn_scores = queries @ keys.T

print(attn_scores)
```

This gives us a 6x6 matrix where cell $(i, j)$ tells us how much the $i$-th token's Query matches the $j$-th token's Key.

### Step 4: Scale the Scores (Scaled Dot-Product Attention)

Here is a new step we didn't have before. We divide the attention scores by the square root of the key dimension ($d_k$).

```python
d_k = keys.shape[-1]
attn_scores_scaled = attn_scores / (d_k ** 0.5)
```

**Why do we do this?**

1.  **Stability**: As the dimension $d_k$ gets larger, the dot products can become huge.
2.  **Gradient Flow**: Large inputs to the softmax function cause it to become extremely "peaky" (one value is 1, the rest are 0). When this happens, the gradients during backpropagation become tiny (vanishing gradients), effectively killing the learning process. Scaling keeps the variance stable (close to 1), ensuring the model learns efficiently.

### Step 5: Normalize with Softmax

Now we apply softmax to get probabilities (attention weights).

```python
attn_weights = torch.softmax(attn_scores_scaled, dim=-1)

print("Attention Weights sum:", attn_weights[0].sum())
    # Output: tensor(1.0000)
```

### Step 6: Compute Context Vectors

Finally, we use these attention weights to aggregate the **Values**.

```python
# (6x6) @ (6x2) -> (6x2)
context_vectors = attn_weights @ values

print(context_vectors)
```

And there we have it! `context_vectors` contains the new, context-enriched representations for our sentence, learned through the interaction of Q, K, and V.

---

## Putting It All Together: A Clean Class

We can wrap all of this into a nice PyTorch class. We have two ways to do this:

### Option 1: Using `nn.Parameter` (The Hard Way)

This gives us full control but requires us to manage the matrix multiplication manually.

```python
class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value

        attn_scores = queries @ keys.T
        # Scale
        attn_scores = attn_scores / keys.shape[-1]**0.5

        attn_weights = torch.softmax(attn_scores, dim=-1)

        context_vec = attn_weights @ values
        return context_vec
```

### Option 2: Using `nn.Linear` (The Pro Way)

In practice, we almost always use `nn.Linear`. It handles the weights (and optional biases) for us and uses optimized underlying code for better performance.

```python
class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        # nn.Linear replaces the manual nn.Parameter + matmul
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        context_vec = attn_weights @ values
        return context_vec
```

This `SelfAttention_v2` class is the foundation of the attention mechanism used in GPT-2 and other major LLMs.

---

## Conclusion

We've successfully upgraded our attention mechanism! By introducing **Query, Key, and Value matrices**, we've given our model the ability to _learn_ which words should attend to which. We also added **scaling** to ensure training stability.

However, there's still a small issue. Right now, our model "cheats" a little bit. When it's looking at the word "journey", it can see the word "step" which comes later in the sentence. In tasks like text generation, we want the model to predict the _next_ word without knowing the future.

In **Part 8**, we will tackle this by implementing **Causal Attention** (also known as Masked Attention), which forces the model to respect the arrow of time.

See you in the next part!
