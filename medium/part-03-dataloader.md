---
title: "Building LLMs From Scratch (Part 3): The Data Pipeline"
date: "2024-09-20"
tags:
  [
    "AI",
    "Machine Learning",
    "LLM",
    "PyTorch",
    "Deep Learning",
    "NLP",
    "Data Engineering",
  ]
series: "Building LLMs from Scratch"
---

# Building LLMs From Scratch (Part 3): Crafting the Data Pipeline

In [Part 1](https://soloshun.medium.com/building-llms-from-scratch-part-1-the-complete-theoretical-foundation-e66b45b7f379), we built the theoretical blueprint. In [Part 2](https://soloshun.medium.com/building-llms-from-scratch-part-2-the-power-of-tokenization), we learned how to convert raw text into a sequence of numerical tokens.

Now, we've arrived at a crucial, often overlooked, step: **how do we feed this data to our model?**

A model doesn't learn from a giant, continuous stream of tokens. It learns from carefully prepared examples. In this part, we'll build the essential machinery—a **data loader**—that transforms our token sequence into the `(input, target)` pairs that our future transformer model will use to learn.

---

## Chapter 1: The Core Task - Predicting the Next Token

At its heart, a GPT-style Large Language Model is a **next-token predictor**. Its one and only job is to look at a sequence of tokens and guess what token comes next.

To teach it this skill, we need to show it millions of examples. Each example consists of:

1.  **An Input:** A chunk of text (a sequence of tokens).
2.  **A Target:** The very next token that follows the input.

This is what we call an **input-target pair**.

![Input-Target Pairs](../../images/input_target_l9.png)

Our goal is to create a pipeline that can automatically generate thousands of these pairs from our source text.

## Chapter 2: Key Concepts - Context Size & Stride

To create these pairs systematically, we need to define two key parameters.

### 1. Context Size (or `max_length`)

The **context size** is the exact number of tokens the model looks at to make a prediction. It's the model's "attention span" or "memory."

- If the context size is **8**, the model will always receive **8 tokens** as input to predict the **9th token**.

During training, we slide a "window" of this size across our text to generate our input sequences.

### 2. Stride

The **stride** is the number of tokens we slide the window forward to get our next chunk.

- If **stride = 1**, the next input chunk will overlap almost entirely with the previous one (e.g., `tokens[0:8]`, then `tokens[1:9]`, etc.). This creates a huge number of training examples but can be computationally intensive.
- If **stride = context size**, there is no overlap. The chunks are sequential (e.g., `tokens[0:8]`, then `tokens[8:16]`, etc.). This is much faster and helps prevent the model from simply memorizing sequences.

![Stride Visualization](../../images/STRIDE_L9.png)

## Chapter 3: Building the Data Pipeline with PyTorch

Now, let's translate these concepts into code using PyTorch's powerful data-handling tools: `Dataset` and `DataLoader`.

### Step 1: The `Dataset` Class

A `Dataset` class in PyTorch is a blueprint that tells Python how to access our data. It needs three key methods:

1.  `__init__(self, ...)`: This is where we do our one-time setup. We'll tokenize the entire text and use the sliding window to create all possible input-target pairs, storing them in memory.
2.  `__len__(self)`: This simply returns the total number of input-target pairs we created.
3.  `__getitem__(self, idx)`: This is the magic function. It fetches a single input-target pair at a specific index `idx`.

Here's our implementation, which we'll save in `llm-from-scratch/src/part03_dataloader.py`:

```python
import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # 1. Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # 2. Use a sliding window to create input-target chunks
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        # 3. Return the total number of chunks
        return len(self.input_ids)

    def __getitem__(self, idx):
        # 4. Return a single input-target pair
        return self.input_ids[idx], self.target_ids[idx]
```

### Step 2: The `DataLoader`

While the `Dataset` knows _how_ to get a single data point, the `DataLoader` is the utility that actually serves up the data. It takes our `Dataset` and automatically:

- **Batches** the data together (e.g., grabs 8 pairs at once).
- **Shuffles** the data to ensure the model doesn't learn the order of the text.
- Uses parallel processes (`num_workers`) to speed things up.

We'll create a simple utility function to wrap this logic:

```python
def create_dataloader_v1(
    txt,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create the dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create the dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )

    return dataloader
```

## Chapter 4: Seeing it in Action

Let's use these tools to see what the final output looks like. We'll use a tiny `max_length=4` and `batch_size=8` to make it easy to inspect.

```python
# Load the text data
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Create a dataloader with a small context size
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=4, stride=4, shuffle=False
)

# Fetch one batch of data
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

print("Inputs:\n", inputs)
print("\nTargets:\n", targets)
```

**Output:**

```
Inputs:
 tensor([[   40,   367,  2885,  1464],
        [ 1807,  3619,   402,   271],
        [10899,  2138,   257,  7026],
        [15632,   438,  2016,   257],
        [  922,  5891,  1576,   438],
        [  568,   340,   373,   645],
        [ 1049,  5975,   284,   502],
        [  284,  3285,   326,    11]])

Targets:
 tensor([[  367,  2885,  1464,  1807],
        [ 3619,   402,   271, 10899],
        [ 2138,   257,  7026, 15632],
        [  438,  2016,   257,   922],
        [ 5891,  1576,   438,   568],
        [  340,   373,   645,  1049],
        [ 5975,   284,   502,   284],
        [ 3285,   326,    11,   287]])
```

Look closely! The `Targets` tensor is exactly the `Inputs` tensor shifted one position to the left. This is the precise format our model needs for training. For each row in the `Inputs`, the model will learn to predict the corresponding row in the `Targets`.

---

## Conclusion & What's Next

We've successfully built a robust data pipeline! This is a huge step. We can now take any raw text and efficiently create batches of training data.

With our data pipeline ready, we can finally turn our attention to the heart of the LLM: the model itself.

**In Part 4**, we'll dive into the concept of **embeddings**—how we turn our token IDs into meaningful vectors that capture semantic relationships—and build our first neural network layer.

---

## 🧑‍💻 Full Code

You can find the complete, interactive code for this part in our GitHub repository:

- **Jupyter Notebook**: [llm-from-scratch/notebooks/part03_dataloader.ipynb](https://github.com/soloeinsteinmit/llm-from-scratch/blob/main/notebooks/part03_dataloader.ipynb)
- **Python Script**: [llm-from-scratch/src/part03_dataloader.py](https://github.com/soloeinsteinmit/llm-from-scratch/blob/main/src/part03_dataloader.py)
