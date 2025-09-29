"""
Building LLMs From Scratch (Part 4): The Embedding Layer
==========================================================

This module contains a clean, reusable implementation of the combined token
and positional embedding layer for a GPT-style model.

Author: Solomon Eshun
Article: https://soloshun.medium.com/link-to-part-4
Repository: https://github.com/soloeinsteinmit/llm-from-scratch
"""

import torch
import torch.nn as nn
import tiktoken
import sys

# Add the parent directory to the system path to allow imports
sys.path.insert(0, '../')
from src.part03_dataloader import create_dataloader_v1

class GPTEmbedding(nn.Module):
    """
    The embedding layer for a GPT-model.

    This module combines token embeddings and positional embeddings into a
    single input representation.
    """
    def __init__(self, vocab_size, emb_dim, context_size):
        """
        Initialize the embedding layers.

        Args:
            vocab_size (int): The size of the vocabulary.
            emb_dim (int): The dimensionality of the embeddings.
            context_size (int): The maximum sequence length.
        """
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(context_size, emb_dim)

    def forward(self, token_ids):
        """
        Forward pass for the embedding layer.

        Args:
            token_ids (torch.Tensor): A tensor of token IDs of shape
                                      (batch_size, seq_len).

        Returns:
            torch.Tensor: The combined token and positional embeddings of
                          shape (batch_size, seq_len, emb_dim).
        """
        batch_size, seq_len = token_ids.shape
        
        # Get token embeddings
        tok_embeds = self.tok_emb(token_ids)
        
        # Create positional IDs and get positional embeddings
        pos_ids = torch.arange(seq_len, device=token_ids.device)
        pos_embeds = self.pos_emb(pos_ids)
        
        # Add the two embeddings together (broadcasting handles the batch dim)
        input_embeds = tok_embeds + pos_embeds
        return input_embeds

def demo_embedding_lookup():
    """
    Demonstrate the embedding lookup process step by step.
    """
    print("🔍 Embedding Lookup Process Demo")
    print("=" * 50)
    
    # Create a simple embedding layer for demonstration
    simple_vocab_size = 6
    simple_emb_dim = 3
    torch.manual_seed(123)
    simple_embedding = nn.Embedding(simple_vocab_size, simple_emb_dim)
    
    print(f"Embedding matrix shape: {simple_embedding.weight.shape}")
    print(f"Embedding matrix:\n{simple_embedding.weight}\n")
    
    # Demonstrate lookup
    test_tokens = torch.tensor([2, 3, 4, 1])
    print(f"Looking up tokens: {test_tokens}")
    
    # Method 1: Using embedding layer
    embeddings1 = simple_embedding(test_tokens)
    print(f"\nUsing embedding layer: {embeddings1.shape}")
    print(embeddings1)
    
    # Method 2: Manual lookup
    embeddings2 = simple_embedding.weight[test_tokens]
    print(f"\nManual lookup: {embeddings2.shape}")
    print(embeddings2)
    
    print(f"\nAre they identical? {torch.equal(embeddings1, embeddings2)}")
    print("✅ Embedding is just a lookup table!\n")


def demo_positional_embeddings():
    """
    Demonstrate positional embeddings and broadcasting.
    """
    print("📍 Positional Embeddings Demo")
    print("=" * 50)
    
    context_size = 4
    emb_dim = 6
    batch_size = 2
    
    # Create positional embedding layer
    pos_emb = nn.Embedding(context_size, emb_dim)
    
    # Create some fake token embeddings
    token_embeddings = torch.randn(batch_size, context_size, emb_dim)
    
    # Get positional embeddings
    pos_ids = torch.arange(context_size)
    pos_embeddings = pos_emb(pos_ids)
    
    print(f"Token embeddings shape: {token_embeddings.shape}")
    print(f"Positional embeddings shape: {pos_embeddings.shape}")
    
    # Add them together (broadcasting)
    combined = token_embeddings + pos_embeddings
    print(f"Combined embeddings shape: {combined.shape}")
    
    print("\n✨ Broadcasting automatically expanded positional embeddings!")
    print(f"Same positional info added to all {batch_size} batches\n")


def demo():
    """
    A comprehensive demonstration of the GPTEmbedding layer.
    """
    print("🚀 Building LLMs From Scratch (Part 4): The Embedding Layer Demo\n")
    
    # --- 1. Define Hyperparameters ---
    vocab_size = 50257
    emb_dim = 256
    context_size = 4
    
    print("Configuration:")
    print(f"  - Vocabulary Size: {vocab_size:,}")
    print(f"  - Embedding Dimension: {emb_dim}")
    print(f"  - Context Size: {context_size}")
    print(f"  - Embedding Matrix Shape: [{vocab_size:,}, {emb_dim}]")
    print(f"  - Positional Matrix Shape: [{context_size}, {emb_dim}]\n")

    # --- 2. Create the Embedding Layer ---
    torch.manual_seed(123)
    embedding_layer = GPTEmbedding(vocab_size, emb_dim, context_size)
    
    total_params = sum(p.numel() for p in embedding_layer.parameters())
    print(f"✅ GPTEmbedding layer created!")
    print(f"Total parameters: {total_params:,}\n")

    # --- 3. Create a DataLoader ---
    try:
        with open("../data/the-verdict.txt", 'r', encoding="utf-8") as f:
            raw_text = f.read()
    except FileNotFoundError:
        # Fallback for different directory structures
        raw_text = "I HAD always thought Jack Gisburn rather a cheap genius"
    
    dataloader = create_dataloader_v1(
        raw_text,
        batch_size=8,
        max_length=context_size,
        stride=context_size,
        shuffle=False
    )
    print("📊 DataLoader created")

    # --- 4. Get a batch of data ---
    data_iter = iter(dataloader)
    inputs, _ = next(data_iter)
    
    print(f"\nInput token IDs shape: {inputs.shape}")
    print(f"Sample token IDs: {inputs[0]}")

    # --- 5. Forward pass ---
    output_embeddings = embedding_layer(inputs)
    
    print(f"\n🎯 Results:")
    print(f"Output embeddings shape: {output_embeddings.shape}")
    print(f"Each token ID → {emb_dim}-dimensional vector with positional info")
    
    # Show that different positions have different embeddings even for same token
    if inputs.shape[1] >= 2:
        token_0_pos_0 = output_embeddings[0, 0]  # First token, position 0
        token_0_pos_1 = output_embeddings[0, 1]  # Second token, position 1
        
        # If the same token appears in different positions
        if inputs[0, 0] == inputs[0, 1]:
            print(f"\nSame token in different positions:")
            print(f"Token {inputs[0, 0]} at position 0 vs position 1")
            print(f"Are embeddings identical? {torch.equal(token_0_pos_0, token_0_pos_1)}")
            print("✅ Different positions → different embeddings!")

    print("\n" + "=" * 60)
    print("🎉 Demo complete!")
    print("\nNext: Part 5 - Self-Attention Mechanism")


def demo_step_by_step():
    """
    Run all the individual demos step by step.
    """
    print("📚 Step-by-Step Embedding Demos\n")
    
    demo_embedding_lookup()
    print()
    demo_positional_embeddings()
    print()
    demo()

if __name__ == "__main__":
    demo()
