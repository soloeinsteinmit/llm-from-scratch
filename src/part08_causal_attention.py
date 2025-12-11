# -*- coding: utf-8 -*-
"""
Building LLMs From Scratch (Part 8): Causal Attention (Masked Self-Attention)
==============================================================================

This module demonstrates causal attention (masked self-attention), which prevents
the model from "peeking" at future tokens. This is essential for autoregressive
text generation in models like GPT.

Author: Solomon Eshun
Article: https://medium.com/@soloshun/building-llms-from-scratch-part-8-causal-attention-6e4a0578c88c
Repository: https://github.com/soloeinsteinmit/llm-from-scratch
"""

import sys
import io

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class SelfAttention_v2(nn.Module):
    """
    Self-attention with trainable weights (from Part 7).
    
    This version doesn't have causal masking, so it can see all tokens
    including future ones. We'll use this to demonstrate the problem.
    """
    
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
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


class CausalAttention(nn.Module):
    """
    Causal self-attention with trainable weights and masking.
    
    This implementation prevents tokens from attending to future positions,
    which is essential for autoregressive language modeling (like GPT).
    
    Key features:
    - Causal mask (upper triangular) to hide future tokens
    - Dropout for regularization
    - Batch support with dynamic sequence lengths
    """
    
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        """
        Initialize causal attention.
        
        Args:
            d_in (int): Input embedding dimension
            d_out (int): Output dimension
            context_length (int): Maximum sequence length supported
            dropout (float): Dropout rate (0.0 to 1.0)
            qkv_bias (bool): Whether to use bias in Q, K, V projections
        """
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        
        # Register causal mask as a buffer (non-trainable, but part of model state)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
        
    def forward(self, x):
        """
        Forward pass with causal masking.
        
        Args:
            x: Input tensor of shape (batch_size, num_tokens, d_in)
            
        Returns:
            context_vectors: Output tensor of shape (batch_size, num_tokens, d_out)
        """
        b, num_tokens, d_in = x.shape
        
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        # Compute attention scores
        # Note: transpose(1, 2) swaps the sequence and embedding dimensions
        # to enable batch matrix multiplication
        attn_scores = queries @ keys.transpose(1, 2)
        
        # Apply causal mask (hide future tokens)
        # We slice the mask to match the current sequence length
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens],
            -torch.inf
        )
        
        # Scale and apply softmax
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        
        # Apply dropout
        attn_weights = self.dropout(attn_weights)
        
        # Compute context vectors
        context_vec = attn_weights @ values
        return context_vec


def demonstrate_masking_problem():
    """
    Demonstrate why we need to mask BEFORE softmax, not after.
    """
    print("🔬 Why Mask BEFORE Softmax?")
    print("=" * 60)
    
    # Sample attention scores
    attn_scores = torch.tensor([
        [0.5, 0.8, 0.2],
        [0.6, 0.4, 0.3],
        [0.2, 0.3, 0.9]
    ])
    
    context_length = attn_scores.shape[0]
    
    print("Original attention scores:")
    print(attn_scores)
    
    # Method 1: Mask AFTER softmax (WRONG - causes data leakage)
    print("\n❌ Method 1: Masking AFTER softmax (WRONG)")
    print("-" * 60)
    
    attn_weights = torch.softmax(attn_scores, dim=-1)
    print("After softmax:")
    print(attn_weights)
    
    # Create lower triangular mask
    mask_simple = torch.tril(torch.ones(context_length, context_length))
    masked_after = attn_weights * mask_simple
    print("\nAfter masking:")
    print(masked_after)
    
    # Renormalize
    row_sums = masked_after.sum(dim=1, keepdim=True)
    masked_after_norm = masked_after / row_sums
    print("\nAfter renormalization:")
    print(masked_after_norm)
    print("\n⚠️  Problem: The softmax denominator included future tokens,")
    print("   causing subtle data leakage!")
    
    # Method 2: Mask BEFORE softmax (CORRECT)
    print("\n\n✅ Method 2: Masking BEFORE softmax (CORRECT)")
    print("-" * 60)
    
    # Create upper triangular mask (1s = positions to mask)
    mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
    print("Mask (1 = hide, 0 = keep):")
    print(mask)
    
    # Mask with -inf
    masked_scores = attn_scores.masked_fill(mask.bool(), -torch.inf)
    print("\nScores after masking with -inf:")
    print(masked_scores)
    
    # Apply softmax
    attn_weights_correct = torch.softmax(masked_scores, dim=-1)
    print("\nAfter softmax:")
    print(attn_weights_correct)
    print("\n✅ Correct! Future tokens had NO influence on the probabilities.")


def demonstrate_dropout():
    """
    Demonstrate dropout on attention weights.
    """
    print("\n🎲 Dropout in Attention")
    print("=" * 60)
    
    # Create sample attention weights
    attn_weights = torch.tensor([
        [1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.4411, 0.5589, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.2846, 0.3595, 0.3559, 0.0000, 0.0000, 0.0000],
        [0.2361, 0.2699, 0.2683, 0.2257, 0.0000, 0.0000],
        [0.1951, 0.2179, 0.2169, 0.1882, 0.1820, 0.0000],
        [0.1604, 0.1907, 0.1893, 0.1511, 0.1432, 0.1653]
    ])
    
    print("Original attention weights:")
    print(attn_weights)
    
    # Apply dropout
    torch.manual_seed(123)
    dropout = nn.Dropout(0.5)
    dropped = dropout(attn_weights)
    
    print("\nAfter dropout (p=0.5):")
    print(dropped)
    
    print("\n💡 Key points:")
    print("- Dropout randomly zeros out 50% of the weights")
    print("- Remaining weights are scaled by 2x (1/(1-p))")
    print("- This prevents overfitting during training")
    print("- During inference, dropout is turned off")


def visualize_causal_mask(context_length=6):
    """
    Visualize the causal attention mask.
    
    Args:
        context_length (int): Sequence length
    """
    # Create mask
    mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Mask visualization
    im1 = ax1.imshow(mask.numpy(), cmap='RdYlGn_r', vmin=0, vmax=1)
    ax1.set_title('Causal Mask\n(0 = allowed, 1 = masked)', fontsize=14, pad=20)
    ax1.set_xlabel('Key Position (attending TO)')
    ax1.set_ylabel('Query Position (attending FROM)')
    
    # Add text annotations
    for i in range(context_length):
        for j in range(context_length):
            text = ax1.text(j, i, 'X' if mask[i, j] == 1 else '✓',
                          ha="center", va="center", color="white", fontsize=16, weight='bold')
    
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    
    # Plot 2: Attention weights with mask
    # Sample attention scores
    torch.manual_seed(123)
    sample_scores = torch.randn(context_length, context_length)
    masked_scores = sample_scores.masked_fill(mask.bool(), -torch.inf)
    attn_weights = torch.softmax(masked_scores, dim=-1)
    
    im2 = ax2.imshow(attn_weights.numpy(), cmap='Blues', vmin=0, vmax=1)
    ax2.set_title('Causal Attention Weights\n(after masking & softmax)', fontsize=14, pad=20)
    ax2.set_xlabel('Key Position')
    ax2.set_ylabel('Query Position')
    
    # Add text annotations
    for i in range(context_length):
        for j in range(context_length):
            text = ax2.text(j, i, f'{attn_weights[i, j]:.2f}',
                          ha="center", va="center",
                          color="white" if attn_weights[i, j] > 0.5 else "black",
                          fontsize=10)
    
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    
    plt.tight_layout()
    plt.show()


def demo_basic_causal_attention():
    """
    Demonstrate basic causal attention with the standard example.
    """
    print("🚀 Causal Attention Demo")
    print("=" * 60)
    
    # Setup
    sentence = "Your journey starts with one step"
    tokens = sentence.split()
    
    inputs = torch.tensor([
        [0.43, 0.15, 0.89], # Your
        [0.55, 0.87, 0.66], # journey
        [0.57, 0.85, 0.64], # starts
        [0.22, 0.58, 0.33], # with
        [0.77, 0.25, 0.10], # one
        [0.05, 0.80, 0.55], # step
    ])
    
    d_in = inputs.shape[-1]
    d_out = 2
    
    print(f"📝 Input sentence: '{sentence}'")
    print(f"🔢 Tokens: {tokens}")
    print(f"🧠 Input shape: {inputs.shape}")
    
    # Compare standard attention vs causal attention
    print(f"\n" + "="*60)
    print("📊 Comparing Standard Attention vs Causal Attention")
    print("="*60)
    
    # Standard attention (can see future)
    torch.manual_seed(789)
    sa = SelfAttention_v2(d_in, d_out)
    output_standard = sa(inputs)
    
    print("\n❌ Standard Attention (sees future tokens):")
    print(f"Output shape: {output_standard.shape}")
    print(f"Output:\n{output_standard}")
    
    # Causal attention (cannot see future)
    torch.manual_seed(789)
    ca = CausalAttention(d_in, d_out, context_length=len(tokens), dropout=0.0)
    output_causal = ca(inputs.unsqueeze(0))  # Add batch dimension
    
    print("\n✅ Causal Attention (respects time):")
    print(f"Output shape: {output_causal.shape}")
    print(f"Output:\n{output_causal[0]}")
    
    print("\n💡 Notice the outputs are different!")
    print("   Causal attention prevents future leakage.")


def demo_batched_causal_attention():
    """
    Demonstrate causal attention with batched inputs.
    """
    print("\n🔢 Batched Causal Attention Demo")
    print("=" * 60)
    
    # Setup
    inputs = torch.tensor([
        [0.43, 0.15, 0.89],
        [0.55, 0.87, 0.66],
        [0.57, 0.85, 0.64],
        [0.22, 0.58, 0.33],
        [0.77, 0.25, 0.10],
        [0.05, 0.80, 0.55],
    ])
    
    # Create a batch (duplicate the inputs)
    batch = torch.stack((inputs, inputs), dim=0)
    
    print(f"Batch shape: {batch.shape}")
    print(f"  - Batch size: {batch.shape[0]}")
    print(f"  - Sequence length: {batch.shape[1]}")
    print(f"  - Embedding dim: {batch.shape[2]}")
    
    # Apply causal attention
    torch.manual_seed(123)
    ca = CausalAttention(d_in=3, d_out=2, context_length=6, dropout=0.0)
    context_vecs = ca(batch)
    
    print(f"\nOutput shape: {context_vecs.shape}")
    print(f"Output:\n{context_vecs}")
    
    print("\n✅ Causal attention handles batches seamlessly!")


def demo():
    """
    Run the complete demonstration of causal attention.
    """
    print("🚀 Building LLMs From Scratch (Part 8):")
    print("   Causal Attention (Masked Self-Attention)\n")
    
    # Demonstrate the masking problem
    demonstrate_masking_problem()
    
    print(f"\n" + "="*60)
    
    # Demonstrate dropout
    demonstrate_dropout()
    
    print(f"\n" + "="*60)
    
    # Basic demo
    demo_basic_causal_attention()
    
    print(f"\n" + "="*60)
    
    # Batched demo
    demo_batched_causal_attention()
    
    print(f"\n" + "="*60)
    print("🎯 Key Takeaways:")
    print("- Causal attention prevents future token leakage")
    print("- Masking MUST be done BEFORE softmax")
    print("- We use -inf to mask out future positions")
    print("- register_buffer stores the mask with the model")
    print("- Dropout adds regularization to prevent overfitting")
    
    print(f"\n" + "="*60)
    print("🎉 Demo complete!")
    print("\nCheck out the full tutorial:")
    print("📝 Medium: https://medium.com/@soloshun/building-llms-from-scratch-part-8-causal-attention-6e4a0578c88c")
    print("📂 GitHub: https://github.com/soloeinsteinmit/llm-from-scratch")
    print("📓 Interactive notebook: notebooks/part08_causal_attention.ipynb")
    
    print("\n🔜 Next up: Part 9 - Multi-Head Attention")


if __name__ == "__main__":
    demo()

