# -*- coding: utf-8 -*-
"""
Building LLMs From Scratch (Part 9): Multi-Head Attention
==========================================================

This module demonstrates multi-head attention, the production-ready attention
mechanism used in modern Transformers like GPT-4 and Llama.

We implement both the "wrapper" approach (naive but clear) and the efficient
"weight split" approach (used in production).

Author: Solomon Eshun
Article: https://medium.com/@soloshun/building-llms-from-scratch-part-9-multi-head-attention
Repository: https://github.com/soloeinsteinmit/llm-from-scratch
"""

import sys
import io

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


class CausalAttention(nn.Module):
    """
    Single-head causal attention (from Part 8).
    
    This serves as the building block for the wrapper approach.
    """
    
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
        
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens],
            -torch.inf
        )
        
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)
        
        context_vec = attn_weights @ values
        return context_vec


class MultiHeadAttentionWrapper(nn.Module):
    """
    Multi-Head Attention using the "Wrapper" approach.
    
    This is the naive but intuitive implementation where we create
    multiple independent CausalAttention instances and concatenate
    their outputs.
    
    Pros: Easy to understand
    Cons: Less efficient (many small matrix multiplications)
    """
    
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) 
             for _ in range(num_heads)]
        )
        
    def forward(self, x):
        # Run each head independently and concatenate
        return torch.cat([head(x) for head in self.heads], dim=-1)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention using the efficient "Weight Split" approach.
    
    This is the production implementation used in PyTorch, TensorFlow,
    and all modern Transformer libraries.
    
    Instead of creating separate attention layers, we:
    1. Create ONE large set of Q, K, V weights
    2. Reshape to split into multiple heads
    3. Process all heads in parallel with a single matrix multiplication
    4. Concatenate heads back together
    
    This is much faster on GPUs.
    """
    
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Dimension per head
        
        # Single large weight matrices for all heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        
        # Output projection to combine heads
        self.out_proj = nn.Linear(d_out, d_out)
        
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
        
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        
        # 1. Linear Projection (for ALL heads at once)
        keys = self.W_key(x)       # (b, num_tokens, d_out)
        queries = self.W_query(x)  # (b, num_tokens, d_out)
        values = self.W_value(x)   # (b, num_tokens, d_out)
        
        # 2. Reshape to split into multiple heads
        # (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        
        # 3. Transpose to group by heads
        # (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        
        # 4. Compute attention scores (for all heads in parallel!)
        attn_scores = queries @ keys.transpose(2, 3)
        
        # 5. Apply causal mask
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        # 6. Softmax and dropout
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 7. Compute context vectors
        context_vec = attn_weights @ values  # (b, num_heads, num_tokens, head_dim)
        
        # 8. Transpose back and flatten heads
        context_vec = context_vec.transpose(1, 2)  # (b, num_tokens, num_heads, head_dim)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        
        # 9. Output projection
        context_vec = self.out_proj(context_vec)
        
        return context_vec


def demo_wrapper_approach():
    """
    Demonstrate the wrapper approach to multi-head attention.
    """
    print("🔄 Multi-Head Attention: Wrapper Approach")
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
    
    batch = torch.stack((inputs, inputs), dim=0)
    
    print(f"Input shape: {batch.shape}")
    print(f"  - Batch size: {batch.shape[0]}")
    print(f"  - Sequence length: {batch.shape[1]}")
    print(f"  - Embedding dim: {batch.shape[2]}")
    
    # Create wrapper multi-head attention
    torch.manual_seed(123)
    d_in, d_out = 3, 2
    num_heads = 2
    
    mha_wrapper = MultiHeadAttentionWrapper(
        d_in=d_in,
        d_out=d_out,
        context_length=batch.shape[1],
        dropout=0.0,
        num_heads=num_heads
    )
    
    context_vecs = mha_wrapper(batch)
    
    print(f"\nOutput shape: {context_vecs.shape}")
    print(f"  - Note: d_out ({d_out}) × num_heads ({num_heads}) = {d_out * num_heads}")
    print(f"\nOutput:\n{context_vecs}")
    
    print("\n✅ Wrapper approach: Each head processes independently,")
    print("   then outputs are concatenated along the feature dimension.")


def demo_efficient_approach():
    """
    Demonstrate the efficient weight-split approach to multi-head attention.
    """
    print("\n\n⚡ Multi-Head Attention: Efficient Weight Split Approach")
    print("=" * 60)
    
    # Setup (same as wrapper for comparison)
    inputs = torch.tensor([
        [0.43, 0.15, 0.89],
        [0.55, 0.87, 0.66],
        [0.57, 0.85, 0.64],
        [0.22, 0.58, 0.33],
        [0.77, 0.25, 0.10],
        [0.05, 0.80, 0.55],
    ])
    
    batch = torch.stack((inputs, inputs), dim=0)
    
    print(f"Input shape: {batch.shape}")
    
    # Create efficient multi-head attention
    torch.manual_seed(789)
    d_in, d_out = 3, 6  # Note: d_out must be divisible by num_heads
    num_heads = 2
    
    mha = MultiHeadAttention(
        d_in=d_in,
        d_out=d_out,
        context_length=batch.shape[1],
        dropout=0.0,
        num_heads=num_heads
    )
    
    context_vecs = mha(batch)
    
    print(f"\nOutput shape: {context_vecs.shape}")
    print(f"  - d_out remains {d_out} (same as input)")
    print(f"  - head_dim = d_out / num_heads = {d_out} / {num_heads} = {d_out // num_heads}")
    print(f"\nOutput:\n{context_vecs}")
    
    print("\n✅ Efficient approach: Single large matrix multiplication,")
    print("   then reshape to split heads. Much faster on GPUs!")


def demo_shape_tracing():
    """
    Trace the shape transformations step by step.
    """
    print("\n\n🔍 Shape Tracing: Following the Tensor Transformations")
    print("=" * 60)
    
    # Setup
    b, num_tokens, d_in = 1, 3, 6
    d_out = 6
    num_heads = 2
    head_dim = d_out // num_heads
    
    print(f"Configuration:")
    print(f"  - Batch size (b): {b}")
    print(f"  - Sequence length (num_tokens): {num_tokens}")
    print(f"  - Input dimension (d_in): {d_in}")
    print(f"  - Output dimension (d_out): {d_out}")
    print(f"  - Number of heads: {num_heads}")
    print(f"  - Head dimension: {head_dim}")
    
    # Create dummy input
    x = torch.randn(b, num_tokens, d_in)
    
    # Create model
    torch.manual_seed(42)
    mha = MultiHeadAttention(d_in, d_out, num_tokens, 0.0, num_heads)
    
    # Manual forward pass with shape printing
    print(f"\n📊 Step-by-Step Shape Transformations:")
    print(f"1. Input:           {tuple(x.shape)}")
    
    queries = mha.W_query(x)
    print(f"2. After Linear:    {tuple(queries.shape)}")
    
    queries = queries.view(b, num_tokens, num_heads, head_dim)
    print(f"3. After Reshape:   {tuple(queries.shape)}")
    
    queries = queries.transpose(1, 2)
    print(f"4. After Transpose: {tuple(queries.shape)}")
    
    # Simulate attention scores
    keys = mha.W_key(x).view(b, num_tokens, num_heads, head_dim).transpose(1, 2)
    attn_scores = queries @ keys.transpose(2, 3)
    print(f"5. Attention Scores: {tuple(attn_scores.shape)}")
    print(f"   (Last 2 dims are {num_tokens}×{num_tokens} attention matrix)")
    
    # Complete forward pass
    output = mha(x)
    print(f"6. Final Output:    {tuple(output.shape)}")
    
    print(f"\n✅ Shape preserved! Input {tuple(x.shape)} → Output {tuple(output.shape)}")
    print("   But information is enriched by multiple attention heads!")


def compare_approaches():
    """
    Compare wrapper vs efficient approach with timing.
    """
    print("\n\n⏱️  Performance Comparison")
    print("=" * 60)
    
    # Setup
    batch_size = 8
    seq_len = 128
    d_in = 512
    d_out_per_head = 64
    num_heads = 8
    
    x = torch.randn(batch_size, seq_len, d_in)
    
    # Wrapper approach
    print("Testing Wrapper Approach...")
    torch.manual_seed(123)
    wrapper = MultiHeadAttentionWrapper(d_in, d_out_per_head, seq_len, 0.0, num_heads)
    
    import time
    start = time.time()
    for _ in range(10):
        _ = wrapper(x)
    wrapper_time = time.time() - start
    
    # Efficient approach
    print("Testing Efficient Approach...")
    torch.manual_seed(123)
    efficient = MultiHeadAttention(d_in, d_out_per_head * num_heads, seq_len, 0.0, num_heads)
    
    start = time.time()
    for _ in range(10):
        _ = efficient(x)
    efficient_time = time.time() - start
    
    print(f"\n📊 Results (10 forward passes):")
    print(f"  Wrapper Approach:   {wrapper_time:.4f}s")
    print(f"  Efficient Approach: {efficient_time:.4f}s")
    print(f"  Speedup: {wrapper_time / efficient_time:.2f}x")
    
    print(f"\n✅ The efficient approach is significantly faster!")
    print("   This is why all production Transformers use it.")


def demo():
    """
    Run the complete demonstration of multi-head attention.
    """
    print("🚀 Building LLMs From Scratch (Part 9):")
    print("   Multi-Head Attention\n")
    
    # Demo 1: Wrapper approach
    demo_wrapper_approach()
    
    print(f"\n" + "="*60)
    
    # Demo 2: Efficient approach
    demo_efficient_approach()
    
    print(f"\n" + "="*60)
    
    # Demo 3: Shape tracing
    demo_shape_tracing()
    
    print(f"\n" + "="*60)
    
    # Demo 4: Performance comparison
    compare_approaches()
    
    print(f"\n" + "="*60)
    print("🎯 Key Takeaways:")
    print("- Multi-head attention runs multiple attention mechanisms in parallel")
    print("- Each head learns different aspects of relationships")
    print("- The 'wrapper' approach is intuitive but slow")
    print("- The 'weight split' approach is efficient and production-ready")
    print("- head_dim = d_out / num_heads")
    print("- All modern Transformers (GPT, BERT, etc.) use the efficient approach")
    
    print(f"\n" + "="*60)
    print("🎉 Demo complete!")
    print("\nCheck out the full tutorial:")
    print("📝 Medium: https://medium.com/@soloshun/building-llms-from-scratch-part-9-multi-head-attention")
    print("📂 GitHub: https://github.com/soloeinsteinmit/llm-from-scratch")
    print("📓 Interactive notebook: notebooks/part09_multi_head_attention.ipynb")
    
    print("\n🔜 Next up: Part 10 - Bird's Eye View of LLM Architecture")


if __name__ == "__main__":
    demo()

