"""
Building LLMs From Scratch (Part 7): Self-Attention with Trainable Weights
===========================================================================

This module demonstrates self-attention with trainable Query, Key, and Value matrices.
This is the foundation of transformer models like GPT, where the model learns optimal
attention patterns during training.

Author: Solomon Eshun
Article: https://soloshun.medium.com/building-llms-from-scratch-part-7-self-attention-with-weights
Repository: https://github.com/soloeinsteinmit/llm-from-scratch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class SelfAttention_v1(nn.Module):
    """
    Self-attention with trainable weights using nn.Parameter.
    
    This version uses raw nn.Parameter objects to give full control over
    the weight matrices. It's educational but less commonly used in practice.
    
    The three weight matrices (W_query, W_key, W_value) are what make this
    attention mechanism "learnable" - they're updated during training.
    """
    
    def __init__(self, d_in, d_out):
        """
        Initialize self-attention with trainable weights.
        
        Args:
            d_in (int): Input embedding dimension
            d_out (int): Output dimension (can be different from d_in)
        """
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        
        # Initialize trainable weight matrices
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))
        
    def forward(self, x):
        """
        Forward pass of self-attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_in)
            
        Returns:
            context_vectors: Output tensor of shape (batch_size, seq_len, d_out)
        """
        # Project inputs into queries, keys, and values
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        
        # Compute attention scores
        attn_scores = queries @ keys.T
        
        # Scale by square root of key dimension (scaled dot-product attention)
        attn_scores_scaled = attn_scores / keys.shape[-1]**0.5
        
        # Normalize with softmax
        attn_weights = torch.softmax(attn_scores_scaled, dim=-1)
        
        # Compute context vectors
        context_vec = attn_weights @ values
        return context_vec


class SelfAttention_v2(nn.Module):
    """
    Self-attention with trainable weights using nn.Linear.
    
    This is the preferred implementation in practice. nn.Linear handles:
    - Weight initialization
    - Optional bias terms
    - Optimized matrix operations
    - Cleaner, more maintainable code
    
    This is the foundation of attention in GPT-2 and other transformer models.
    """
    
    def __init__(self, d_in, d_out, qkv_bias=False):
        """
        Initialize self-attention with nn.Linear layers.
        
        Args:
            d_in (int): Input embedding dimension
            d_out (int): Output dimension
            qkv_bias (bool): Whether to include bias in Q, K, V projections
        """
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        
        # Use nn.Linear for cleaner, more efficient implementation
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        
    def forward(self, x):
        """
        Forward pass of self-attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_in)
            
        Returns:
            context_vectors: Output tensor of shape (batch_size, seq_len, d_out)
        """
        # Project inputs into queries, keys, and values
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        # Compute attention scores
        attn_scores = queries @ keys.T
        
        # Apply scaled dot-product attention
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        
        # Compute context vectors
        context_vec = attn_weights @ values
        return context_vec


def compute_qkv_step_by_step(inputs, W_query, W_key, W_value, query_idx=1, verbose=True):
    """
    Compute Query, Key, Value projections step-by-step for educational purposes.
    
    Args:
        inputs (torch.Tensor): Input embeddings of shape (seq_len, d_in)
        W_query (torch.Tensor): Query weight matrix
        W_key (torch.Tensor): Key weight matrix
        W_value (torch.Tensor): Value weight matrix
        query_idx (int): Index of the query token to focus on
        verbose (bool): Whether to print detailed steps
        
    Returns:
        tuple: (context_vector, attention_weights, queries, keys, values)
    """
    seq_len, d_in = inputs.shape
    
    if verbose:
        print(f"🎯 Computing Self-Attention with Trainable Weights")
        print("=" * 60)
        print(f"Input shape: {inputs.shape}")
        print(f"W_query shape: {W_query.shape}")
        print(f"W_key shape: {W_key.shape}")
        print(f"W_value shape: {W_value.shape}")
    
    # Step 1: Project inputs into Q, K, V
    queries = inputs @ W_query
    keys = inputs @ W_key
    values = inputs @ W_value
    
    if verbose:
        print(f"\n📊 Step 1: Project inputs into Q, K, V")
        print(f"Queries shape: {queries.shape}")
        print(f"Keys shape: {keys.shape}")
        print(f"Values shape: {values.shape}")
    
    # Focus on one query for detailed analysis
    query = queries[query_idx]
    
    if verbose:
        print(f"\n🔍 Focusing on query at index {query_idx}")
        print(f"Query vector: {query}")
    
    # Step 2: Compute attention scores
    attn_scores = query @ keys.T
    
    if verbose:
        print(f"\n📊 Step 2: Compute attention scores")
        print(f"Attention scores: {attn_scores}")
    
    # Step 3: Scale the scores
    d_k = keys.shape[-1]
    attn_scores_scaled = attn_scores / (d_k ** 0.5)
    
    if verbose:
        print(f"\n📊 Step 3: Scale by sqrt(d_k) = sqrt({d_k}) = {d_k**0.5:.4f}")
        print(f"Scaled scores: {attn_scores_scaled}")
    
    # Step 4: Apply softmax
    attn_weights = torch.softmax(attn_scores_scaled, dim=-1)
    
    if verbose:
        print(f"\n📊 Step 4: Apply softmax")
        print(f"Attention weights: {attn_weights}")
        print(f"Sum of weights: {attn_weights.sum():.6f}")
    
    # Step 5: Compute context vector
    context_vector = attn_weights @ values
    
    if verbose:
        print(f"\n📊 Step 5: Compute context vector")
        print(f"Context vector: {context_vector}")
    
    return context_vector, attn_weights, queries, keys, values


def visualize_qkv_projections(inputs, queries, keys, values, tokens):
    """
    Visualize how inputs are projected into Q, K, V spaces.
    
    Args:
        inputs (torch.Tensor): Original input embeddings
        queries (torch.Tensor): Query projections
        keys (torch.Tensor): Key projections
        values (torch.Tensor): Value projections
        tokens (list): List of token strings
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original inputs
    im1 = axes[0, 0].imshow(inputs.T.numpy(), cmap='RdYlBu', aspect='auto')
    axes[0, 0].set_title('Original Input Embeddings', fontsize=12, pad=10)
    axes[0, 0].set_xlabel('Tokens')
    axes[0, 0].set_ylabel('Dimensions')
    axes[0, 0].set_xticks(range(len(tokens)))
    axes[0, 0].set_xticklabels(tokens, rotation=45)
    plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)
    
    # Queries
    im2 = axes[0, 1].imshow(queries.T.numpy(), cmap='RdYlBu', aspect='auto')
    axes[0, 1].set_title('Query Projections (Q)', fontsize=12, pad=10)
    axes[0, 1].set_xlabel('Tokens')
    axes[0, 1].set_ylabel('Dimensions')
    axes[0, 1].set_xticks(range(len(tokens)))
    axes[0, 1].set_xticklabels(tokens, rotation=45)
    plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
    
    # Keys
    im3 = axes[1, 0].imshow(keys.T.numpy(), cmap='RdYlBu', aspect='auto')
    axes[1, 0].set_title('Key Projections (K)', fontsize=12, pad=10)
    axes[1, 0].set_xlabel('Tokens')
    axes[1, 0].set_ylabel('Dimensions')
    axes[1, 0].set_xticks(range(len(tokens)))
    axes[1, 0].set_xticklabels(tokens, rotation=45)
    plt.colorbar(im3, ax=axes[1, 0], shrink=0.8)
    
    # Values
    im4 = axes[1, 1].imshow(values.T.numpy(), cmap='RdYlBu', aspect='auto')
    axes[1, 1].set_title('Value Projections (V)', fontsize=12, pad=10)
    axes[1, 1].set_xlabel('Tokens')
    axes[1, 1].set_ylabel('Dimensions')
    axes[1, 1].set_xticks(range(len(tokens)))
    axes[1, 1].set_xticklabels(tokens, rotation=45)
    plt.colorbar(im4, ax=axes[1, 1], shrink=0.8)
    
    plt.tight_layout()
    plt.show()


def demonstrate_scaling_importance():
    """
    Demonstrate why we scale attention scores by sqrt(d_k).
    
    Shows two key reasons:
    1. Prevents softmax saturation (peaky distributions)
    2. Keeps variance stable as dimensions increase
    """
    print("🔬 Why Scale by sqrt(d_k)?")
    print("=" * 60)
    
    # Reason 1: Softmax saturation
    print("\n📊 Reason 1: Preventing Softmax Saturation")
    print("-" * 60)
    
    tensor = torch.tensor([0.1, -0.2, 0.3, -0.2, 0.5])
    
    # Without scaling
    softmax_normal = torch.softmax(tensor, dim=-1)
    print(f"Original scores: {tensor}")
    print(f"Softmax (no scaling): {softmax_normal}")
    print(f"Distribution: relatively balanced")
    
    # With large scaling (simulating large d_k without correction)
    scaled_tensor = tensor * 8
    softmax_scaled = torch.softmax(scaled_tensor, dim=-1)
    print(f"\nScaled scores (×8): {scaled_tensor}")
    print(f"Softmax (scaled): {softmax_scaled}")
    print(f"Distribution: extremely peaky! (largest value dominates)")
    
    print(f"\n💡 When scores are too large, softmax becomes too confident,")
    print(f"   making gradients vanish and learning unstable.")
    
    # Reason 2: Variance stability
    print("\n\n📊 Reason 2: Maintaining Stable Variance")
    print("-" * 60)
    
    def compute_variance(dim, num_trials=1000):
        """Compute variance of dot products before and after scaling."""
        dot_products = []
        scaled_dot_products = []
        
        for _ in range(num_trials):
            q = torch.randn(dim)
            k = torch.randn(dim)
            
            dot_product = torch.dot(q, k)
            dot_products.append(dot_product.item())
            
            scaled_dot_product = dot_product / (dim ** 0.5)
            scaled_dot_products.append(scaled_dot_product.item())
        
        var_before = np.var(dot_products)
        var_after = np.var(scaled_dot_products)
        
        return var_before, var_after
    
    # Test with different dimensions
    for dim in [5, 20, 100]:
        var_before, var_after = compute_variance(dim)
        print(f"\nDimension = {dim}:")
        print(f"  Variance before scaling: {var_before:.4f}")
        print(f"  Variance after scaling:  {var_after:.4f}")
    
    print(f"\n💡 Scaling keeps variance close to 1 regardless of dimension,")
    print(f"   ensuring stable training across different model sizes.")


def demo_basic_self_attention_with_weights():
    """
    Demonstrate self-attention with trainable weights step-by-step.
    """
    print("🚀 Self-Attention with Trainable Weights Demo")
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
    
    d_in = inputs.shape[1]  # 3
    d_out = 2  # Project to 2 dimensions
    
    print(f"📝 Input sentence: '{sentence}'")
    print(f"🔢 Tokens: {tokens}")
    print(f"🧠 Input shape: {inputs.shape} ({len(tokens)} tokens, {d_in}-dim embeddings)")
    print(f"🎯 Output dimension: {d_out}")
    
    # Initialize weight matrices
    torch.manual_seed(123)
    W_query = nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_key = nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_value = nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    
    print(f"\n🔧 Initialized weight matrices:")
    print(f"W_query:\n{W_query}")
    print(f"\nW_key:\n{W_key}")
    print(f"\nW_value:\n{W_value}")
    
    # Step-by-step computation
    print(f"\n" + "="*60)
    context_vec, attn_weights, queries, keys, values = compute_qkv_step_by_step(
        inputs, W_query, W_key, W_value, query_idx=1, verbose=True
    )
    
    return inputs, queries, keys, values, tokens


def demo_self_attention_classes():
    """
    Demonstrate both SelfAttention_v1 and SelfAttention_v2 classes.
    """
    print("\n🧪 Testing Self-Attention Classes")
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
    
    d_in = 3
    d_out = 2
    
    # Test SelfAttention_v1 (using nn.Parameter)
    print("\n📦 Testing SelfAttention_v1 (nn.Parameter):")
    print("-" * 60)
    torch.manual_seed(123)
    sa_v1 = SelfAttention_v1(d_in=d_in, d_out=d_out)
    output_v1 = sa_v1(inputs)
    
    print(f"Input shape: {inputs.shape}")
    print(f"Output shape: {output_v1.shape}")
    print(f"Output:\n{output_v1}")
    
    # Test SelfAttention_v2 (using nn.Linear)
    print("\n\n📦 Testing SelfAttention_v2 (nn.Linear):")
    print("-" * 60)
    torch.manual_seed(789)
    sa_v2 = SelfAttention_v2(d_in=d_in, d_out=d_out, qkv_bias=False)
    output_v2 = sa_v2(inputs)
    
    print(f"Input shape: {inputs.shape}")
    print(f"Output shape: {output_v2.shape}")
    print(f"Output:\n{output_v2}")
    
    print("\n\n💡 Key Differences:")
    print("-" * 60)
    print("SelfAttention_v1 (nn.Parameter):")
    print("  ✓ Full manual control over weights")
    print("  ✓ Educational and explicit")
    print("  ✗ More verbose code")
    print("  ✗ Manual initialization needed")
    
    print("\nSelfAttention_v2 (nn.Linear):")
    print("  ✓ Cleaner, more maintainable code")
    print("  ✓ Automatic weight initialization")
    print("  ✓ Optional bias terms")
    print("  ✓ Optimized backend operations")
    print("  ✓ Industry standard (used in GPT-2, BERT, etc.)")
    
    return sa_v1, sa_v2


def demo():
    """
    Run the complete demonstration of self-attention with trainable weights.
    """
    print("🚀 Building LLMs From Scratch (Part 7):")
    print("   Self-Attention with Trainable Weights\n")
    
    # Basic demo
    inputs, queries, keys, values, tokens = demo_basic_self_attention_with_weights()
    
    print(f"\n" + "="*60)
    
    # Demonstrate scaling importance
    demonstrate_scaling_importance()
    
    print(f"\n" + "="*60)
    
    # Test classes
    sa_v1, sa_v2 = demo_self_attention_classes()
    
    print(f"\n" + "="*60)
    print("🎯 Key Takeaways:")
    print("- Query, Key, Value matrices make attention learnable")
    print("- Q determines 'what to look for'")
    print("- K represents 'what content is available'")
    print("- V contains 'the actual information to retrieve'")
    print("- Scaling by sqrt(d_k) ensures training stability")
    print("- nn.Linear is preferred over nn.Parameter in practice")
    
    print(f"\n" + "="*60)
    print("🎉 Demo complete!")
    print("\nCheck out the full tutorial:")
    print("📝 Medium: https://soloshun.medium.com/building-llms-from-scratch-part-7-self-attention-with-weights")
    print("📂 GitHub: https://github.com/soloeinsteinmit/llm-from-scratch")
    print("📓 Interactive notebook: notebooks/part07_self_attention_with_weights.ipynb")
    
    print("\n🔜 Next up: Part 8 - Causal Attention (Masked Self-Attention)")


if __name__ == "__main__":
    demo()

