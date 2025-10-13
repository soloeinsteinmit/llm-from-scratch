"""
Building LLMs From Scratch (Part 6): The Attention Mechanism
============================================================

This module demonstrates the simplified self-attention mechanism - the core concept
behind transformer models. We implement attention without trainable weights to
understand the fundamental mechanics.

Author: Solomon Eshun
Article: https://soloshun.medium.com/building-llms-from-scratch-part-6-the-attention-mechanism-b7ffc18c0dae
Repository: https://github.com/soloeinsteinmit/llm-from-scratch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class SimplifiedSelfAttention(nn.Module):
    """
    A simplified self-attention mechanism without trainable weights.
    
    This implementation demonstrates the core concepts of attention:
    1. Compute attention scores (dot products)
    2. Normalize with softmax
    3. Compute weighted sum (context vectors)
    """
    
    def __init__(self, d_in, d_out=None):
        """
        Initialize the simplified self-attention layer.
        
        Args:
            d_in (int): Input embedding dimension
            d_out (int, optional): Output dimension. Defaults to d_in.
        """
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out if d_out is not None else d_in
        
    def forward(self, x):
        """
        Forward pass of simplified self-attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_in)
            
        Returns:
            tuple: (context_vectors, attention_weights)
                - context_vectors: Output tensor of shape (batch_size, seq_len, d_out)
                - attention_weights: Attention weights of shape (batch_size, seq_len, seq_len)
        """
        batch_size, seq_len, d_in = x.shape
        
        # Step 1: Compute attention scores (dot products)
        # x @ x.transpose(-2, -1) computes all pairwise dot products
        attn_scores = x @ x.transpose(-2, -1)  # (batch_size, seq_len, seq_len)
        
        # Step 2: Apply softmax normalization
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # Step 3: Compute context vectors (weighted sum)
        context_vectors = attn_weights @ x  # (batch_size, seq_len, d_in)
        
        return context_vectors, attn_weights


def compute_attention_step_by_step(inputs, query_idx=1, verbose=True):
    """
    Compute attention step-by-step for educational purposes.
    
    Args:
        inputs (torch.Tensor): Input embeddings of shape (seq_len, d_in)
        query_idx (int): Index of the query token to focus on
        verbose (bool): Whether to print detailed steps
        
    Returns:
        tuple: (context_vector, attention_weights, attention_scores)
    """
    seq_len, d_in = inputs.shape
    query = inputs[query_idx]
    
    if verbose:
        print(f"🎯 Computing attention for token at index {query_idx}")
        print("=" * 50)
    
    # Step 1: Compute attention scores
    attn_scores = torch.empty(seq_len)
    
    if verbose:
        print("Step 1: Computing attention scores (dot products)")
    
    for i, x_i in enumerate(inputs):
        score = torch.dot(x_i, query)
        attn_scores[i] = score
        
        if verbose:
            print(f"  Token {i}: dot_product = {score:.4f}")
    
    # Step 2: Normalize with softmax
    attn_weights = torch.softmax(attn_scores, dim=0)
    
    if verbose:
        print(f"\nStep 2: Normalizing with softmax")
        for i, (score, weight) in enumerate(zip(attn_scores, attn_weights)):
            print(f"  Token {i}: {score:.4f} → {weight:.4f} ({weight*100:.1f}%)")
    
    # Step 3: Compute context vector
    context_vector = torch.zeros(d_in)
    
    if verbose:
        print(f"\nStep 3: Computing context vector (weighted sum)")
    
    for i, (x_i, weight) in enumerate(zip(inputs, attn_weights)):
        weighted_vector = weight * x_i
        context_vector += weighted_vector
        
        if verbose:
            print(f"  Token {i}: {weight:.4f} × {x_i} = {weighted_vector}")
    
    if verbose:
        print(f"\n✨ Final context vector: {context_vector}")
    
    return context_vector, attn_weights, attn_scores


def visualize_attention_weights(attention_weights, tokens, title="Self-Attention Weights"):
    """
    Create a heatmap visualization of attention weights.
    
    Args:
        attention_weights (torch.Tensor): Attention weights matrix
        tokens (list): List of token strings
        title (str): Title for the plot
    """
    plt.figure(figsize=(10, 8))
    
    # Create the heatmap
    sns.heatmap(
        attention_weights.numpy(), 
        annot=True, 
        fmt='.3f',
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='Blues',
        cbar_kws={'label': 'Attention Weight'}
    )
    
    plt.title(f'{title}\n(Rows: Query tokens, Columns: Key tokens)', 
              fontsize=14, pad=20)
    plt.xlabel('Key Tokens (what we attend TO)', fontsize=12)
    plt.ylabel('Query Tokens (what is attending)', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def analyze_attention_patterns(attention_weights, tokens):
    """
    Analyze and print attention patterns.
    
    Args:
        attention_weights (torch.Tensor): Attention weights matrix
        tokens (list): List of token strings
    """
    print("🔍 Attention Pattern Analysis:")
    print("=" * 40)
    
    for i, query_token in enumerate(tokens):
        weights = attention_weights[i]
        
        # Get top 3 attended tokens
        top_indices = torch.argsort(weights, descending=True)[:3]
        
        print(f"\n'{query_token}' pays most attention to:")
        for j, idx in enumerate(top_indices):
            print(f"  {j+1}. '{tokens[idx]}': {weights[idx]:.3f} ({weights[idx]*100:.1f}%)")


def compare_vectors(original_vectors, context_vectors, tokens):
    """
    Compare original embeddings with context vectors.
    
    Args:
        original_vectors (torch.Tensor): Original input embeddings
        context_vectors (torch.Tensor): Context vectors from attention
        tokens (list): List of token strings
    """
    print("📊 Vector Comparison Analysis:")
    print("=" * 50)
    
    for i, token in enumerate(tokens):
        original = original_vectors[i]
        context = context_vectors[i]
        
        # Calculate cosine similarity
        cos_sim = torch.cosine_similarity(original.unsqueeze(0), context.unsqueeze(0))
        
        # Calculate L2 distance
        l2_dist = torch.norm(original - context)
        
        print(f"{token:>8}: cos_sim={cos_sim.item():.3f}, L2_dist={l2_dist.item():.3f}")
    
    print(f"\n💡 Interpretation:")
    print(f"- Cosine similarity close to 1.0 means the direction is preserved")
    print(f"- L2 distance shows how much the magnitude changed")
    print(f"- Context vectors blend information from multiple tokens")


def demo_basic_attention():
    """
    Demonstrate basic attention mechanism with a simple example.
    """
    print("🚀 Basic Attention Mechanism Demo")
    print("=" * 60)
    
    # Create simple input embeddings
    sentence = "Your journey starts with one step"
    tokens = sentence.split()
    
    # Use specific embeddings for reproducibility
    inputs = torch.tensor([
        [0.43, 0.15, 0.89], # Your
        [0.55, 0.87, 0.66], # journey
        [0.57, 0.85, 0.64], # starts
        [0.22, 0.58, 0.33], # with
        [0.77, 0.25, 0.10], # one
        [0.05, 0.80, 0.55], # step
    ])
    
    print(f"📝 Input sentence: '{sentence}'")
    print(f"🔢 Tokens: {tokens}")
    print(f"🧠 Input embeddings shape: {inputs.shape}")
    print(f"Input embeddings:\n{inputs}")
    
    # Step-by-step computation for one token
    print(f"\n" + "="*60)
    context_vec, attn_weights, attn_scores = compute_attention_step_by_step(
        inputs, query_idx=1, verbose=True
    )
    
    # Matrix computation for all tokens
    print(f"\n" + "="*60)
    print("🚀 Computing attention for all tokens simultaneously:")
    
    # Compute attention scores matrix
    attn_scores_matrix = inputs @ inputs.T
    attn_weights_matrix = torch.softmax(attn_scores_matrix, dim=-1)
    all_context_vectors = attn_weights_matrix @ inputs
    
    print(f"📊 Attention scores matrix shape: {attn_scores_matrix.shape}")
    print(f"🎯 Attention weights matrix shape: {attn_weights_matrix.shape}")
    print(f"✨ All context vectors shape: {all_context_vectors.shape}")
    
    # Verify consistency
    print(f"\n✅ Verification:")
    print(f"Step-by-step matches matrix: {torch.allclose(context_vec, all_context_vectors[1])}")
    
    return inputs, all_context_vectors, attn_weights_matrix, tokens


def demo_attention_class():
    """
    Demonstrate the SimplifiedSelfAttention class.
    """
    print("🧪 SimplifiedSelfAttention Class Demo")
    print("=" * 60)
    
    # Get demo data
    inputs, expected_context, expected_weights, tokens = demo_basic_attention()
    
    # Add batch dimension
    batch_inputs = inputs.unsqueeze(0)  # Shape: (1, 6, 3)
    
    # Create attention layer
    attention_layer = SimplifiedSelfAttention(d_in=3)
    
    # Forward pass
    context_vectors, attention_weights = attention_layer(batch_inputs)
    
    print(f"\n🔧 Attention Layer:")
    print(f"Input shape: {batch_inputs.shape}")
    print(f"Output context vectors shape: {context_vectors.shape}")
    print(f"Output attention weights shape: {attention_weights.shape}")
    
    # Verify results
    print(f"\n✅ Class Verification:")
    print(f"Context vectors match: {torch.allclose(expected_context, context_vectors[0])}")
    print(f"Attention weights match: {torch.allclose(expected_weights, attention_weights[0])}")
    
    return attention_layer, context_vectors[0], attention_weights[0]


def demo():
    """
    Run the complete demonstration of simplified self-attention.
    """
    print("🚀 Building LLMs From Scratch (Part 6): The Attention Mechanism\n")
    
    # Basic attention demo
    inputs, context_vectors, attention_weights, tokens = demo_basic_attention()
    
    print(f"\n" + "="*60)
    
    # Analyze attention patterns
    analyze_attention_patterns(attention_weights, tokens)
    
    print(f"\n" + "="*60)
    
    # Compare original vs context vectors
    compare_vectors(inputs, context_vectors, tokens)
    
    print(f"\n" + "="*60)
    
    # Class demo
    attention_layer, class_context, class_weights = demo_attention_class()
    
    print(f"\n" + "="*60)
    print("🎯 Key Takeaways:")
    print("- Attention converts input vectors into context-aware vectors")
    print("- Three steps: compute scores, normalize, weighted sum")
    print("- Matrix operations enable efficient computation")
    print("- No trainable parameters in this simplified version")
    print("- Next: Add Query, Key, Value matrices for learnable attention")
    
    print(f"\n" + "="*60)
    print("🎉 Demo complete!")
    print("\nCheck out the full tutorial:")
    print("📝 Medium: https://soloshun.medium.com/building-llms-from-scratch-part-6-the-attention-mechanism-b7ffc18c0dae")
    print("📂 GitHub: https://github.com/soloeinsteinmit/llm-from-scratch")
    print("📓 Interactive notebook: notebooks/part06_attention_mechanism.ipynb")


if __name__ == "__main__":
    demo()
