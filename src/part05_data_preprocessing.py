"""
Building LLMs From Scratch (Part 5): The Complete Data Preprocessing Pipeline
=============================================================================

This module demonstrates the complete end-to-end data preprocessing pipeline
that transforms raw text into model-ready tensors for LLM training.

Author: Solomon Eshun
Article: https://soloshun.medium.com/building-llms-from-scratch-part-5-the-complete-data-preprocessing-pipeline-5247a8ee232a
Repository: https://github.com/soloeinsteinmit/llm-from-scratch
"""

import torch
import torch.nn as nn
import tiktoken
import sys

# Add the parent directory to the system path to allow imports
sys.path.insert(0, '../')
from src.part03_dataloader import create_dataloader_v1
from src.part04_embeddings import GPTEmbedding


def complete_preprocessing_pipeline(
    raw_text, 
    batch_size=8, 
    context_size=4, 
    emb_dim=256, 
    vocab_size=50257,
    verbose=True
):
    """
    Complete data preprocessing pipeline from raw text to model-ready tensors.
    
    This function demonstrates the entire pipeline we've built across Parts 2-4:
    1. Tokenization with BPE (handled internally by DataLoader)
    2. Input-target pair creation with sliding window
    3. Token and positional embeddings
    4. Batching for efficient training
    
    Args:
        raw_text (str): Raw input text to process
        batch_size (int): Number of examples per batch
        context_size (int): Length of each input sequence
        emb_dim (int): Embedding dimension
        vocab_size (int): Vocabulary size (GPT-2 default: 50257)
        verbose (bool): Whether to print progress information
    
    Returns:
        tuple: (model_ready_inputs, targets)
            - model_ready_inputs: Tensor [batch_size, context_size, emb_dim]
            - targets: Tensor [batch_size, context_size]
    """
    if verbose:
        print("🚀 Running Complete Preprocessing Pipeline")
        print("=" * 60)
    
    # Step 1: Create DataLoader (handles tokenization + input-target pairs)
    dataloader = create_dataloader_v1(
        raw_text,
        batch_size=batch_size,
        max_length=context_size,
        stride=context_size,
        shuffle=False,
        drop_last=True
    )
    
    if verbose:
        print(f"✅ Step 1: DataLoader created")
        print(f"   - Total batches: {len(dataloader)}")
        print(f"   - Total examples: {len(dataloader.dataset)}")
    
    # Step 2: Initialize embedding layer
    torch.manual_seed(123)  # For reproducible results
    embedding_layer = GPTEmbedding(vocab_size, emb_dim, context_size)
    
    if verbose:
        total_params = sum(p.numel() for p in embedding_layer.parameters())
        print(f"✅ Step 2: Embedding layer initialized")
        print(f"   - Total parameters: {total_params:,}")
        print(f"   - Token embeddings: {vocab_size * emb_dim:,}")
        print(f"   - Positional embeddings: {context_size * emb_dim:,}")
    
    # Step 3: Get one batch and process it
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    
    if verbose:
        print(f"✅ Step 3: Batch loaded")
        print(f"   - Input shape: {inputs.shape}")
        print(f"   - Target shape: {targets.shape}")
    
    # Step 4: Convert to embeddings
    model_ready_inputs = embedding_layer(inputs)
    
    if verbose:
        print(f"✅ Step 4: Embeddings created")
        print(f"   - Output shape: {model_ready_inputs.shape}")
        print("=" * 60)
        print("🎉 Pipeline Complete!")
    
    return model_ready_inputs, targets


def demo_tokenization():
    """
    Demonstrate the tokenization step with detailed analysis.
    """
    print("🔢 Tokenization Demo with BPE")
    print("=" * 40)
    
    # Sample text
    sample_text = "I HAD always thought Jack Gisburn rather a cheap genius"
    
    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Tokenize
    tokens = tokenizer.encode(sample_text)
    
    print(f"Original text: '{sample_text}'")
    print(f"Token IDs: {tokens}")
    print(f"Number of tokens: {len(tokens)}")
    print(f"Vocabulary size: {tokenizer.n_vocab:,}")
    
    # Show individual tokens
    print(f"\nToken breakdown:")
    for i, token_id in enumerate(tokens):
        token_text = tokenizer.decode([token_id])
        print(f"  {i:2d}: ID={token_id:5d} → '{token_text}'")
    
    # Verify round-trip
    decoded = tokenizer.decode(tokens)
    print(f"\nDecoded text: '{decoded}'")
    print(f"Round-trip successful: {sample_text == decoded}")
    print()


def demo_data_loading():
    """
    Demonstrate the data loading step with input-target pairs.
    """
    print("📊 Data Loading Demo")
    print("=" * 40)
    
    # Sample text for demo
    sample_text = "The quick brown fox jumps over the lazy dog. " * 10
    
    # Create dataloader
    dataloader = create_dataloader_v1(
        sample_text,
        batch_size=4,
        max_length=6,
        stride=3,
        shuffle=False
    )
    
    print(f"DataLoader configuration:")
    print(f"  - Batch size: 4")
    print(f"  - Context size: 6") 
    print(f"  - Stride: 3")
    print(f"  - Total batches: {len(dataloader)}")
    
    # Get one batch
    inputs, targets = next(iter(dataloader))
    
    print(f"\nSample batch:")
    print(f"  - Inputs shape: {inputs.shape}")
    print(f"  - Targets shape: {targets.shape}")
    
    # Show first example
    print(f"\nFirst example:")
    print(f"  - Input:  {inputs[0].tolist()}")
    print(f"  - Target: {targets[0].tolist()}")
    print(f"  - Notice: Target = Input shifted by 1")
    print()


def demo_embeddings():
    """
    Demonstrate the embedding step.
    """
    print("🧠 Embeddings Demo")
    print("=" * 40)
    
    # Create sample token IDs
    sample_tokens = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    
    # Initialize embedding layer
    embedding_layer = GPTEmbedding(vocab_size=50257, emb_dim=128, context_size=4)
    
    # Get embeddings
    embeddings = embedding_layer(sample_tokens)
    
    print(f"Input tokens shape: {sample_tokens.shape}")
    print(f"Output embeddings shape: {embeddings.shape}")
    print(f"Each token ID → 128-dimensional vector")
    
    # Show parameter counts
    total_params = sum(p.numel() for p in embedding_layer.parameters())
    print(f"\nEmbedding layer parameters:")
    print(f"  - Token embeddings: {50257 * 128:,}")
    print(f"  - Positional embeddings: {4 * 128:,}")
    print(f"  - Total: {total_params:,}")
    print()


def demo():
    """
    Run the complete demonstration of the preprocessing pipeline.
    """
    print("🚀 Building LLMs From Scratch (Part 5): Complete Data Preprocessing Pipeline\n")
    
    # Run individual demos
    demo_tokenization()
    demo_data_loading()
    demo_embeddings()
    
    # Load sample text
    try:
        with open("../data/the-verdict.txt", 'r', encoding="utf-8") as f:
            raw_text = f.read()
        print(f"📖 Loaded text data: {len(raw_text):,} characters")
    except FileNotFoundError:
        # Fallback text for demo
        raw_text = "I HAD always thought Jack Gisburn rather a cheap genius. " * 100
        print(f"📖 Using sample text: {len(raw_text):,} characters")
    
    print()
    
    # Run complete pipeline
    model_inputs, targets = complete_preprocessing_pipeline(
        raw_text,
        batch_size=8,
        context_size=4,
        emb_dim=256
    )
    
    print(f"\n📊 Final Pipeline Results:")
    print(f"Model-ready inputs: {model_inputs.shape}")
    print(f"Training targets: {targets.shape}")
    print(f"Data type: {model_inputs.dtype}")
    print(f"Device: {model_inputs.device}")
    
    print(f"\n✨ Success! Raw text → Model-ready tensors")
    print(f"Next: Part 6 - Self-Attention Mechanism")
    
    print("\n" + "=" * 60)
    print("🎉 Demo complete!")
    print("\nCheck out the full tutorial:")
    print("📝 Medium: https://soloshun.medium.com/building-llms-from-scratch-part-5-the-complete-data-preprocessing-pipeline-5247a8ee232a")
    print("📂 GitHub: https://github.com/soloeinsteinmit/llm-from-scratch")
    print("📓 Interactive notebook: notebooks/part05_data_preprocessing.ipynb")


if __name__ == "__main__":
    demo()
