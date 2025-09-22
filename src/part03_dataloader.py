"""
Building LLMs From Scratch (Part 3): Crafting the Data Pipeline
================================================================

This module contains clean, production-ready implementations of the data pipeline
developed in Part 3 of the "Building LLMs from Scratch" series.

Author: Solomon Eshun
Article: https://soloshun.medium.com/building-llms-from-scratch-part-3-the-data-pipeline
Repository: https://github.com/soloeinsteinmit/llm-from-scratch
"""

import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken
from typing import Optional


class GPTDatasetV1(Dataset):
    """
    PyTorch Dataset for GPT-style language model training.
    
    Creates input-target pairs using a sliding window approach where:
    - Input: A sequence of tokens of length max_length
    - Target: The same sequence shifted by one position (next token prediction)
    """
    
    def __init__(self, txt: str, tokenizer, max_length: int, stride: int):
        """
        Initialize the dataset with text and create all input-target pairs.
        
        Args:
            txt: Raw text to process
            tokenizer: Tokenizer to convert text to tokens (e.g., tiktoken)
            max_length: Length of each input sequence (context size)
            stride: Step size for sliding window (controls overlap)
        """
        self.input_ids = []
        self.target_ids = []
        
        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        
        # Create input-target pairs using sliding window
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
            
    def __len__(self) -> int:
        """Return the total number of input-target pairs."""
        return len(self.input_ids)
       
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a single input-target pair."""
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt: str, 
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """
    Create a DataLoader for GPT-style language model training.
    
    Args:
        txt: Raw text to process
        batch_size: Number of examples per batch
        max_length: Length of each input sequence (context size)
        stride: Step size for sliding window
        shuffle: Whether to shuffle the data
        drop_last: Whether to drop the last incomplete batch
        num_workers: Number of worker processes for data loading
        
    Returns:
        PyTorch DataLoader ready for training
    """
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
        num_workers=num_workers
    )
    
    return dataloader


def demonstrate_sliding_window(tokens: list, context_size: int, stride: int, max_examples: int = 5):
    """
    Demonstrate how the sliding window creates input-target pairs.
    
    Args:
        tokens: List of token IDs
        context_size: Size of the context window
        stride: Step size for the sliding window
        max_examples: Maximum number of examples to show
    """
    print(f"📊 Sliding Window Demo: context_size={context_size}, stride={stride}")
    print("-" * 60)
    
    count = 0
    for i in range(0, len(tokens) - context_size, stride):
        if count >= max_examples:
            print("... (and more)")
            break
            
        input_chunk = tokens[i:i + context_size]
        target_chunk = tokens[i + 1:i + context_size + 1]
        
        print(f"Window {count+1}: Input={input_chunk}, Target={target_chunk}")
        count += 1
    
    total_chunks = len(range(0, len(tokens) - context_size, stride))
    print(f"📈 Total chunks created: {total_chunks}")
    print()


def demo_basic_concepts():
    """
    Demonstrate the basic concepts of tokenization and input-target pairs.
    """
    print("=== Basic Concepts Demo ===\n")
    
    # Load tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Example text
    text = "I HAD always thought Jack Gisburn rather a cheap genius"
    tokens = tokenizer.encode(text)
    
    print(f"Text: '{text}'")
    print(f"Tokens: {tokens}")
    print(f"Number of tokens: {len(tokens)}\n")
    
    # Show input-target pair creation
    context_size = 8
    input_tokens = tokens[:context_size]
    target_tokens = tokens[1:context_size+1]
    
    print("🎯 Input-Target Pair Example:")
    print(f"Input:  {input_tokens}")
    print(f"Target: {target_tokens}\n")
    
    # Show individual prediction tasks
    print("📝 Individual Prediction Tasks:")
    for i in range(len(input_tokens)):
        context = input_tokens[:i+1]
        target = target_tokens[i]
        context_text = tokenizer.decode(context)
        target_text = tokenizer.decode([target])
        print(f"'{context_text}' → '{target_text}'")
    print()


def demo_dataloader():
    """
    Demonstrate the complete data pipeline with real data.
    """
    print("=== DataLoader Demo ===\n")
    
    # Load sample text (you can replace with your own file)
    sample_text = """I HAD always thought Jack Gisburn rather a cheap genius--though a 
    good fellow enough--so it was no great surprise to me to hear that he had been caught. 
    The surprise was that he had been caught in such a simple way. I should have expected 
    him to be trapped by some scheme of elaborate complexity."""
    
    print(f"Sample text: '{sample_text[:100]}...'\n")
    
    # Create a small dataloader for inspection
    print("🔬 Creating a small dataloader for inspection...")
    small_dataloader = create_dataloader_v1(
        sample_text, 
        batch_size=4, 
        max_length=8, 
        stride=4, 
        shuffle=False
    )
    
    print(f"Dataset size: {len(small_dataloader.dataset)} chunks")
    print(f"Number of batches: {len(small_dataloader)}\n")
    
    # Get the first batch
    data_iter = iter(small_dataloader)
    inputs, targets = next(data_iter)
    
    print(f"🎯 First Batch:")
    print(f"Inputs shape:  {inputs.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Inputs:\n{inputs}")
    print(f"Targets:\n{targets}\n")
    
    # Decode examples
    tokenizer = tiktoken.get_encoding("gpt2")
    print("📖 Decoded Examples:")
    for i in range(min(2, inputs.shape[0])):
        input_text = tokenizer.decode(inputs[i].tolist())
        target_text = tokenizer.decode(targets[i].tolist())
        print(f"Example {i+1}:")
        print(f"  Input:  '{input_text}'")
        print(f"  Target: '{target_text}'")
    print()


def demo_stride_comparison():
    """
    Compare different stride values to show their effect.
    """
    print("=== Stride Comparison Demo ===\n")
    
    tokenizer = tiktoken.get_encoding("gpt2")
    sample_text = "I HAD always thought Jack Gisburn rather a cheap genius"
    tokens = tokenizer.encode(sample_text)
    
    # Compare different stride values
    demonstrate_sliding_window(tokens, context_size=4, stride=1)
    demonstrate_sliding_window(tokens, context_size=4, stride=4)


if __name__ == "__main__":
    """
    Run all demonstrations when script is executed directly.
    """
    print("🚀 Building LLMs From Scratch (Part 3): Crafting the Data Pipeline\n")
    print("=" * 70)
    
    demo_basic_concepts()
    demo_stride_comparison()
    demo_dataloader()
    
    print("=" * 70)
    print("🎉 Demo complete! Check out the full tutorial:")
    print("📝 Medium: https://soloshun.medium.com/building-llms-from-scratch-part-3-the-data-pipeline")
    print("📂 GitHub: https://github.com/soloeinsteinmit/llm-from-scratch")
    print("📓 Interactive notebook: notebooks/part03_dataloader.ipynb")
