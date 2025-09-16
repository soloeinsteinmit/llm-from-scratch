"""
Building LLMs From Scratch (Part 2): The Power of Tokenization
================================================================

This module contains clean, production-ready implementations of the tokenizers
developed in Part 2 of the "Building LLMs from Scratch" series.

Author: Solomon Eshun
Article: https://soloshun.medium.com/building-llms-from-scratch-part-2-the-power-of-tokenization
Repository: https://github.com/soloeinsteinmit/llm-from-scratch
"""

import re
from typing import Dict, List


class SimpleTokenizerV1:
    """
    A simple word-based tokenizer that splits text using regex patterns.
    
    This tokenizer will crash if it encounters words not in its vocabulary.
    Use SimpleTokenizerV2 for production applications.
    """
    
    def __init__(self, vocab: Dict[str, int]):
        """
        Initialize the tokenizer with a vocabulary.
        
        Args:
            vocab: Dictionary mapping tokens to integer IDs
        """
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}
        
    def encode(self, text: str) -> List[int]:
        """
        Convert text to a list of token IDs.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of token IDs
            
        Raises:
            KeyError: If text contains tokens not in vocabulary
        """
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert token IDs back to text.
        
        Args:
            ids: List of token IDs
            
        Returns:
            Decoded text string
        """
        text = " ".join([self.int_to_str[i] for i in ids])
        # Clean up spacing around punctuation
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


class SimpleTokenizerV2(SimpleTokenizerV1):
    """
    Improved tokenizer that handles unknown words using special tokens.
    
    Unknown words are replaced with <|unk|> token.
    """
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to token IDs, handling unknown words.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of token IDs (unknown words become <|unk|>)
        """
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        
        # Replace unknown tokens with <|unk|>
        preprocessed = [
            item if item in self.str_to_int else "<|unk|>" 
            for item in preprocessed
        ]
        
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids


def create_vocabulary(text: str, add_special_tokens: bool = True) -> Dict[str, int]:
    """
    Create a vocabulary from text using regex-based tokenization.
    
    Args:
        text: Input text to build vocabulary from
        add_special_tokens: Whether to add <|unk|> and <|endoftext|> tokens
        
    Returns:
        Dictionary mapping tokens to integer IDs
    """
    # Tokenize the text
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    
    # Create sorted unique tokens
    all_tokens = sorted(list(set(preprocessed)))
    
    # Add special tokens if requested
    if add_special_tokens:
        all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    
    # Create vocabulary dictionary
    vocab = {token: idx for idx, token in enumerate(all_tokens)}
    return vocab


def demo_tokenization():
    """
    Demonstrate the different tokenization approaches.
    """
    print("=== Tokenization Demo ===\n")
    
    # Sample text
    text = "My hobby is playing cricket"
    
    # 1. Word-based tokenization
    word_tokens = text.split()
    print("1. Word-based tokenization:")
    print(f"   Input: '{text}'")
    print(f"   Tokens: {word_tokens}")
    print("   Problem: Huge vocabulary with all possible words!\n")
    
    # 2. Character-based tokenization
    char_tokens = list(text)
    print("2. Character-based tokenization:")
    print(f"   Input: '{text}'")
    print(f"   Tokens: {char_tokens}")
    print("   Problem: Very long sequences, meaning is lost!\n")
    
    # 3. Sub-word tokenization
    print("3. Sub-word tokenization (BPE):")
    print("   - Common words like 'is' stay as one token")
    print("   - Rare words like 'snowboarding' -> ['snow', 'board', 'ing']")
    print("   - Best of both worlds! ✨\n")


def demo_simple_tokenizer():
    """
    Demonstrate our simple tokenizer implementations.
    """
    print("=== Simple Tokenizer Demo ===\n")
    
    # Create sample text and vocabulary
    sample_text = """I HAD always thought Jack Gisburn rather a cheap genius--though a 
    good fellow enough--so it was no great surprise to me to hear that he had been caught."""
    
    # Build vocabulary
    vocab = create_vocabulary(sample_text, add_special_tokens=True)
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Sample tokens: {list(vocab.keys())[:10]}\n")
    
    # Test SimpleTokenizerV1
    print("--- SimpleTokenizerV1 (crashes on unknown words) ---")
    tokenizer_v1 = SimpleTokenizerV1(vocab)
    
    test_text = "Hello, world!"
    try:
        encoded = tokenizer_v1.encode(test_text)
        decoded = tokenizer_v1.decode(encoded)
        print(f"✅ Success: '{test_text}' -> {encoded} -> '{decoded}'")
    except KeyError as e:
        print(f"❌ KeyError: {e}")
    
    # Test with unknown word
    unknown_text = "Hello, do you like pizza?"
    try:
        encoded = tokenizer_v1.encode(unknown_text)
        print(f"✅ Encoded unknown word successfully")
    except KeyError as e:
        print(f"❌ Failed on unknown word: {e}")
    
    print()
    
    # Test SimpleTokenizerV2
    print("--- SimpleTokenizerV2 (handles unknown words) ---")
    tokenizer_v2 = SimpleTokenizerV2(vocab)
    
    encoded = tokenizer_v2.encode(unknown_text)
    decoded = tokenizer_v2.decode(encoded)
    print(f"✅ With unknown words: '{unknown_text}'")
    print(f"   Encoded: {encoded}")
    print(f"   Decoded: '{decoded}'")


def demo_tiktoken_comparison():
    """
    Demonstrate BPE tokenization using tiktoken.
    
    Note: Requires 'pip install tiktoken'
    """
    try:
        import tiktoken
        
        print("=== BPE Tokenization with tiktoken ===\n")
        
        # Load GPT-2 tokenizer
        gpt2_tokenizer = tiktoken.get_encoding("gpt2")
        print("GPT-2 BPE tokenizer loaded!\n")
        
        # Test with various texts
        test_texts = [
            "Hello, world!",
            "Hello, do you like tea? <|endoftext|> Akwirw ier",
            "tokenization",
            "snowboarding",
            "artificialintelligence"
        ]
        
        for text in test_texts:
            encoded = gpt2_tokenizer.encode(
                text, 
                allowed_special={"<|endoftext|>"}
            )
            decoded = gpt2_tokenizer.decode(encoded)
            
            # Show subword breakdown
            tokens = [gpt2_tokenizer.decode([id_]) for id_ in encoded]
            
            print(f"Text: '{text}'")
            print(f"Tokens: {tokens}")
            print(f"IDs: {encoded}")
            print(f"Decoded: '{decoded}'")
            print()
            
    except ImportError:
        print("❌ tiktoken not installed. Run 'pip install tiktoken' to try BPE demo")


if __name__ == "__main__":
    """
    Run all demonstrations when script is executed directly.
    """
    print("🚀 Building LLMs From Scratch (Part 2): The Power of Tokenization\n")
    print("=" * 70)
    
    demo_tokenization()
    demo_simple_tokenizer()
    demo_tiktoken_comparison()
    
    print("=" * 70)
    print("🎉 Demo complete! Check out the full tutorial:")
    print("📝 Medium: https://soloshun.medium.com/building-llms-from-scratch-part-2-the-power-of-tokenization")
    print("📂 GitHub: https://github.com/soloeinsteinmit/llm-from-scratch")
    print("📓 Interactive notebook: notebooks/part02_tokenization.ipynb")
