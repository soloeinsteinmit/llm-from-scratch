# 🚀 Building LLMs from Scratch

> **A comprehensive, step-by-step journey into building Large Language Models from the ground up**
>
> **Author:** Solomon Eshun

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red.svg)](https://pytorch.org/)

This repository contains the complete source code, explanations, and visualizations for the **"Building LLMs from Scratch"** series. Whether you're a beginner curious about how ChatGPT works or an experienced developer wanting to understand transformer architecture deeply, this series will guide you through every component step by step.

## 📚 About This Series

This educational series breaks down the complexity of Large Language Models into digestible, hands-on tutorials. Each part builds upon the previous one, gradually constructing a complete transformer-based language model from scratch using PyTorch.

**🎯 Learning Objectives:**

- Understand the fundamental architecture of transformer models
- Implement each component (tokenization, embeddings, attention, etc.) from scratch
- Gain practical experience with PyTorch and deep learning concepts
- Learn best practices for training and evaluating language models
- Explore modern techniques used in state-of-the-art LLMs

**👥 Target Audience:**

- Students and researchers in AI/ML
- Software engineers interested in NLP
- Anyone curious about how LLMs actually work
- Developers wanting to build custom language models

## 🛣️ Series Roadmap

| Part   | Topic                                  | Status         | Article                                                                                                                       | Code                                       |
| ------ | -------------------------------------- | -------------- | ----------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------ |
| **01** | The Complete Theoretical Foundation    | ✅ Complete    | [Medium](https://soloshun.medium.com/building-llms-from-scratch-part-1-the-complete-theoretical-foundation-e66b45b7f379)      | N/A                                        |
| **02** | Tokenization                           | ✅ Complete    | [Medium](https://medium.com/@soloshun/building-llms-from-scratch-part-2-tokenization-e0bf05d24094)                            | [Code](./src/part02_tokenization.py)       |
| **03** | Data Pipeline(Input-Target Pairs)      | ✅ Complete    | [Medium](https://soloshun.medium.com/building-llms-from-scratch-part-3-data-pipeline-4ef6eb7ad154)                            | [Code](./src/part03_dataloader.py)         |
| **04** | Token Embeddings & Positional Encoding | ✅ Complete    | [Medium](https://soloshun.medium.com/building-llms-from-scratch-part-4-embedding-layer-0803f6b8495b)                          | [Code](./src/part04_embeddings.py)         |
| **05** | Complete Data Preprocessing Pipeline   | ✅ Complete    | [Medium](https://soloshun.medium.com/building-llms-from-scratch-part-5-the-complete-data-preprocessing-pipeline-5247a8ee232a) | [Code](./src/part05_data_preprocessing.py) |
| **0-** | Self-Attention Mechanism               | 🔄 In progress | [Medium](.)                                                                                                                   | [Code](./src/)                             |
| **0-** | Casual Attention                       | ⏳ Planned     | [Medium](.)                                                                                                                   | [Code](./src/)                             |
| **0-** | Multi-Head Attention                   | ⏳ Planned     | [Medium](.)                                                                                                                   | [Code](./src/)                             |
| **0-** | Transformer Blocks & Architecture      | ⏳ Planned     | [Medium](.)                                                                                                                   | [Code](./src/)                             |
| **0-** | Training Loop & Optimization           | ⏳ Planned     | [Medium](.)                                                                                                                   | [Code](./src/)                             |
| **0-** | Model Evaluation & Fine-tuning         | ⏳ Planned     | [Medium](.)                                                                                                                   | [Code](./src/)                             |

_Legend: ✅ Complete | 🔄 In Progress | ⏳ Planned_

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Basic understanding of Python and neural networks
- Familiarity with PyTorch (helpful but not required)

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/soloeinsteinmit/llm-from-scratch.git
    cd llm-from-scratch
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the first code example (from Part 2):**
    ```bash
    python src/part02_tokenization.py
    ```

## 📁 Repository Structure

```
llm-from-scratch/
├── README.md                 # You are here!
├── requirements.txt          # Python dependencies
├── LICENSE                   # MIT License
│
├── notebooks/                # Jupyter notebooks for interactive learning
│   ├── part02_tokenization.ipynb
│   └── ...
│
├── animations/               # Manim visualizations and diagrams
│   └── part-02-WordTokenizationScene.mp4    # Generated animation files
│
│
└── src/                      # Source code for each part
    ├── part02_tokenization.py
    └── utils/                # Helper functions and utilities
```

## 🎓 How to Use This Repository

### For Learners

1.  **Start with Part 01** on Medium for the theoretical foundation.
2.  **Follow Part 02** and subsequent parts for hands-on coding.
3.  **Run the code** to see practical implementation.
4.  **Experiment** with the parameters and try modifications.
5.  **Check the notebooks** for interactive exploration.

### For Educators

- Use the code examples in your courses
- Reference the visualizations for explanations
- Adapt the materials for your curriculum
- Contribute improvements and additional examples

### For Researchers

- Use as a foundation for your own model implementations
- Reference the clean, well-documented code structure
- Build upon the base architecture for your experiments

## 🎨 Visualizations

This series includes custom **Manim animations** that visualize complex concepts:

- 🔄 **Attention mechanisms** - See how tokens "attend" to each other
- 📊 **Data flow** - Understand how information moves through the model
- 🧮 **Matrix operations** - Visualize the math behind transformers
- 📈 **Training dynamics** - Watch the model learn in real-time

_Animations are generated using [Manim](https://www.manim.community/) and available in the `animations/` directory._

## 🤝 Contributing

We welcome contributions from the community! This is an open-source educational project aimed at making LLM understanding accessible to everyone.

**Ways to contribute:**

- 🐛 Report bugs or suggest improvements
- 📝 Improve documentation and explanations
- 🎨 Create additional visualizations
- 🔧 Add new features or optimizations
- 🌍 Translate content to other languages

<!-- Please read our [Contributing Guidelines](./CONTRIBUTING.md) and [Code of Conduct](./CODE_OF_CONDUCT.md) before submitting contributions. -->

## 📖 Additional Resources

### Related Articles & Tutorials

- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) by Jay Alammar
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer Paper
- [GPT-3 Paper](https://arxiv.org/abs/2005.14165) - Language Models are Few-Shot Learners

<!-- ### Recommended Resources

- **Books:** "Deep Learning" by Goodfellow, Bengio, and Courville
- **Courses:** CS224N (Stanford NLP), Fast.ai Deep Learning
- **Papers:** Start with the transformer paper, then explore GPT, BERT, and modern architectures -->

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## 🙏 Acknowledgments

- **Community:** Thanks to all contributors and learners who make this project better
- **Inspiration:** Built upon the excellent work of researchers and educators in the field
- **Tools:** Created with PyTorch, Manim, and lots of coffee ☕

## 📱 Connect & Follow

- 📝 **Medium:** Follow the series on [Medium](https://soloshun.medium.com/)
- 💼 **LinkedIn:** Connect and discuss on [LinkedIn](https://www.linkedin.com/in/solomon-eshun-788568317/)
- 🐙 **GitHub:** Star this repo and follow for updates

---

**⭐ If you find this helpful, please give it a star! It helps others discover this resource.**

_Built with ❤️ for the open-source community_
