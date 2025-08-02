# Medium Experiments

![Medium Experiments Banner](https://source.unsplash.com/800x200/?artificial-intelligence,research)  
*A collection of AI/ML experiments and visualizations accompanying Medium articles on deep learning and optimization.*

This repository serves as a central hub for experiments and code supporting my Medium articles on AI, machine learning, and optimization. As a Senior AI & Machine Learning Engineer, I share practical implementations and insights through projects like neural network testing for MNIST and optimizer visualizations for large language models (e.g., Kimi K2’s MuonClip). Each subdirectory corresponds to a specific article or experiment, designed for researchers, engineers, and enthusiasts.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/) 
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/) 
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📚 Table of Contents

- [Overview](#overview)
- [Projects](#projects)
- [Getting Started](#getting-started)
- [Dependencies](#dependencies)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

---

## 📖 Overview

This repository hosts code and experiments for my Medium articles, focusing on AI/ML topics like neural network testing, optimizer design, and computer vision. It includes:
- **Neural Network Testing Suite for MNIST**: A unit testing framework for validating neural network models on the MNIST dataset.
- **Kimi K2 MuonClip Story**: A Jupyter notebook visualizing the evolution from AdamW to MuonClip for training large language models.
- **Future Experiments**: Additional projects supporting upcoming Medium articles on AI/ML advancements.

These projects reflect my expertise in deep learning, neuromorphic computing, and model optimization, as showcased in my CV and publications.

---

## 📂 Projects

| Project | Description | Article Link | Directory |
|---------|-------------|--------------|-----------|
| **Neural Network Testing Suite** | Unit tests for MNIST neural networks, including intentional failure scenarios to ensure model robustness. | [Unit Test for Neural Network](https://medium.com/gitconnected/unit-test-for-neural-network-types-and-examples-022504afcaf2) | [unit_test](./unit_test/) |
| **Kimi K2 MuonClip Story** | Interactive visualizations of optimizer evolution from AdamW to MuonClip, inspired by the Kimi K2 paper. | [TBD](https://medium.com/@osama1339669) | [kimi_k2_muonclip](./kimi_k2_muonclip/) |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or higher
- Git
- Jupyter Notebook (for `kimi_k2_muonclip`)

### Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Novalis133/Medium.git
   cd Medium
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   .\venv\Scripts\activate   # Windows
   ```

3. **Install Dependencies**:
   Each project has its own `requirements.txt`. Install for a specific project:
   ```bash
   cd unit_test
   pip install -r requirements.txt
   cd ../kimi_k2_muonclip
   pip install -r requirements.txt
   ```

4. **Run Projects**:
   - **Neural Network Testing Suite**:
     ```bash
     cd unit_test
     python -m unittest
     ```
   - **Kimi K2 MuonClip Story**:
     ```bash
     cd kimi_k2_muonclip
     jupyter notebook Kimi_K2_MounClip_Story.ipynb
     ```

**Troubleshooting**:
- **Module Not Found**: Ensure `requirements.txt` is installed for each project.
- **Jupyter Issues**: Run `pip install jupyter` and verify Jupyter is accessible.
- **Path Errors**: Run commands from the correct subdirectory.

---

## 📋 Dependencies

| Dependency | Version | Installation | Used In |
|------------|---------|--------------|---------|
| Python | 3.10+ | `brew install python@3.10` (macOS) or [python.org](https://www.python.org/) | All |
| PyTorch | 2.1+ | `pip install torch>=2.1.0` | unit_test |
| NumPy | 1.23+ | `pip install numpy>=1.23.0` | kimi_k2_muonclip |
| Matplotlib | 3.5+ | `pip install matplotlib>=3.5.0` | kimi_k2_muonclip |
| IPython | Latest | `pip install ipython` | kimi_k2_muonclip |
| Pillow | Latest | `pip install Pillow` | kimi_k2_muonclip |

Install all dependencies for both projects:
```bash
pip install torch>=2.1.0 numpy>=1.23.0 matplotlib>=3.5.0 ipython Pillow
```

---

## 📂 Project Structure

```
Medium/
├── unit_test/                     # Neural Network Testing Suite
│   ├── tests/                    # Unit test scripts
│   ├── requirements.txt          # Dependencies
│   ├── LICENSE                   # MIT License
│   └── README.md                 # Project documentation
├── kimi_k2_muonclip/             # Kimi K2 MuonClip Story
│   ├── Kimi_K2_MounClip_Story.ipynb  # Jupyter notebook
│   ├── qk_clip_lifecycle.gif     # Output: Lifecycle animation
│   ├── qk_clip_tradeoff.gif      # Output: Trade-off animation
│   ├── requirements.txt          # Dependencies
│   ├── LICENSE                   # MIT License
│   └── README.md                 # Project documentation
├── images/                       # Shared images for documentation
├── LICENSE                       # MIT License for the repository
└── README.md                     # Main documentation
```

---

## 🤝 Contributing

Contributions to add new experiments or improve existing ones are welcome! Follow these steps:
1. **Fork the Repository**:
   ```bash
   git fork https://github.com/Novalis133/Medium.git
   ```
2. **Create a Feature Branch**:
   ```bash
   git checkout -b feature/add-new-experiment
   ```
3. **Commit Changes**:
   Use [Conventional Commits](https://www.conventionalcommits.org/):
   ```bash
   git commit -m "feat: add new Medium experiment for article X"
   ```
4. **Run Tests and Linting**:
   For `unit_test`:
   ```bash
   cd unit_test
   python -m unittest
   black .
   flake8 .
   ```
   For `kimi_k2_muonclip`:
   ```bash
   cd kimi_k2_muonclip
   python -m doctest Kimi_K2_MounClip_Story.ipynb -v
   black .
   flake8 .
   ```
5. **Submit a Pull Request**:
   ```bash
   git push origin feature/add-new-experiment
   ```
   Open a PR with a detailed description.

**Guidelines**:
- Follow the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/).
- Update `README.md` and `requirements.txt` in the relevant subdirectory.
- Ensure code runs with specified dependencies.

---

## 📫 Contact

- **Email**: osama1339669@gmail.com
- **LinkedIn**: [Osama](https://www.linkedin.com/in/osamat339669/)
- **GitHub Issues**: [Issues Page](https://github.com/Novalis133/Medium/issues)
- **Medium Blog**: [Osama’s Medium](https://medium.com/@osama1339669)

---

## 🙏 Acknowledgments

- [PyTorch](https://pytorch.org/) for deep learning support.
- [Matplotlib](https://matplotlib.org/) and [NumPy](https://numpy.org/) for visualization.
- [Jupyter](https://jupyter.org/) for interactive notebooks.
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/) for testing.
- [Kimi K2 Paper](https://arxiv.org/abs/XXXX.XXXXX) for optimizer insights.