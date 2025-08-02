# Neural Network Testing Suite for MNIST

*A comprehensive unit testing suite for neural network models, tailored for the MNIST dataset.*

This repository provides a robust suite of unit tests for validating neural network models on the MNIST dataset, ensuring integrity, performance, and reliability. Built with Python’s `unittest.TestCase`, it’s ideal for AI/ML engineers and researchers, like myself (a Senior AI & Machine Learning Engineer), working on deep learning model validation and optimization.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/) 
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/) 
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📚 Table of Contents

- [Features](#features)
- [Intentional Failures](#intentional-failures)
- [Getting Started](#getting-started)
- [Dependencies](#dependencies)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Contributing](#contributing)
- [Status](#status)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

---

## ✨ Features

| Feature | Description | Use Case |
|---------|-------------|----------|
| **Model Validation** | Tests neural network initialization, training, and inference. | Ensure model functionality |
| **Data Preprocessing** | Validates data loading, preprocessing, and augmentation for MNIST. | Consistent data pipeline |
| **Performance Testing** | Benchmarks model accuracy and inference speed. | Optimize model efficiency |
| **Model Persistence** | Checks model saving/loading integrity. | Reliable model deployment |
| **Failure Scenarios** | Includes intentional failures to test robustness. | Identify model weaknesses |

---

## 🚨 Intentional Failures

The test suite includes deliberate failure scenarios to demonstrate its ability to detect common neural network issues:

| Test | Description | Observed Failure | Severity | Suggestion |
|------|-------------|------------------|----------|------------|
| `test_error_handling` | Validates handling of invalid inputs. | Model fails to raise errors for invalid inputs. | High | Add robust input validation in the forward method. |
| `test_overfitting_on_small_data` | Ensures model can overfit a small dataset. | Model does not overfit as expected. | Medium | Increase model capacity or adjust learning rate. |
| `test_reproducibility` | Checks consistent results with fixed seed. | Inconsistent outputs across runs with same seed. | High | Enforce deterministic behavior (e.g., set random seeds globally). |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or higher
- PyTorch 2.1 or higher
- Git

### Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Novalis133/Medium/unit_test.git
   cd unit_test
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   .\venv\Scripts\activate   # Windows
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Tests**:
   ```bash
   python -m unittest
   ```

**Troubleshooting**:
- **Module Not Found**: Ensure `requirements.txt` includes `torch>=2.1.0` and `unittest`.
- **Test Failures**: Check Python version (`python --version`) and PyTorch installation (`pip show torch`).
- **Path Issues**: Run from the `unit_test` directory.

---

## 📋 Dependencies

| Dependency | Version | Installation |
|------------|---------|--------------|
| Python | 3.10+ | `brew install python@3.10` (macOS) or download from [python.org](https://www.python.org/) |
| PyTorch | 2.1+ | `pip install torch>=2.1.0` |
| unittest | Built-in | Included with Python |

Install dependencies:
```bash
pip install torch>=2.1.0
```

---

## 📂 Project Structure

```
unit_test/
├── tests/                  # Unit test scripts
│   ├── test_model.py      # Tests for model initialization and training
│   ├── test_data.py       # Tests for data preprocessing and augmentation
│   ├── test_performance.py # Tests for performance and benchmarks
│   └── test_persistence.py # Tests for model saving/loading
├── requirements.txt       # Python dependencies
├── LICENSE                # MIT License file
└── README.md              # Project documentation
```

---

## 🧪 Testing

Run the full test suite:
```bash
python -m unittest discover -s tests
```

**Test Coverage**:
- Model initialization, forward pass, and training loops
- Data loading and preprocessing for MNIST
- Performance benchmarks (accuracy, inference time)
- Model persistence (save/load functionality)
- Error handling and reproducibility

**Expected Output**:
```bash
test_error_handling (tests.test_model.ModelTest) ... FAIL
test_overfitting_on_small_data (tests.test_model.ModelTest) ... FAIL
test_reproducibility (tests.test_model.ModelTest) ... FAIL
...
Ran 10 tests in 0.123s
FAILED (failures=3)
```

**Troubleshooting**:
- **Tests Not Found**: Ensure you’re in the `unit_test` directory.
- **PyTorch Errors**: Verify GPU availability or use CPU (`torch.cuda.is_available()`).
- **Unexpected Failures**: Review failure logs for specific issues.

---

## 🤝 Contributing

Contributions are welcome! Follow these steps:
1. **Fork the Repository**:
   ```bash
   git fork https://github.com/Novalis133/Medium/unit_test.git
   ```

2. **Create a Feature Branch**:
   ```bash
   git checkout -b feature/add-new-test
   ```

3. **Commit Changes**:
   Use [Conventional Commits](https://www.conventionalcommits.org/):
   ```bash
   git commit -m "feat: add test for model regularization"
   ```

4. **Run Tests and Linting**:
   ```bash
   pytest tests/
   black .
   flake8 .
   ```

5. **Submit a Pull Request**:
   ```bash
   git push origin feature/add-new-test
   ```
   Open a PR with a detailed description.

**Guidelines**:
- Follow the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/).
- Ensure tests pass and code is formatted with Black.
- Update documentation for new tests or features.

---

## 📈 Status

- **Test Suite**: Stable, covers core model functionality.
- **Failure Scenarios**: Intentionally included to highlight robustness.
- **Future Work**: Adding tests for advanced architectures (e.g., CNNs, Transformers).

[![Build Status](https://img.shields.io/badge/Build-Passing-green)](https://github.com/Novalis133/Medium/unit_test/actions)

---

## 📫 Contact

- **Email**: osama1339669@gmail.com
- **LinkedIn**: [Osama](https://www.linkedin.com/in/osamat339669/)
- **GitHub Issues**: [Issues Page](https://github.com/Novalis133/Medium/issues)
- **Article**: [Unit Test for Neural Network: Types and Examples](https://medium.com/gitconnected/unit-test-for-neural-network-types-and-examples-022504afcaf2)

---

## 🙏 Acknowledgments

- [PyTorch](https://pytorch.org/) for the deep learning framework.
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/) for benchmarking.
- [Python unittest](https://docs.python.org/3/library/unittest.html) for robust testing.
