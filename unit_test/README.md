# Neural Network Testing Suite for MNIST

This repository contains a comprehensive suite of unit tests designed for a neural network model, specifically tailored for the MNIST dataset. The suite extends from Python's `unittest.TestCase` and offers various tests to ensure the model's integrity and functionality.

## Features

- Extensive testing of neural network initialization, training, and inference.
- Validation of data preprocessing and augmentation.
- Checks for model saving/loading, performance benchmarks, and more.
- Intentional failure scenarios to demonstrate test robustness.

## Intentional Failures

The test suite includes three intentional failures to demonstrate how the tests can detect common issues in neural network models:

1. **Model Initialization Failure:** Tests the model's ability to initialize without errors. An intentional failure here would indicate an issue in the model's `__init__` method.

2. **Input-Output Dimension Mismatch:** Ensures the model's output dimensions are as expected for given input dimensions. A failure in this test could point to a mismatch in the model architecture.

3. **Training Step Error:** Verifies that the model can perform a training step correctly. An error here might suggest issues in the forward pass, loss computation, or backward pass.

## Getting Started

To run the tests, follow these steps:

1. Clone the repository:
git clone https://github.com/novalis133/Medium/unit_test.git

2. Navigate to the cloned directory:

3. Run the tests:
python -m unit_test


## Dependencies

- Python 3.10
- PyTorch 2.1


## Contributing

Contributions, issues, and feature requests are welcome. Feel free to check the [issues page](https://github.com/novalis133/Medium/issues) for open problems or discussions.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Project Link: [https://github.com/novalis133/Medium/unit_test.git](https://github.com/novalis133/Medium/unit_test.git)

