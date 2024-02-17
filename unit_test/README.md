# Neural Network Testing Suite for MNIST

This repository contains a comprehensive suite of unit tests designed for a neural network model, specifically tailored for the MNIST dataset. The suite extends from Python's `unittest.TestCase` and offers various tests to ensure the model's integrity and functionality.

## Features

- Extensive testing of neural network initialization, training, and inference.
- Validation of data preprocessing and augmentation.
- Checks for model saving/loading, performance benchmarks, and more.
- Intentional failure scenarios to demonstrate test robustness.

## Intentional Failures

The test suite includes specific failure scenarios to showcase its capability in detecting common issues in neural network models:

1. **Error Handling Failure:**
   - Test: `test_error_handling`
   - Description: Validates the model's ability to handle invalid inputs.
   - Observed Failure: The model did not raise an error with invalid input, indicating a need for improved input validation.
   - Suggestion: Implement and enhance input validation in the model's forward method.

2. **Overfitting Test Failure:**
   - Test: `test_overfitting_on_small_data`
   - Description: Ensures the model can overfit a small dataset, a sign of its learning capability.
   - Observed Failure: The model did not overfit the small dataset as expected.
   - Suggestion: Review and possibly enhance the model's capacity to ensure it can learn detailed features from a small dataset.

3. **Reproducibility Test Failure:**
   - Test: `test_reproducibility`
   - Description: Checks if the model produces consistent results under the same conditions.
   - Observed Failure: Model outputs were different across two runs with the same seed.
   - Suggestion: Ensure controlled randomness and deterministic behavior in the model architecture to achieve reproducibility.

These failures are deliberately included to demonstrate the effectiveness of the testing suite in identifying potential areas of improvement in neural network models.

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

- Project Link: [https://github.com/novalis133/Medium/unit_test.git](https://github.com/novalis133/Medium/unit_test.git)
- Article Link: [Unit Test for Neural Network: Types and Examples](https://medium.com/gitconnected/unit-test-for-neural-network-types-and-examples-022504afcaf2)

