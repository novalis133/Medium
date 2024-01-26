import random
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import unittest
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from SimpleNet import SimpleMNISTNet


def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

class NeuralNetworkUnitTest(unittest.TestCase):
    """
    A suite of unit tests for validating a neural network model, specifically designed for the MNIST dataset. 
    This class extends from unittest.TestCase and provides a series of tests to ensure the integrity and 
    functionality of the neural network.

    Attributes:
        model (nn.Module): The neural network model to be tested. In this case, it's an instance of SimpleMNISTNet.
        optimizer (torch.optim.Optimizer): The optimizer used for training the model. Configured with the model parameters.
        criterion (torch.nn.Module): The loss function used for training. Suitable for the model's specific task.
        transform (torchvision.transforms.Compose): A series of transformations applied to the input data.
        dataset (torch.utils.data.Dataset): The dataset used for testing, which is preprocessed and augmented.
        input_dim (tuple): The expected dimension of the input data to the model.

    Methods:
        test_model_initialization(self): Tests whether the model initializes correctly without any errors.
        test_input_output_dimension(self): Validates the input-output dimension compatibility of the model.
        test_training_step(self): Ensures that a single training step proceeds without errors.
        test_loss_computation(self): Checks if the loss is computed correctly by the model.
        test_gradient_flow(self): Verifies that gradients are correctly calculated and propagated through the model.
        test_overfitting_on_small_data(self): Tests if the model can overfit a small dataset, which is a way to check its learning capability.
        test_data_preprocessing_and_augmentation(self): Ensures that data preprocessing and augmentation steps are performed correctly.
        test_model_loading_and_saving(self): Checks if the model can be saved and then loaded correctly.
        test_inference_mode(self): Verifies the model's behavior in inference mode, particularly for layers like dropout and batch normalization.
        test_dependency(self): Ensures that the model integrates well with required dependencies and external components.
        test_hyperparameter_sensitivity(self): Evaluates the model's sensitivity to changes in hyperparameters.
        test_reproducibility(self): Confirms that the model provides consistent outputs under controlled random seed settings.
        test_error_handling(self): Tests the model's ability to handle invalid inputs and other error conditions gracefully.
        test_performance_benchmarks(self): Assesses the model's performance against predefined benchmarks.
        test_integration(self): Verifies the end-to-end functionality and integration of the model within a larger system or workflow.

    The class utilizes PyTorch's deep learning framework for the model definition and training procedures. 
    It is designed to be modular, allowing easy adaptation and extension to other models and datasets.
    """

    def __init__(self, *args, **kwargs):
        super(NeuralNetworkUnitTest, self).__init__(*args, **kwargs)
        self.model = SimpleMNISTNet()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalizing MNIST dataset
        ])
        # Using torchvision's MNIST dataset
        self.dataset = datasets.MNIST('mnist_data', train=True, download=True, transform=self.transform)
        self.input_dim = (1,28,28)  # MNIST images are 28x28 pixels in size and grayscale


    def test_model_initialization(self):
        try:
            model = self.model
        except Exception as e:
            self.fail(f"Model initialization failed with an error: {e}. Suggestion: Check the model's __init__ method for errors.")

        self.assertIsInstance(model, SimpleMNISTNet, "The initialized object is not an instance of SimpleNet. Suggestion: Ensure that SimpleNet is defined correctly.")


    def test_input_output_dimension(self):
        model = self.model # Assuming the model is defined correctly
        dummy_input = torch.randn(1, *self.input_dim)  # Assuming input size of 10

        try:
            output = model(dummy_input)
        except Exception as e:
            self.fail(f"Model failed to process input with error: {e}. Suggestion: Check the forward method for compatibility with input dimensions.")

        expected_output_dim = (1, 10)  # Assuming output size of 2
        actual_output_dim = output.shape

        self.assertEqual(actual_output_dim, expected_output_dim,
                         f"Output dimension mismatch. Expected: {expected_output_dim}, Got: {actual_output_dim}. Suggestion: Ensure model's output layer matches expected output dimensions.")


    def test_training_step(self):
        model = self.model
        optimizer = self.optimizer
        criterion = self.criterion

        data = torch.randn(10, *self.input_dim)  # Assuming input size of 10
        labels = torch.randint(0, 2, (10,))  # Assuming binary classification

        try:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        except Exception as e:
            self.fail(f"Training step failed with an error: {e}. Suggestion: Check the forward pass, loss computation, and backward pass for compatibility and correctness.")

        # Check if gradients are non-zero for at least one parameter
        has_non_zero_gradients = any(param.grad is not None and param.grad.sum() != 0 for param in model.parameters())
        self.assertTrue(has_non_zero_gradients,
                        "No gradients found in any parameter after training step. Suggestion: Check if the model's parameters require_grad and the loss function.")


    def test_loss_computation(self):
        model = self.model
        criterion = self.criterion

        # Generate a predictable output and label
        outputs = torch.randn(1, 2)  # Assuming output size of 2
        target = torch.tensor([1])    # Sample target label

        try:
            loss = criterion(outputs, target)
        except Exception as e:
            self.fail(f"Loss computation failed with an error: {e}. Suggestion: Check the compatibility of the model's outputs and the target labels with the loss function.")

        self.assertTrue(torch.is_tensor(loss), "Computed loss is not a tensor. Suggestion: Ensure the loss function returns a tensor.")

        # Additional checks can be performed here, such as ensuring the loss is not NaN or infinity
        self.assertFalse(torch.isnan(loss) or torch.isinf(loss), "Loss is NaN or infinity. Suggestion: Check for numerical stability issues in the model's output.")


    def test_gradient_flow(self):
        model = self.model
        criterion = self.criterion
        optimizer = self.optimizer

        # Generate dummy data and labels
        data = torch.randn(1, *self.input_dim)  # Assuming input size of 10
        labels = torch.tensor([1])  # Sample target label

        try:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        except Exception as e:
            self.fail(f"An error occurred during the training step: {e}. Suggestion: Check the forward pass, loss computation, and backward pass.")

        # Check if gradients are present and non-zero in at least one parameter
        has_non_zero_gradients = any(param.grad is not None and not torch.all(param.grad == 0) for param in model.parameters())
        self.assertTrue(has_non_zero_gradients,
                        "Gradients are missing or zero for all parameters. Suggestion: Ensure the model's layers are properly connected and parameters require gradients.")


    def test_overfitting_on_small_data(self):
        model = self.model
        optimizer = self.optimizer
        criterion = self.criterion

        # Small dataset: just a few data points
        data = torch.randn(5, *self.input_dim)  # Assuming input size of 10
        labels = torch.randint(0, 2, (5,))  # Random binary labels

        try:
            for _ in range(100):  # A large number of epochs to encourage overfitting
                optimizer.zero_grad()
                outputs = model(data)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        except Exception as e:
            self.fail(f"Training failed with an error: {e}. Suggestion: Check the model's architecture and the training loop.")

        # Check if the model has overfitted: loss should be very low
        final_loss = criterion(model(data), labels)
        self.assertTrue(final_loss < 0.01, f"Model did not overfit the small dataset. Final loss: {final_loss}. Suggestion: Check if the model's capacity is sufficient to overfit.")

    def test_data_preprocessing_and_augmentation(self):
        # Define a simple dataset with preprocessing and augmentation
        class TestDataset(torch.utils.data.Dataset):
            def __init__(self):
                self.input_dim = (1,28,28)
                self.data = torch.randn(100, *self.input_dim)  # Example: 100 images, 3 channels, 64x64 size
                self.transform = transforms.Compose([
                    transforms.Resize((32, 32)),  # Resize images
                    transforms.RandomHorizontalFlip(),  # Random horizontal flip
                    # Add other transformations as needed
                ])

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                sample = self.data[idx]
                if self.transform:
                    sample = self.transform(sample)
                return sample

        dataset = self.dataset
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=10)

        try:
            for batch,_ in data_loader:
                self.assertEqual(batch.shape, (10, *self.input_dim),
                                 "Batch shape does not match expected shape. Suggestion: Check the preprocessing and augmentation steps.")
                break  # Testing with just the first batch for this example
        except Exception as e:
            self.fail(
                f"Data processing failed with an error: {e}. Suggestion: Check the preprocessing and augmentation pipeline for compatibility issues.")


    def test_model_loading_and_saving(self):
        model = self.model
        optimizer = self.optimizer

        # Save the model and optimizer state
        model_path = "test_model.pth"
        try:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, model_path)
        except Exception as e:
            self.fail(f"Failed to save the model with error: {e}. Suggestion: Check file permissions and availability of disk space.")

        # Load the model and optimizer state
        loaded_model = self.model
        loaded_optimizer = self.optimizer
        try:
            checkpoint = torch.load(model_path)
            loaded_model.load_state_dict(checkpoint['model_state_dict'])
            loaded_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception as e:
            self.fail(f"Failed to load the model with error: {e}. Suggestion: Check the compatibility of the saved model and the current model architecture.")

        # Compare model parameters between original and loaded model
        for param, loaded_param in zip(model.parameters(), loaded_model.parameters()):
            self.assertTrue(torch.equal(param, loaded_param), "Model parameters do not match after loading. Suggestion: Ensure the model architecture has not changed between saving and loading.")

        # Cleanup: Remove the saved model file
        if os.path.exists(model_path):
            os.remove(model_path)

    def test_inference_mode(self):
        model = self.model
        model.eval()  # Set the model to evaluation mode

        # Generate a sample input
        sample_input = torch.randn(1, *self.input_dim)

        try:
            with torch.no_grad():  # Ensure no gradients are computed
                first_pass_output = model(sample_input)
                second_pass_output = model(sample_input)
        except Exception as e:
            self.fail(f"Model inference failed with an error: {e}. Suggestion: Check the forward pass for compatibility with input data and model's state.")

        # Check if the outputs are consistent across multiple inference passes
        self.assertTrue(torch.equal(first_pass_output, second_pass_output),
                        "Model outputs are inconsistent across inference passes. Suggestion: Ensure layers like dropout or batch normalization are correctly configured for inference.")


    def test_dependency(self):
        try:
            # Attempt to instantiate the model which may rely on certain dependencies
            model = self.model
        except Exception as e:
            self.fail(f"Model instantiation failed, possibly due to a dependency issue: {e}. Suggestion: Check that all dependencies are correctly installed and configured.")

        # Generate a sample input (assuming input size simial for SimpleNet)
        sample_input = torch.randn(1,*self.input_dim)

        try:
            # Perform a forward pass to test dependency integration within the model's operation
            with torch.no_grad():
                _ = model(sample_input)
        except Exception as e:
            self.fail(f"Model forward pass failed, indicating a potential issue with a dependency: {e}. Suggestion: Verify that external dependencies or custom layers are functioning as expected.")


    def test_hyperparameter_sensitivity(self):
        dummy_input = torch.randn(1, *self.input_dim)
        dummy_labels = torch.tensor([1])

        # List of learning rates to test
        learning_rates = [0.001, 0.01, 0.1]

        for lr in learning_rates:
            try:
                model = self.model
                criterion = self.criterion
                optimizer = self.optimizer

                # Perform a training step
                optimizer.zero_grad()
                outputs = model(dummy_input)
                loss = criterion(outputs, dummy_labels)
                loss.backward()
                optimizer.step()
            except Exception as e:
                self.fail(f"Training step failed for learning rate {lr} with an error: {e}. Suggestion: Check the model's compatibility with different learning rates.")

            # Additional checks can be performed here, such as verifying the loss value is reasonable
            self.assertFalse(torch.isnan(loss) or torch.isinf(loss), f"Loss is NaN or infinity at learning rate {lr}. Suggestion: Check for numerical stability issues.")


    def test_reproducibility(self):
        set_seed()  # Set a fixed seed for reproducibility

        # Common setup for both runs
        dummy_input = torch.randn(1, *self.input_dim)
        dummy_labels = torch.tensor([1])
        lr = 0.01

        # Function to perform a training step and return the loss
        def perform_training():
            model = self.model
            criterion = self.criterion
            optimizer = self.optimizer
            optimizer.zero_grad()
            outputs = model(dummy_input)
            loss = criterion(outputs, dummy_labels)
            loss.backward()
            optimizer.step()
            return loss.item()

        # Perform training twice under the same conditions
        set_seed()  # Ensure the same seed before each training run
        first_run_loss = perform_training()

        set_seed()  # Reset the seed again
        second_run_loss = perform_training()

        # Check if the losses from both runs are similar
        self.assertAlmostEqual(first_run_loss, second_run_loss, places=5,
                               msg="Model produced different results on two runs with the same seed. Suggestion: Ensure randomness is controlled and the model architecture is deterministic.")

    def test_error_handling(self):
        model = self.model

        # Testing with an invalid input shape
        invalid_input = torch.randn(1, *self.input_dim)  # Incorrect input shape assuming model expects size 10

        try:
            _ = model(invalid_input)
        except Exception as e:
            # Check if the error message is informative and appropriate
            self.assertIsInstance(e, RuntimeError,
                                  f"Expected a RuntimeError, got {type(e)}. Suggestion: Raise specific exceptions for known error scenarios.")
            self.assertIn('size mismatch', str(e),
                          "Error message is not informative about the size mismatch. Suggestion: Include detailed error descriptions.")
        else:
            self.fail(
                "Model did not raise an error with invalid input. Suggestion: Implement input validation in the model's forward method.")

        # Add more tests for different types of invalid inputs (e.g., wrong data type, out-of-range values)

    def test_performance_benchmarks(self):
        model = self.model
        model.eval()  # Set the model to evaluation mode

        # Generate a sample input
        sample_input = torch.randn(1, *self.input_dim)

        # Measure inference time
        start_time = time.time()
        with torch.no_grad():
            _ = model(sample_input)
        end_time = time.time()
        inference_time = end_time - start_time

        # Define a threshold for acceptable inference time (in seconds)
        max_acceptable_time = 0.1  # Example threshold
        self.assertTrue(inference_time < max_acceptable_time,
                        f"Inference time exceeds the acceptable threshold: {inference_time}s. Suggestion: Optimize the model for faster inference.")

        # Additional tests can be added for other performance metrics like memory usage


    def test_integration(self):
        # Example: Integration with a data pipeline and post-processing step

        # Define a simple data pipeline (assuming a preprocessing step)
        def preprocessing(data):
            # Example preprocessing
            return data * 2  # Placeholder for actual preprocessing logic

        # Define a post-processing step
        def postprocessing(output):
            # Example post-processing
            return output + 1  # Placeholder for actual post-processing logic

        model = self.model
        model.eval()

        # Generate a sample input (assuming input size of 10 for SimpleNet)
        sample_input = torch.randn(1, *self.input_dim)

        try:
            # Simulate the end-to-end workflow
            processed_input = preprocessing(sample_input)
            with torch.no_grad():
                raw_output = model(processed_input)
            final_output = postprocessing(raw_output)

            # Here, add assertions or checks relevant to your application
            self.assertTrue(isinstance(final_output, torch.Tensor),
                            "Final output is not a tensor. Suggestion: Check the post-processing step for compatibility with model output.")
        except Exception as e:
            self.fail(f"Integration test failed with an error: {e}. Suggestion: Review the entire pipeline for compatibility and correctness of each component.")

if __name__ == '__main__':
    unittest.main()
