import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMNISTNet(nn.Module):
    def __init__(self):
        super(SimpleMNISTNet, self).__init__()
        # First convolutional layer (1 input channel, 10 output channels, kernel size 5)
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # Second convolutional layer (10 input channels, 20 output channels, kernel size 5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # Dropout layer
        self.conv2_drop = nn.Dropout2d()
        # First fully connected layer
        self.fc1 = nn.Linear(320, 50)
        # Second fully connected layer (output layer)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # Apply first convolutional layer, then ReLU, then max pooling
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # Apply second convolutional layer, then ReLU, then max pooling
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # Flatten the tensor
        x = x.view(-1, 320)
        # Apply first fully connected layer, then ReLU
        x = F.relu(self.fc1(x))
        # Apply dropout
        x = F.dropout(x, training=self.training)
        # Apply second fully connected layer
        x = self.fc2(x)
        # Apply log softmax to the output
        return F.log_softmax(x, dim=1)
