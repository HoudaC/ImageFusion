import torch
import torch.nn as nn


class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()

        # First convolutional layer (input: 10 channels, output: 64 channels, kernel size: 5x5)
        self.conv1 = nn.Conv2d(10, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        # Second convolutional layer (input: 64 channels, output: 32 channels, kernel size: 3x3)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()

        # Third convolutional layer (input: 32 channels, output: 10 channels, kernel size: 3x3)
        self.conv3 = nn.Conv2d(32, 10, kernel_size=3, padding=1)

        # Optionally, a transposed convolution layer for upsampling
        # self.upsample = nn.ConvTranspose2d(10, 10, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        # Residual connection: save original input
        residual = x

        # First block (conv + batchnorm + relu)
        x = self.bn1(self.relu1(self.conv1(x)))

        # Second block (conv + batchnorm + relu)
        x = self.bn2(self.relu2(self.conv2(x)))

        # Final convolution
        x = self.conv3(x)

        # Add residual connection
        x = x + residual  # Residual learning

        # Optionally: upsample
        # x = self.upsample(x)

        return x




class ConditionalSRCNN(nn.Module):
    def __init__(self, input_channels=14, output_channels=10):
        """
        Initialize the Conditional SRCNN Model.

        Parameters:
        - input_channels (int): The number of input channels (e.g., 10 Sentinel bands + 4 additional channels = 14).
        - output_channels (int): The number of output channels (should be the same as the original input channels).
        """
        super(ConditionalSRCNN, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        # Second convolutional layer
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()

        # Third convolutional layer
        self.conv3 = nn.Conv2d(32, output_channels, kernel_size=3, padding=1)

    def forward(self, low_res, conditional_input):
        """
        Forward pass of the Conditional SRCNN model.

        Parameters:
        - low_res (tensor): Low-resolution input image (Tensor of shape [batch_size, 10, H, W]).
        - conditional_input (tensor): Additional conditional input (Tensor of shape [batch_size, 4, H, W]).

        Returns:
        - output (tensor): High-resolution output image (Tensor of shape [batch_size, 10, H, W]).
        """
        # Concatenate low-res image and conditional input along the channel dimension
        x = torch.cat((low_res, conditional_input), dim=1)  # Concatenate along channels (C, H, W)

        # First block (conv + batchnorm + relu)
        x = self.bn1(self.relu1(self.conv1(x)))

        # Second block (conv + batchnorm + relu)
        x = self.bn2(self.relu2(self.conv2(x)))

        # Final convolution (output)
        x = self.conv3(x)

        return x