import torch
import torch.nn as nn
import numpy as np

# This class defines the DQN network structure
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, filename):
        super(DQN, self).__init__()
        self.input_dim = input_dim
        channels, _, _ = input_dim

        # 3 conv layers, all with relu activations, first one with maxpool
        self.l1 = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # Calculate output dimensions for linear layer
        conv_output_size = self.conv_output_dim()
        lin1_output_size = 512

        # Two fully connected layers with one relu activation
        self.l2 = nn.Sequential(
            nn.Linear(conv_output_size, lin1_output_size),
            nn.ReLU(),
            nn.Linear(lin1_output_size, output_dim)
        )

        # Save filename for saving model
        self.filename = filename

    # Calulates output dimension of conv layers
    def conv_output_dim(self):
        x = torch.zeros(1, *self.input_dim)
        x = self.l1(x)
        return int(np.prod(x.shape))

    # Performs forward pass through the network, returns action values
    def forward(self, x):
        x = self.l1(x)
        x = x.view(x.shape[0], -1)
        actions = self.l2(x)

        return actions

    # Save a model
    def save_model(self):
        torch.save(self.state_dict(), './models/' + self.filename + '.pth')

    # Loads a model
    def load_model(self):
        self.load_state_dict(torch.load('./models/' + self.filename + '.pth'))