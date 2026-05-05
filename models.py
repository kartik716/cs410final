import torch.nn as nn
import torch.nn.functional as F
import torch

def save_model(path: str, model):
    """
    Save model to a file
    Input:
        path: path to save model to
        model: Pytorch model to save
    """
    torch.save({
        'model_state_dict': model.state_dict(),
    }, path)

def load_model(path: str, model):
    """
    Load model from file

    Note: you still need to provide a model (with the same architecture as the saved model))

    Input:
        path: path to load model from
        model: Pytorch model to load
    Output:
        model: Pytorch model loaded from file
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

class ValueNetwork(nn.Module):
    def __init__(self, input_size):
        super(ValueNetwork, self).__init__()
        
        # Output size: 1 for value prediction (scalar between -1 and 1)
        output_size = 1
        
        # Hidden layers for better representation learning
        hidden_size1 = 128
        hidden_size2 = 64
        hidden_size3 = 32
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, hidden_size3),
            nn.ReLU(),
            nn.Linear(hidden_size3, output_size),
            nn.Tanh()  # Tanh to output values between -1 and 1
        )

    def forward(self, x):
        """
        Run forward pass of network

        Input:
            x: input to network
        Output:
            output of network
        """
        return self.network(x)