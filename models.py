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

      # TODO: What should the output size of a Value function be?
      output_size = 0

      # TODO: Add more layers, non-linear functions, etc.
      self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
      """
      Run forward pass of network

      Input:
        x: input to network
      Output:
        output of network
      """
      # TODO: Update as more layers are added
      return self.linear(x)
