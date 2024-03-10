from torch import nn
import torch

class LeNet_TryOut(nn.Module):
  def __init__(self):
    super(LeNet_TryOut, self).__init__()

    self.conv1 = nn.Conv2d(in_channels=1, out_channels=6,
                               kernel_size=5, padding=0, stride=1)
    self.conv2 = nn.Conv2d(in_channels=6, out_channels=16,
                               kernel_size=5, padding=0, stride=1)

    self.pool = nn.AvgPool2d(2,2)

    self.fc1 = nn.Linear(16*4*4, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84,10)

  def forward(self, x):
    x = self.pool(torch.tanh(self.conv1(x)))
    x = self.pool(torch.tanh(self.conv2(x)))
    x = x.view(-1, 16 * 4 * 4)  # Flatten the tensor
    x = torch.tanh(self.fc1(x))
    x = torch.tanh(self.fc2(x))
    x = self.fc3(x)  # Apply fc3 to x
    return x