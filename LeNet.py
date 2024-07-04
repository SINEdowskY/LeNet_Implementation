from torch import nn
import torch

class LeNet_TryOut(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        # (1 kanał wejściowy, 6 kanałów wyjściowych, 5x5 kernel, padding 2)
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        # 2x2 average pooling, stride 2
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        # (6 input channels, 16 output channels, 5x5 kernel, no padding)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # 2x2 average pooling, stride 2
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # warstwa fully connected (16*5*5 wejść, 120 wyjść)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # warstwa fully connected (120 wejść, 84 wyjść)
        self.fc2 = nn.Linear(120, 84)
        # warstwa fully connected (84 wejść, 10 wyjść)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        #Pierwsza warstwa konwolucji z funkcją aktywacji sigmoid
        x = F.sigmoid(self.conv1(x))
        #Pierwsza warstwa poolingu
        x = self.pool1(x)
        #Druga warstwa konwolucji z funkcją aktywacji sigmoid
        x = F.sigmoid(self.conv2(x))
        #Druga warstwa poolingu                          
        x = self.pool2(x)           

        #Spłaszczanie tensora
        x = x.view(-1, 16 * 5 * 5)

        #Pierwsza warstwa fully connected z funkcją aktywacji sigmoid
        x = F.sigmoid(self.fc1(x))
        #Druga warstwa fully connected z funkcją aktywacji sigmoid
        x = F.sigmoid(self.fc2(x))
        #Druga warstwa fully connected z funkcją aktywacji sigmoid                         
        x = self.fc3(x)

        return x
