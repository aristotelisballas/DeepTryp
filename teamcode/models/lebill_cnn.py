import torch
import torch.nn.functional as F

from torch import nn


class LebillCNN(nn.Module):
    def __init__(self):
        super(LebillCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=12, out_channels=16, kernel_size=2, stride=1, padding=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=4, stride=1, padding=2)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=48, kernel_size=4, stride=1, padding=2)
        self.conv4 = nn.Conv1d(in_channels=48, out_channels=64, kernel_size=4, stride=1, padding=2)
        self.conv5 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2)

        self.maxpool_2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.maxpool_4 = nn.MaxPool1d(kernel_size=2, stride=2)


        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=8064, out_features=200)
        self.fc2 = nn.Linear(in_features=200, out_features=200)
        self.out = nn.Linear(in_features=200, out_features=1)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool_2(x)

        x = F.relu(self.conv2(x))
        x = self.maxpool_4(x)

        x = F.relu(self.conv3(x))
        x = self.maxpool_4(x)

        x = F.relu(self.conv4(x))
        x = self.maxpool_4(x)

        x = F.relu(self.conv5(x))
        x = self.maxpool_4(x)

        x = self.flatten(x)

        # Dense Part
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        out = self.out(x)

        return out



