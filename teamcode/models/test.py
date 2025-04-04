import torch
import torch.nn as nn

# A simple model for debugging
class Tester(nn.Module):
    def __init__(self, wsize):
        super(Tester, self).__init__()
        self.fc = nn.Linear(12 * wsize, 1)  # 2 input features, 1 output (logit)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten to (b_size, 12 * wsize)
        return self.fc(x)