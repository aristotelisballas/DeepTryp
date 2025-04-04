import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Weighting factor for the class
        self.gamma = gamma  # Focusing parameter
        self.reduce = reduce  # Whether to average the loss or not

    def forward(self, inputs, targets):
        # Sigmoid activation to get probabilities
        sigmoid_p = torch.sigmoid(inputs)

        # Get the prediction probability for the true class (i.e., targets)
        p_t = sigmoid_p * targets + (1 - sigmoid_p) * (1 - targets)

        # Compute the loss
        loss = -self.alpha * (1 - p_t) ** self.gamma * torch.log(p_t + 1e-8)  # Prevent log(0)

        if self.reduce:
            return loss.mean()  # Average the loss over the batch
        else:
            return loss  # Return the loss for each sample in the batch

