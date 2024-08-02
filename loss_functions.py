import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np


class L1AngularLoss(nn.Module):
    """Loss function to the nearest 0 or 180"""

    def __init__(self):
        super(L1AngularLoss, self).__init__()


    def forward(self, predicted, target):
        # Normalize predicted values to the range [0, 180]
        predicted = predicted % 180
        predicted = (predicted + 180) % 180

        # Compute the angular difference
        angular_diff = torch.abs(predicted - target)
        angular_diff = torch.min(angular_diff, 180 - angular_diff)

        return torch.mean(angular_diff)
    

class L2AngularLoss(nn.Module):
    """MSE Loss function to the nearest 0 or 180"""
    def __init__(self):
        super(L2AngularLoss, self).__init__()

    def forward(self, predicted, target):
        # Normalize predicted values to the range [0, 180]
        predicted = predicted % 180
        predicted = (predicted + 180) % 180

        # Compute the angular difference
        angular_diff = torch.abs(predicted - target)
        angular_diff = torch.min(angular_diff, 180 - angular_diff)

        return torch.mean(angular_diff ** 2)
    

class OpeningLoss(nn.Module):
    def __init__(self):
        super(OpeningLoss, self).__init__()

    def forward(self, predicted, target):
        # Normalize predicted values to the range [0, 10]
        predicted = predicted % 10
        predicted = (predicted + 10) % 10

        # Compute the angular difference
        angular_diff = torch.abs(predicted - target)
        angular_diff = torch.min(angular_diff, 10 - angular_diff)

        more_than_one = angular_diff > 1
        angular_diff[more_than_one] = 10**angular_diff[more_than_one]

        return torch.mean(angular_diff)
    
class MultiLoss(nn.Module):
    def __init__(self, strike_weight, opening_weight):
        super(MultiLoss, self).__init__()
        self.strike_weight = strike_weight
        self.opening_weight = opening_weight


    def forward(self, predicted, target):
        strike_predicted = predicted[:, 0]
        opening_predicted = predicted[:, 1]

        strike_target = target['strike']
        opening_target = target['opening']

        strike_loss = L1AngularLoss()(strike_predicted, strike_target)
        opening_loss = OpeningLoss()(opening_predicted, opening_target)

        total_loss = self.strike_weight*strike_loss + self.opening_weight*opening_loss

        return torch.mean(total_loss)