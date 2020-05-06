import numpy as np

from pathlib import Path

import torch
from torch import nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def dilate(x, s):
    return torch.max(x + s, dim=1)


def erode(x, s):
    return torch.min(x - s, dim=1)


class DilateErode(nn.Module):

    def __init__(self, in_features: int, number_of_dilations: int, number_of_erosions: int):
        super().__init__()
        self.in_features = in_features
        self.number_of_dilations = number_of_dilations
        self.number_of_erosions = number_of_erosions

        if self.number_of_dilations > 0:
            # +1 for the bias
            init_dilation_weight = nn.init.xavier_uniform_(torch.zeros(in_features + 1, number_of_dilations))
            self.dilations = nn.Parameter(init_dilation_weight)
        else:
            self.dilations = torch.Tensor()

        if self.number_of_erosions > 0:
            # +1 for the bias
            init_erosion_weight = nn.init.xavier_uniform_(torch.zeros(in_features + 1, number_of_erosions))
            self.erosions = nn.Parameter(init_erosion_weight)
        else:
            self.erosions = torch.Tensor()
        self.dilation_bias = nn.Parameter(torch.zeros(number_of_dilations))
        self.erosion_bias = nn.Parameter(torch.zeros(number_of_erosions))

    def forward(self, input: torch.Tensor):
        batch_size = input.shape[0]
        flattened = input.view(batch_size, self.in_features, 1)
        print(flattened.shape)
        flattened_with_bias = torch.cat((flattened, torch.zeros(batch_size, 1, 1).to(device)), dim=1)
        print(flattened_with_bias.shape)

        if self.number_of_dilations > 0:
            # Each dilation is a max of a sum of all the input features.
            dsum = flattened_with_bias + self.dilations
            dilated = torch.max(dsum, dim=1)[0]
        else:
            dilated = torch.Tensor().to(device)

        if self.number_of_erosions > 0:
            # Each erosion is a min of a difference of all the input features.
            esub = flattened_with_bias - self.erosions
            eroded = torch.min(esub, dim=1)[0]
        else:
            eroded = torch.Tensor().to(device)

        combined = torch.cat((eroded, dilated), dim=1)
        return combined


class DenMoNet(nn.Module):
    """The dilation-erosion network."""

    def __init__(self, input_space_dim: int, number_dilations: int, number_erosions: int, output_space_dim: int):
        super().__init__()
        self.de_layer = DilateErode(input_space_dim, number_dilations, number_erosions)
        # The linear combination size is the number of erosions plus the number of dilations, plus
        # one bias node for each (if there's at least one, that is).
        lc_size = number_erosions + number_dilations
        self.linear_combination_layer = nn.Linear(lc_size, output_space_dim)

    def name(self, dset_name: str):
        return "denmo_{}x{}_{}".format(self.de_layer.number_of_dilations,
                self.de_layer.number_of_erosions, dset_name)

    def forward(self, input: torch.Tensor):
        print(input.shape)
        temp = self.de_layer(input)
        self.temp = temp
        print(temp.shape)
        classification = self.linear_combination_layer(temp)
        print(classification.shape)
        return classification

    def store(self, dset_name: str, directory: Path):
        name = self.name(dset_name)
        fname = "{}.tch".format(name)
        torch.save(self, str(directory / fname))

class DeroNet(nn.Module):
    """The dilation-erosion network."""

    def __init__(self, input_space_dim: int, number_dilations: int, number_erosions: int, output_space_dim: int):
        super().__init__()
        self.de_layer = DilateErode(input_space_dim, number_dilations, number_erosions)
        # The linear combination size is the number of erosions plus the number of dilations, plus
        # one bias node for each (if there's at least one, that is).
        lc_size = number_erosions + number_dilations
        self.linear_combination_layer = nn.Linear(lc_size, output_space_dim)
        self.dim = input_width
        self.conv1 = nn.Conv2d(input_channels, 50, kernel_size=3)
        self.conv2 = nn.Conv2d(50, 100, kernel_size=5)
        self.conv3 = nn.Conv2d(100, 200, kernel_size=5)
        self.conv4 = nn.Conv2d(200, 400, kernel_size=2)
        self.fc1 = nn.Linear(400, 300)
        self.fc2 = nn.Linear(300, num_classes)
    def name(self, dset_name: str):
        return "dero_{}x{}_{}".format(self.de_layer.number_of_dilations,
                self.de_layer.number_of_erosions, dset_name)

    def forward(self, input: torch.Tensor):
        x = F.relu(self.conv1(input))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.relu(self.conv4(x), 2)
        x = x.view(-1, 400)
        

    def store(self, dset_name: str, directory: Path):
        name = self.name(dset_name)
        fname = "{}.tch".format(name)
        torch.save(self, str(directory / fname))
        
        
        
class SimpleCNN(nn.Module):
    def __init__(self, input_channels, input_width, num_classes):
        super(SimpleCNN, self).__init__()
        self.dim = input_width
        self.conv1 = nn.Conv2d(input_channels, 50, kernel_size=3)
        self.conv2 = nn.Conv2d(50, 100, kernel_size=5)
        self.conv3 = nn.Conv2d(100, 200, kernel_size=5)
        self.conv4 = nn.Conv2d(200, 400, kernel_size=2)
        self.fc1 = nn.Linear(400, 300)
        self.fc2 = nn.Linear(300, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.relu(self.conv4(x), 2)
        x = x.view(-1, 400)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

