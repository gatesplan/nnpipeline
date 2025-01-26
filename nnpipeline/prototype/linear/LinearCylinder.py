import torch
import torch.nn as nn

from nnpipeline.prototype.Pipe import Pipe


class LinearCylinder(Pipe):

    def __init__(self, width:int, depth:int,
                 use_normalization:bool = True,
                 normalization:str = 'batch',
                 use_dropout:bool = False,
                 dropout_rate:float = 0.5):
        super(LinearCylinder, self).__init__()
        self.width = width
        self.expected_output = width
        self.depth = depth
        self.use_normalization = use_normalization
        self.normalization = normalization
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate

        self.layers = nn.Sequential()

        # Create layers
        for i in range(depth):
            self.layers.append(nn.Linear(width, width))

            # if its last layer
            if i == depth - 1:
                break

            # normalization
            if use_normalization:
                if normalization == 'batch':
                    self.layers.append(nn.BatchNorm1d(width))
                elif normalization == 'layer':
                    self.layers.append(nn.LayerNorm(width))
                else:
                    raise ValueError(f"Invalid normalization: {normalization}")

            # activation
            self.layers.append(nn.ReLU())

            # dropout
            if use_dropout:
                self.layers.append(nn.Dropout(dropout_rate))

    def forward(self, x: torch.Tensor):
        return self.layers(x)
