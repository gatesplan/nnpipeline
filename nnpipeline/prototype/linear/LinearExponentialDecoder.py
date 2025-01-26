import torch.nn as nn

from nnpipeline.prototype.Pipe import Pipe
from nnpipeline.tools import generate_exponential_int_sequence


class LinearExponentialDecoder(Pipe):
    def __init__(self, in_features:int, out_features:int, expansion_rate:float = 1.618,
                 use_normalization:bool = True, normalization:str = 'batch',
                 use_dropout:bool = False, dropout_rate:float = 0.5):

        super(LinearExponentialDecoder, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.expected_output = out_features
        self.expansion_rate = expansion_rate
        self.use_normalization = use_normalization
        self.normalization = normalization
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate

        self.layers = nn.Sequential()

        node_seq = list(generate_exponential_int_sequence(in_features,
                                                          out_features,
                                                          expansion_rate))
        head = node_seq[:-1]
        tail = node_seq[1:]

        for i, (h, t) in enumerate(zip(head, tail)):
            self.layers.append(nn.Linear(h, t))

            if tail == out_features:
                break

            if use_normalization:
                if normalization == 'batch':
                    self.layers.append(nn.BatchNorm1d(t))
                elif normalization == 'layer':
                    self.layers.append(nn.LayerNorm(t))
                else:
                    raise ValueError(f"Invalid normalization: {normalization}")

            self.layers.append(nn.ReLU())

            if use_dropout:
                self.layers.append(nn.Dropout(dropout_rate))