import torch.nn as nn

from nnpipeline.prototype.Pipe import Pipe
from nnpipeline.tools import generate_exponential_int_sequence


class LinearExponentialEncoder(Pipe):
    """
    Linear Exponential Encoder 선형층 지수적 인코더
    선형층의 크기를 지수적으로 줄여 데이터를 압축하는 파이프라인.
    """

    def __init__(self, in_features: int, out_features: int, compression_rate: float = 0.618,
                 use_normalization: bool = True, normalization: str = 'batch',
                 use_dropout: bool = False, dropout_rate: float = 0.5):

        super(LinearExponentialEncoder, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.expected_output = out_features
        self.compression_rate = compression_rate
        self.use_normalization = use_normalization
        self.normalization = normalization
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate

        self.layers = nn.Sequential()

        node_seq = list(generate_exponential_int_sequence(in_features,
                                                          out_features,
                                                          compression_rate))
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

    def forward(self, x):
        return self.layers(x)