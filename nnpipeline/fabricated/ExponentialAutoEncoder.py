from nnpipeline import Pipe, LinearExponentialEncoder, LinearExponentialDecoder


class ExponentialAutoEncoder(Pipe):
    """
    training 모드에 따라 다르게 동작하는 Exponential Auto Encoder
    지수적 압축/확장을 수행하는 오토인코더이며,
    train 모드일 때 decoded된 값을 반환하고
    eval 모드일 때 encoded된 값을 반환한다.
    """
    def __init__(self, input_size:int, output_size:int,
                 compression_rate:float = 0.618):
        super(ExponentialAutoEncoder, self).__init__()
        self.encoder = LinearExponentialEncoder(input_size, output_size,
                                                compression_rate=compression_rate)
        self.decoder = LinearExponentialDecoder(output_size, input_size,
                                                expansion_rate=1/compression_rate)

    def forward(self, x):
        if self.training:
            x = self.encoder(x)
            x = self.decoder(x)
            return x

        else:
            return self.encoder(x)

