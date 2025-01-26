from nnpipeline import Pipe, LinearJoint, LinearExponentialEncoder


class LinearExponentialComposition(Pipe):
    """
    LinearComposition 선형층 합성 파이프
    여러 선형층을 합성하는 파이프.
    """
    def __init__(self, compositioning_pipes: list[Pipe], output_features:int, options: dict = None):
        super(LinearExponentialComposition, self).__init__()

        self.input_sizes = [pipe.get_expected_output_size() for pipe in compositioning_pipes]
        self.expected_output = output_features

        options = options or {}
        generated_options = {
            'compression_rate': options.get('compression_rate', 0.618),
            'use_normalization': options.get('use_normalization', True),
            'normalization': options.get('normalization', 'batch'),
            'use_dropout': options.get('use_dropout', False),
            'dropout_rate': options.get('dropout_rate', 0.5),
        }

        # 결합층
        self.joint = LinearJoint(input_pipes=compositioning_pipes)

        # 압축 선형층
        self.encoder = LinearExponentialEncoder(
            in_features=sum(self.input_sizes),
            out_features=output_features,
            **generated_options
        )

    def forward(self, *args):
        return self.encoder(self.joint(*args))

