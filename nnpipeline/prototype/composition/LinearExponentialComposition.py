from torch.nn import ModuleList
from nnpipeline import Pipe, LinearJoint, LinearExponentialEncoder


class LinearExponentialComposition(Pipe):
    """
    LinearExponentialComposition 파이프

    여러 Pipe(선형층 등)를 합성한 뒤,
    Joint 층(LinearJoint)으로 결합하고,
    ExponentialEncoder(LinearExponentialEncoder)로 압축/처리하여 출력하는 과정.
    """

    def __init__(self, compositioning_pipes: list[Pipe], output_features: int, options: dict = None):
        super(LinearExponentialComposition, self).__init__()

        # 파이프 저장
        self.pipes = ModuleList(compositioning_pipes)
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

        # 여러 파이프를 결합하는 Joint
        self.joint = LinearJoint(input_pipes=compositioning_pipes)

        # Joint로 합쳐진 결과를 압축하는 ExponentialEncoder
        self.encoder = LinearExponentialEncoder(
            in_features=sum(self.input_sizes),
            out_features=output_features,
            **generated_options
        )

    def forward(self, *args):
        """
        입력값은 각 파이프에 들어갈 데이터이며, 순서대로 self.pipes에 대입.
        -> 그 결과들을 Joint 층에 통합.
        -> Encoder로 압축/변환.
        -> 최종 결과 반환.
        """
        # 1) 각 파이프에 대응하는 입력값을 전달하여 결과 리스트를 얻음
        pipe_outputs = [pipe(arg) for pipe, arg in zip(self.pipes, args)]

        # 2) Joint로 결합
        joined_output = self.joint(*pipe_outputs)

        # 3) Encoder로 압축/처리
        final_output = self.encoder(joined_output)

        # 4) 최종 결과 반환
        return final_output
