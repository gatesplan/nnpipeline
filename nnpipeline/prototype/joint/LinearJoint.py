import torch

from nnpipeline.prototype.Pipe import Pipe


class LinearJoint(Pipe):
    """
    여러 선형층을 하나의 층으로 합치는 연결부 파이프라인 네트워크.
    초기화시 각 선형층 인스턴스를 인자로 받아 출력값 사이즈를 측정하여 연결해 사용한다.
    """
    expected_output = 0

    def __init__(self, input_pipes:list[Pipe]):
        super(LinearJoint, self).__init__()

        # 각 입력 파이프라인의 출력값 배열
        self.input_sizes = [pipe.get_expected_output_size() for pipe in input_pipes]

        # 총 입력 크기
        self.total_input_size = sum(self.input_sizes)

        # 출력 크기
        self.expected_output = self.total_input_size
        # 출력 크기는 각 입력 파이프라인의 출력 크기의 합이다.
        # 뉴런 연결과 비교하면 이 구현은 다수의 뉴런 커넥톰으로부터 전달된 시그널이
        # 하나의 단말로 들어오는 병합 과정이며
        # 이 클래스는 이 병합을 본격적으로 수행하기 전 입력 크기에 대한 단일 층 연산으로,
        # 각 입력으로 인해 길이가 길어진 데이터를 단순 연결하여 계산 없이 반환한다.


    def forward(self, *args):
        """
        입력된 데이터를 연결하여 반환한다.
        """
        try:
            return torch.cat(args, dim=0)
        except Exception as e:
            # 입력 데이터 길이 검사
            if len(args) != len(self.input_sizes):
                raise ValueError(f"입력 데이터 길이가 {len(self.input_sizes)}개가 아닙니다.")

            if not all([arg.size(1) == size for arg, size in zip(args, self.input_sizes)]):
                raise ValueError("입력 데이터 크기가 일치하지 않습니다.")

            raise e
