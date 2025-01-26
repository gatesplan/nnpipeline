"""
BasePipe
Pipeline 기본 인터페이스를 정의
"""
import torch.nn as nn


class BasePipe:
    expected_output:int = 0

    def get_expected_output_size(self) -> int:
        return self.expected_output