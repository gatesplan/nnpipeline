def generate_exponential_int_sequence(start: int, end: int, rate: float):
    """
    정수 지수 수열 생성기
    """

    if rate == 1:
        raise ValueError("Rate should not be 1. (변화가 없으므로)")

    # 내부적으로 다음 값을 생성하는 보조 함수
    def make_next_value(prev_value: int, multiplier: float):
        next_value = int(prev_value * multiplier)
        # 만약 곱해도 값이 그대로라면 (정수부 손실 때문에) +1 또는 -1로 보정
        if next_value == prev_value:
            if multiplier > 1:
                next_value += 1
            else:
                next_value -= 1
        return next_value

    value = start

    # 감소 수열
    if rate < 1:
        while value > end:
            yield value
            value = make_next_value(value, rate)

    else:
        # 증가 수열
        while value < end:
            yield value
            value = make_next_value(value, rate)

    yield end
