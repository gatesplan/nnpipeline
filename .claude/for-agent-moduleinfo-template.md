# for-agent-moduleinfo.md 템플릿

모듈별 고해상도 상세 문서. 클래스 API, 동작 방식, 예외, 설계 이유 설명.

## 위치

```
src/fishfactory/
  ln/
    modulename/
      for-agent-moduleinfo.md
```

## 형식

```markdown
# ModuleName

모듈 목적 1-2문장.

## ClassName

클래스 목적 1-2문장.

### Properties
property_name: type          # 설명
another_property: type       # 설명

### __init__
__init__(arg1: type, arg2: type = default)
    raise ExceptionType
    초기화 동작 설명.
    검증 로직, 제약 조건 설명.

### Methods

method_name(arg1: type, arg2: type = default) -> return_type
    동작 설명.
    파라미터 상세 (필요시).
    반환값 형식 (필요시).

another_method() -> return_type
    raise ExceptionType
    동작 설명.
    예외 발생 조건.
```

**규칙:**
- 시그니처 정확히 작성
- `raise` 문으로 예외 표시
- 주석은 `#` 사용
- 메서드별 독립 블럭

## 예시: l1/order/for-agent-moduleinfo.md

```markdown
# Order

주문 객체 및 검증.

## Order

주문 정보 관리. 체결, 취소, 상태 추적.

### Properties
symbol: str                  # 거래쌍
side: str                    # "buy" 또는 "sell"
price: float                 # 주문 가격
quantity: float              # 주문 수량
filled_quantity: float       # 체결된 수량

### __init__
__init__(symbol: str, side: str, price: float, quantity: float)
    raise ValueError
    주문 객체 초기화.
    price, quantity는 양수여야 함.
    side는 "buy" 또는 "sell"만 허용.

### Methods

fill(quantity: float) -> Trade
    raise ValueError
    부분 체결 처리.
    filled_quantity 증가.
    quantity는 양수, 미체결 수량 이하여야 함.
    Trade 객체 반환.

cancel() -> None
    주문 취소.
    상태를 "cancelled"로 변경.

get_unfilled() -> float
    미체결 수량 반환.
    quantity - filled_quantity 계산.

is_filled() -> bool
    완전 체결 여부 반환.
    filled_quantity == quantity 확인.
```

## 3단계 해상도 구조

```
1. for-agent-layerinfo.md (저해상도)
   l1: order, pair, market

2. for-agent-layerinfo-l1.md (중해상도)
   Order.fill(quantity: float) -> Trade

3. for-agent-moduleinfo.md (고해상도)
   fill() 상세 설명, 예외, 동작
```
