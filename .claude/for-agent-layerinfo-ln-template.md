# for-agent-layerinfo-ln.md 템플릿

레벨별 중해상도 API 문서. 해당 레벨의 모든 모듈 공개 메서드 시그니처 나열.

## 위치

```
src/fishfactory/
  ln/
    for-agent-layerinfo-ln.md
```

예시:
- `l0/for-agent-layerinfo-l0.md`
- `l1/for-agent-layerinfo-l1.md`
- `l2/for-agent-layerinfo-l2.md`

## 형식

```markdown
# ln

## modulename
ClassName.__init__(args) -> None
ClassName.method_name(args) -> return_type
ClassName.another_method(args) -> return_type

## another_module
ClassName.__init__(args) -> None
ClassName.method_name(args) -> return_type
```

**규칙:**
- 시그니처만 작성 (설명 없음)
- 한 줄에 하나씩
- `ClassName.method_name()` 형식
- 함수명이 자기설명적이어야 함

## 예시: for-agent-layerinfo-l1.md

```markdown
# l1

## order
Order.__init__(symbol: str, side: str, price: float, quantity: float)
Order.fill(quantity: float) -> Trade
Order.cancel() -> None
Order.get_unfilled() -> float
Order.is_filled() -> bool

## pair
Pair.__init__(token: Token, value: float)
Pair.merge(other: Pair) -> Pair
Pair.split(ratio: float) -> tuple[Pair, Pair]
Pair.get_average_price() -> float

## market
Market.__init__(candles: list[Candle])
Market.get_at(timestamp: int) -> Candle
Market.get_range(start: int, end: int) -> list[Candle]
Market.to_dataframe() -> DataFrame

## order_book
OrderBook.__init__(symbol: str)
OrderBook.add_bid(price: float, quantity: float) -> None
OrderBook.add_ask(price: float, quantity: float) -> None
OrderBook.get_best_bid() -> float | None
OrderBook.get_best_ask() -> float | None
OrderBook.match_order(side: str, quantity: float) -> list[tuple[float, float]]
```

## 3단계 해상도 구조

```
1. for-agent-layerinfo.md (저해상도)
   위치: src/fishfactory/
   내용: l1: order, pair, market, order_book

2. for-agent-layerinfo-l1.md (중해상도)
   위치: src/fishfactory/l1/
   내용: Order.fill(quantity: float) -> Trade

3. for-agent-moduleinfo.md (고해상도)
   위치: src/fishfactory/l1/order/
   내용: fill() 상세 설명, 예외, 설계 이유, 동작 방식
```

## 에이전트 활용

```
사용자: "주문 취소 함수 어디있어?"

에이전트:
1. for-agent-layerinfo.md 읽기
   → l1에 order 모듈 있음
2. l1/for-agent-layerinfo-l1.md 읽기
   → Order.cancel() -> None 발견
3. 필요시 l1/order/for-agent-moduleinfo.md 읽기
   → 상세 정보
```
