# for-agent-layerinfo.md

## l0
- Candle, Token, Pair, StockAddress, Constants

## l1
- market: 단일 종목 시계열
- order: 주문 객체/검증
- pair: 자산-가치 쌍
- order_book: 호가창
- signal_interpreter: 신호 해석

## l2
- wallet: 다중 자산 지갑
- pair_stack: 포지션 스택
- trade: 체결 기록
- multiseries_tensor: 다중 종목 텐서

## l3
- portfolio: 포트폴리오 관리, 3계층 저장
- order_manager: 미체결 주문 관리
- order_issuer: 주문 발행/검증
- trade_simulator: 주문 체결 시뮬레이션
- trade_settler: 거래 정산
- benchmark: 벤치마크 계산

## l4
- market_order_executor: 시장가 주문 즉시 체결

## l5
- signal_performance_tester: 신호 전략 백테스트

## l6
- monte_carlo_tester: 몬테카를로 시뮬레이션
- stress_tester: 스트레스 테스트
