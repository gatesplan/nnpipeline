# for-agent-layerinfo.md

nnpipeline 전체 모듈 목록 (저해상도).

## prototype
- Cylinder: 동일 폭 1D MLP 빌더
- Pyramid: 선형 보간 폭 1D MLP 빌더
- OHLCVReceptor: 캔들 + 거래량 → 3-dim 임베딩 per-candle tokenizer
- ReceptorBundle: receptor/bundle 을 자식으로 묶는 multi-resolution composite 컨테이너
