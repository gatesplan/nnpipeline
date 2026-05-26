# Ln 구조 표준

Ln 레벨 기반 계층 구조의 디렉토리 및 파일 배치 규칙.

## 기본 규칙

1. **항상 폴더 구조 사용**
   - 모든 모듈은 `ln/modulename/` 폴더로 구성
   - 단순/복잡 여부 무관하게 일관성 유지

2. **l0 = core**
   - 외부 라이브러리 의존 없음 (표준 라이브러리만)
   - 기본 데이터 타입, 상수

3. **미러 테스트 구조**
   - `tests/` 구조는 `src/` 구조와 동일
   - `src/l1/order/` → `tests/l1/order/`

## 디렉토리 구조

### 단순 모듈

```
src/fishfactory/
  ln/
    modulename/
      modulename.py
      for-agent-moduleinfo.md
      __init__.py

tests/
  ln/
    modulename/
      test_modulename.py
```

### 복잡 모듈 (중첩 Ln)

```
src/fishfactory/
  ln/
    modulename/
      l0/
      l1/
      l2/
      for-agent-moduleinfo.md
      __init__.py

tests/
  ln/
    modulename/
      test_modulename.py
      test_integration.py
```

## __init__.py 패턴

**규칙: 항상 상대 임포트 사용**

### 모듈 __init__.py

```python
# src/fishfactory/l1/order/__init__.py

from .order import Order  # 상대 임포트

__all__ = ['Order']
```

**이유:**
- 모듈 이동 시 경로 불변 (l1 -> l2 이동 시 수정 불필요)
- 레벨 변경 자동 반영 (디렉토리 위치 = 계층 상태)
- 다른 프로젝트 복사 시 패키지명 변경 불필요

**효과:**
```python
# 사용자 코드
from fishfactory.l1.order import Order  # 짧은 import
```

### 레이어 __init__.py

각 레이어 폴더(ln/)의 `__init__.py`는 해당 레이어의 모든 모듈을 re-export한다.

```python
# src/fishfactory/l1/__init__.py

from .order import Order
from .pair import Pair

__all__ = ['Order', 'Pair']
```

**효과:**
```python
from fishfactory.l1 import Order, Pair  # 레이어 단위 import
```

### 패키지 최상단 __init__.py

패키지 루트의 `__init__.py`는 최상위 레이어의 메인 비즈니스 모듈만 노출한다.

```python
# src/fishfactory/__init__.py

from .l3.portfolio import Portfolio  # 최상위 파사드만

__all__ = ['Portfolio']
```

**효과:**
```python
from fishfactory import Portfolio  # 최단 경로 import
```

## 3단계 해상도 문서 구조

```
1. for-agent-layerinfo.md (저해상도)
   위치: src/fishfactory/
   내용: 전체 시스템 모듈 목록

2. for-agent-layerinfo-ln.md (중해상도)
   위치: src/fishfactory/ln/
   내용: 레벨별 모든 모듈 공개 메서드 시그니처

3. for-agent-moduleinfo.md (고해상도)
   위치: src/fishfactory/ln/modulename/
   내용: 모듈 상세 설명, 예외, 설계 이유
```

## 전체 예시

```
project/
  src/fishfactory/
    for-agent-layerinfo.md                # 저해상도 (전체)

    l0/
      for-agent-layerinfo-l0.md           # 중해상도 (l0)
      candle/
        candle.py
        for-agent-moduleinfo.md           # 고해상도 (candle)
        __init__.py
      token/
        token.py
        for-agent-moduleinfo.md           # 고해상도 (token)
        __init__.py

    l1/
      for-agent-layerinfo-l1.md           # 중해상도 (l1)
      order/
        order.py
        for-agent-moduleinfo.md           # 고해상도 (order)
        __init__.py
      pair/
        pair.py
        for-agent-moduleinfo.md           # 고해상도 (pair)
        __init__.py

    l3/
      for-agent-layerinfo-l3.md           # 중해상도 (l3)
      portfolio/
        l0/
          tick_snapshot.py
          __init__.py
        l1/
          file_backend.py
          __init__.py
        l2/
          storage_l1.py
          __init__.py
        l3/
          portfolio.py
          __init__.py
        for-agent-moduleinfo.md           # 고해상도 (portfolio)
        __init__.py

  tests/
    l0/
      candle/
        test_candle.py
      token/
        test_token.py

    l1/
      order/
        test_order.py
      pair/
        test_pair.py

    l3/
      portfolio/
        test_portfolio.py
        test_integration.py

## 파일 배치 규칙

### 소스 코드
- 위치: `src/fishfactory/ln/modulename/`
- 메인 파일: `modulename.py` (또는 중첩 Ln)
- 문서: `for-agent-moduleinfo.md`
- Export: `__init__.py`

### 테스트
- 위치: `tests/ln/modulename/`
- 명명: `test_*.py`
- 구조: 소스 미러

### 문서

#### 저해상도 (전체 시스템)
- 위치: `src/fishfactory/for-agent-layerinfo.md`
- 템플릿: `for-agent-layerinfo-template.md` 참조

#### 중해상도 (레벨별 API)
- 위치: `src/fishfactory/ln/for-agent-layerinfo-ln.md`
- 템플릿: `for-agent-layerinfo-ln-template.md` 참조

#### 고해상도 (모듈별 상세)
- 위치: `src/fishfactory/ln/modulename/for-agent-moduleinfo.md`
- 템플릿: `for-agent-moduleinfo-template.md` 참조
