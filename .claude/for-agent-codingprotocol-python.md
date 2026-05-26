# Python Coding Protocol

Python 코딩 스타일 및 품질 규칙.

## 기본 규칙

1. **1파일 1클래스**
   - 하나의 파일은 하나의 클래스만 포함
   - 파일명과 클래스명은 CamelCase

2. **단일 책임 원칙**
   - 각 클래스는 하나의 책임만

3. **문서화**
   - 코드 내: 한줄 주석(`#`)만 사용
   - Docstring, 멀티라인 주석 사용 금지
   - 상세 설명: `for-agent-moduleinfo.md`에 작성

## 코드 예시

```python
# 주문 처리 서비스
class OrderService:
    def __init__(self, db_url: str):
        self.db_url = db_url

    def create_order(self, user_id: int):
        # 검증
        if not user_id:
            raise ValueError("Invalid user_id")

        # 주문 생성
        return {"order_id": 123}
```

## 클래스 작성 규칙

@.claude/for-agent-codingprotocol-makeclass.md 참조

## 로깅

`loguru` 사용.

```python
from loguru import logger

class OrderService:
    def __init__(self, db_url: str):
        logger.info(f"OrderService 초기화: db_url={db_url}")
        self.db_url = db_url

    def create_order(self, user_id: int):
        logger.info(f"create_order 시작: user_id={user_id}")
        # 중요 단계는 DEBUG
        logger.debug(f"DB 연결 확인")
        return {"order_id": 123}
```
