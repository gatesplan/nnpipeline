# Research Protocol

연구 기록물의 양식과 규칙.

## 실험 폴더

### 폴더명
- 형식: `NNN-yymmdd-CamelCaseTitle` (순번-날짜-제목)
- agent 자율실험: `NNN-agent-CamelCaseTitle` (날짜 대신 `agent`)
- 순번은 전체 프로젝트에서 누적

### 실험 스크립트
- 파일명: `00-title.py`, `01-title.py`, ... (순번-기능설명)
- 기능이 무엇인지 상세히 기술 (예: `02-init-strategy-compare.py`)
- 실험 로직에 집중. 인프라 코드는 `_*.py`로 분리

### 재사용 모듈
- 파일명: `_name.py` (underscore prefix, 기능별 분리)
- 예: `_fim.py`, `_data.py`, `_model.py`, `_split.py`, `_config.py`
- `_common.py`에는 상수, 경로, 단순 유틸만 배치
- 규모가 큰 함수나 핵심 로직은 별도 파일로 분리

### 결과 폴더
- 폴더명: `00-result/`, `01-result/`, ... (스크립트와 동일 순번)
- flat 구조: 하위 폴더 없이 직접 배치
- 파일명: `{dataset}_{content}.{ext}` (예: `digits_spectrum.png`)
- 시각화(png) = 연구자 확인용, 수치(csv/txt) = agent 분석용
- 동일 내용에 대해 png + csv/txt 병행 생성

## 실험.md

각 실험 폴더에 작성. 5개 섹션 고정.
- 양식: `.claude/for-agent-experiment-template.md`

### agent 통합 폴더의 서브실험
- 서브실험이 많은 경우 `NNN-실험.md`로 분리 가능 (예: `010-실험.md`)
- 루트 `실험.md`에 전체 흐름과 의존성 정리

## 논문 관리

### Papers/
- 원본 PDF 보관
- 파일명: `Paper - 연도 - 제목.pdf`

### Notes/
- 논문 노트
- 파일명: `Notes - 연도 - 제목.md`

### 논문 노트 양식
- 양식: `.claude/for-agent-papernotes-template.md`

### 알고리즘 추출 문서
- 구현에 직접 필요한 알고리즘을 논문에서 추출
- 파일명: `제목-알고리즘-추출.md` 또는 `제목-Implementation-Guide.md`
- Notes/ 폴더에 배치

## 공통 실험 환경

[프로젝트별 데이터셋, 학습 조건, 기타 공통 환경을 여기에 작성]

## 코드 작성 원칙

- 시간복잡도와 로직 효율 중시
- 불필요한 반복, 중복 계산, 비효율적 자료구조 사용 금지
- 출력에서 이모지/유니코드 특수문자 사용 금지 (ASCII + 한글만)
