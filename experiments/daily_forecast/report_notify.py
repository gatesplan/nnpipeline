"""Discord 웹훅 보고 유틸리티 (daily_forecast 용).

- 웹훅 URL: 환경변수 DISCORD_API 우선, 없으면 stock-analyzer 프로젝트 .env 에서 로드
- 전송 인코딩: JSON 을 ensure_ascii=False 로 직렬화한 뒤 UTF-8 바이트로 명시 전송
  (기본 json= 인자는 ASCII 이스케이프를 거치므로 한글 포함 시 명시 인코딩이 안전)
- HTML 등 파일 첨부는 multipart 업로드, 바이트는 UTF-8 로 읽음
"""

import json
import os
from pathlib import Path

import requests

_STOCK_ANALYZER_ENV = Path("C:/Projects/stock-analyzer/.env")
_MAX_LEN = 1900


def _load_webhook_url():
    url = os.getenv("DISCORD_API")
    if url:
        return url
    if _STOCK_ANALYZER_ENV.exists():
        for line in _STOCK_ANALYZER_ENV.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("DISCORD_API="):
                return line.split("=", 1)[1].strip()
    return None


_WEBHOOK_URL = _load_webhook_url()


def notify(message: str, prefix: str = "[daily_forecast]") -> bool:
    """텍스트 메시지 전송. 2000자 제한 대응 분할 (최대 4조각)."""
    if _WEBHOOK_URL is None:
        print(f"[report_notify] DISCORD_API 미설정 — 콘솔 출력:\n{message}")
        return False

    content = f"{prefix}\n{message}" if prefix else message
    ok = True
    for chunk in _chunk(content, _MAX_LEN)[:4]:
        payload = json.dumps({"content": chunk}, ensure_ascii=False).encode("utf-8")
        try:
            resp = requests.post(
                _WEBHOOK_URL, data=payload,
                headers={"Content-Type": "application/json"}, timeout=10,
            )
            if resp.status_code not in (200, 204):
                print(f"[report_notify] HTTP {resp.status_code}: {resp.text}")
                ok = False
        except Exception as exc:
            print(f"[report_notify] 전송 실패: {exc}")
            ok = False
    return ok


def send_file(path, message: str = "", prefix: str = "[daily_forecast]") -> bool:
    """파일 첨부 전송. 텍스트 파일은 UTF-8 바이트 그대로 첨부."""
    if _WEBHOOK_URL is None:
        print(f"[report_notify] DISCORD_API 미설정 — 파일 전송 생략: {path}")
        return False

    path = Path(path)
    content = f"{prefix}\n{message}" if prefix else message
    try:
        with path.open("rb") as f:
            resp = requests.post(
                _WEBHOOK_URL,
                data={"payload_json": json.dumps({"content": content}, ensure_ascii=False)},
                files={"file": (path.name, f)},
                timeout=30,
            )
        if resp.status_code not in (200, 204):
            print(f"[report_notify] HTTP {resp.status_code}: {resp.text}")
            return False
        return True
    except Exception as exc:
        print(f"[report_notify] 파일 전송 실패: {exc}")
        return False


def _chunk(text: str, size: int):
    if len(text) <= size:
        return [text]
    out, cur, cur_len = [], [], 0
    for line in text.splitlines(keepends=True):
        if cur_len + len(line) > size:
            out.append("".join(cur))
            cur, cur_len = [line], len(line)
        else:
            cur.append(line)
            cur_len += len(line)
    if cur:
        out.append("".join(cur))
    return out
