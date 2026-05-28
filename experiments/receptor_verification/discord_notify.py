"""Discord 웹훅 보고 유틸리티.

stock-analyzer 프로젝트의 .env 파일에서 DISCORD_API를 읽거나,
환경변수 DISCORD_API를 우선 사용.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import requests

_STOCK_ANALYZER_ENV = Path("C:/projects/stock-analyzer/.env")


def _load_webhook_url() -> Optional[str]:
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


def notify(message: str, *, prefix: str = "[Receptor Verification]") -> bool:
    """Discord 채널에 메시지 전송. 2000자 초과 시 자동 분할 (최대 4조각)."""
    if _WEBHOOK_URL is None:
        print(f"[discord_notify] DISCORD_API 미설정 — 콘솔 출력만:\n{message}")
        return False

    content = f"{prefix}\n{message}" if prefix else message
    chunks = _chunk(content, 1900)
    ok = True
    for chunk in chunks[:4]:
        try:
            resp = requests.post(_WEBHOOK_URL, json={"content": chunk}, timeout=10)
            if resp.status_code not in (200, 204):
                print(f"[discord_notify] HTTP {resp.status_code}: {resp.text}")
                ok = False
        except Exception as exc:
            print(f"[discord_notify] 전송 실패: {exc}")
            ok = False
    return ok


def _chunk(text: str, size: int) -> list[str]:
    if len(text) <= size:
        return [text]
    out = []
    cur = []
    cur_len = 0
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


if __name__ == "__main__":
    notify("Discord 연결 테스트.", prefix="[Setup]")
