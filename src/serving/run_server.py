#!/usr/bin/env python3
"""
AnimeGAN2 Preprocess Server 실행 스크립트
uv 환경에서 실행됩니다.
"""

import uvicorn
from app import app

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8004,
        reload=True,  # 개발 모드에서 자동 재시작
        log_level="info",
        access_log=True,
    )
