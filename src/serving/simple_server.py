#!/usr/bin/env python3
"""
간단한 서버 테스트
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app

if __name__ == "__main__":
    import uvicorn

    print("AnimeGAN2 Preprocess API 서버를 시작합니다...")
    print("서버 URL: http://localhost:8000")
    print("API 문서: http://localhost:8000/docs")
    print("서버를 중지하려면 Ctrl+C를 누르세요.")

    try:
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=True, log_level="info")
    except KeyboardInterrupt:
        print("\n서버가 중지되었습니다.")
