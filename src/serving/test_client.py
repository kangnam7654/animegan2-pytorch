#!/usr/bin/env python3
"""
AnimeGAN2 Preprocess API 테스트 클라이언트
uv 환경에서 실행: uv run python test_client.py
"""

import base64
from pathlib import Path

import requests

# 서버 URL
SERVER_URL = "http://localhost:8000"


def test_health_check():
    """서버 상태 확인"""
    try:
        response = requests.get(f"{SERVER_URL}/health")
        print(f"Health Check: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False


def test_file_upload(image_path):
    """파일 업로드 테스트"""
    try:
        with open(image_path, "rb") as f:
            files = {"file": f}
            data = {
                "target_width": 256,
                "target_height": 256,
                "return_format": "base64",
            }
            response = requests.post(f"{SERVER_URL}/preprocess", files=files, data=data)

        if response.status_code == 200:
            result = response.json()
            print("File upload successful!")
            print(f"Shape: {result['shape']}")
            print(f"Data type: {result['dtype']}")
            print(f"Data preview: {result['data'][:50]}...")
            return True
        else:
            print(f"File upload failed: {response.text}")
            return False

    except Exception as e:
        print(f"File upload test failed: {e}")
        return False


def test_base64_input(image_path):
    """Base64 입력 테스트"""
    try:
        # 이미지를 base64로 인코딩
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        payload = {
            "image_data": {"image": image_data},
            "target_width": 256,
            "target_height": 256,
        }

        response = requests.post(f"{SERVER_URL}/preprocess/base64", json=payload)

        if response.status_code == 200:
            result = response.json()
            print("Base64 input successful!")
            print(f"Shape: {result['shape']}")
            print(f"Data type: {result['dtype']}")
            print(f"Data preview: {result['data'][:50]}...")
            return True
        else:
            print(f"Base64 input failed: {response.text}")
            return False

    except Exception as e:
        print(f"Base64 input test failed: {e}")
        return False


def main():
    print("AnimeGAN2 Preprocess API 테스트 시작")
    print("=" * 50)

    # 건강 상태 확인
    print("\n1. 서버 상태 확인")
    if not test_health_check():
        print("서버가 실행되지 않았습니다. 먼저 서버를 시작하세요.")
        return

    # 테스트 이미지 경로 찾기
    test_image_paths = [
        "/mnt/d/datasets/anime_dataset/female_000000.png",
        "/home/kangnam/projects/animegan2/samples/inputs",
        "test_image.jpg",
    ]

    test_image = None
    for path in test_image_paths:
        if Path(path).exists():
            if Path(path).is_file():
                test_image = path
                break
            elif Path(path).is_dir():
                # 디렉토리에서 첫 번째 이미지 파일 찾기
                for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                    images = list(Path(path).glob(ext))
                    if images:
                        test_image = str(images[0])
                        break
                if test_image:
                    break

    if not test_image:
        print("\n테스트할 이미지 파일을 찾을 수 없습니다.")
        print("테스트 이미지 경로를 확인하세요.")
        return

    print(f"\n테스트 이미지: {test_image}")

    # 파일 업로드 테스트
    print("\n2. 파일 업로드 테스트")
    test_file_upload(test_image)

    # Base64 입력 테스트
    print("\n3. Base64 입력 테스트")
    test_base64_input(test_image)

    print("\n테스트 완료!")


if __name__ == "__main__":
    main()
