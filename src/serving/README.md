# AnimeGAN2 Preprocess API

FastAPI를 사용한 AnimeGAN2 이미지 전처리 서버입니다.

## 기능

- 이미지 파일 업로드 및 전처리
- Base64 인코딩된 이미지 처리
- 커스텀 이미지 크기 조정
- 다양한 출력 형식 지원 (base64, numpy)

## 설치 및 실행

### 1. uv를 사용한 의존성 설치

```bash
# uv 동기화 (가상환경 생성 및 의존성 설치)
uv sync
```

### 2. 서버 실행

#### 방법 1: uv run 사용
```bash
uv run python app.py
```

#### 방법 2: 실행 스크립트 사용
```bash
uv run python run_server.py
```

#### 방법 3: 쉘 스크립트 사용
```bash
chmod +x start_server.sh
./start_server.sh
```

서버가 실행되면 `http://localhost:8000`에서 접근할 수 있습니다.

## API 엔드포인트

### 기본 정보

- **GET** `/` - API 기본 정보
- **GET** `/health` - 서버 상태 확인
- **GET** `/docs` - Swagger UI (자동 생성)
- **GET** `/redoc` - ReDoc (자동 생성)

### 이미지 전처리

#### 1. 파일 업로드
- **POST** `/preprocess`
- **Parameters:**
  - `file`: 업로드할 이미지 파일
  - `target_width`: 목표 너비 (기본값: 256)
  - `target_height`: 목표 높이 (기본값: 256)
  - `return_format`: 반환 형식 ("base64" 또는 "numpy")

#### 2. Base64 입력
- **POST** `/preprocess/base64`
- **Body:**
  ```json
  {
    "image_data": {
      "image": "base64_encoded_image_data"
    },
    "target_width": 256,
    "target_height": 256
  }
  ```

## 사용 예제

### Python 클라이언트 (uv 환경)

```python
import requests
import base64

# 파일 업로드
with open('image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/preprocess', files=files)
    result = response.json()
    print(result)

# Base64 입력
with open('image.jpg', 'rb') as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

payload = {
    'image_data': {'image': image_data}
}
response = requests.post('http://localhost:8000/preprocess/base64', json=payload)
result = response.json()
print(result)
```

### cURL 사용

```bash
# 파일 업로드
curl -X POST "http://localhost:8000/preprocess" \
     -F "file=@image.jpg" \
     -F "target_width=256" \
     -F "target_height=256"

# Base64 입력
curl -X POST "http://localhost:8000/preprocess/base64" \
     -H "Content-Type: application/json" \
     -d '{"image_data": {"image": "base64_encoded_data"}}'
```

## 테스트

테스트 클라이언트를 사용하여 API를 테스트할 수 있습니다:

```bash
uv run python test_client.py
```

## 개발 환경 설정

### uv 설치
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 개발 의존성 설치
```bash
uv sync --dev
```

## Docker 사용

### 이미지 빌드
```bash
docker build -t animegan2-preprocess .
```

### 컨테이너 실행
```bash
docker run -p 8000:8000 animegan2-preprocess
```

### Docker Compose 사용
```bash
docker-compose up -d
```

## 응답 형식

성공 응답:
```json
{
  "success": true,
  "data": "base64_encoded_processed_image",
  "shape": [1, 3, 256, 256],
  "dtype": "float32",
  "format": "base64"
}
```

오류 응답:
```json
{
  "detail": "Error message"
}
```

## 개발 환경

- Python 3.11+
- uv (패키지 관리자)
- FastAPI
- OpenCV
- NumPy
- Uvicorn

## 라이선스

MIT License
