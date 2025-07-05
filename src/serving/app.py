import base64
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from serving.services.preprocess import preprocess_image

app = FastAPI(
    title="AnimeGAN2 Preprocess API",
    description="이미지 전처리를 위한 FastAPI 서버",
    version="1.0.0",
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "AnimeGAN2 Preprocess API", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/preprocess")
async def preprocess_image_endpoint(
    file: UploadFile = File(...),
    target_width: Optional[int] = 256,
    target_height: Optional[int] = 256,
    return_format: Optional[str] = "base64",  # "base64" or "numpy"
):
    """
    이미지 파일을 업로드하고 전처리된 결과를 반환합니다.

    Args:
        file: 업로드할 이미지 파일
        target_width: 목표 너비 (기본값: 256)
        target_height: 목표 높이 (기본값: 256)
        return_format: 반환 형식 ("base64" 또는 "numpy")

    Returns:
        전처리된 이미지 데이터
    """
    try:
        # 파일 읽기
        contents = await file.read()

        # 이미지 전처리
        processed_image = preprocess_image(
            contents, target_size=(target_width, target_height)
        )

        if return_format == "base64":
            # numpy 배열을 base64로 인코딩
            image_bytes = processed_image.tobytes()
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")

            return {
                "success": True,
                "data": image_base64,
                "shape": processed_image.shape,
                "dtype": str(processed_image.dtype),
                "format": "base64",
            }

        elif return_format == "numpy":
            return {
                "success": True,
                "data": processed_image.tolist(),
                "shape": processed_image.shape,
                "dtype": str(processed_image.dtype),
                "format": "numpy",
            }

        else:
            raise HTTPException(
                status_code=400, detail="return_format must be 'base64' or 'numpy'"
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/preprocess/base64")
async def preprocess_base64_endpoint(
    image_data: dict,
    target_width: Optional[int] = 256,
    target_height: Optional[int] = 256,
):
    """
    Base64로 인코딩된 이미지를 받아 전처리합니다.

    Args:
        image_data: {"image": "base64_encoded_image_data"}
        target_width: 목표 너비
        target_height: 목표 높이

    Returns:
        전처리된 이미지 데이터
    """
    try:
        if "image" not in image_data:
            raise HTTPException(
                status_code=400, detail="Missing 'image' field in request body"
            )

        # Base64 디코딩
        image_base64 = image_data["image"]
        if image_base64.startswith("data:image"):
            # data:image/jpeg;base64, 형태의 prefix 제거
            image_base64 = image_base64.split(",")[1]

        image_bytes = base64.b64decode(image_base64)

        # 이미지 전처리
        processed_image = preprocess_image(
            image_bytes, target_size=(target_width, target_height)
        )

        # numpy 배열을 base64로 인코딩
        image_bytes = processed_image.tobytes()
        result_base64 = base64.b64encode(image_bytes).decode("utf-8")

        return {
            "success": True,
            "data": result_base64,
            "shape": processed_image.shape,
            "dtype": str(processed_image.dtype),
            "format": "base64",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8004)
