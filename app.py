#!/usr/bin/env python3
"""
Qwen ASR API Server
使用 FastAPI 提供语音识别 HTTP 服务
"""

import os
import argparse
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import torch

from funasr import AutoModel

app = FastAPI(title="Qwen ASR API", version="1.0.0")

# 全局模型实例
model = None


def get_device():
    """获取设备类型"""
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


@app.on_event("startup")
async def startup_event():
    """启动时加载模型"""
    global model
    
    model_name = os.getenv("MODEL_NAME", "paraformer-zh")
    use_gpu = os.getenv("USE_GPU", "true").lower() == "true"
    device = "cuda:0" if (use_gpu and torch.cuda.is_available()) else "cpu"
    
    print(f"Loading model: {model_name} on {device}...")
    
    model = AutoModel(
        model=model_name,
        model_revision="v2.0.4",
        device=device
    )
    
    print(f"Model loaded successfully!")


@app.get("/")
async def root():
    """健康检查"""
    return {
        "status": "ok",
        "model": os.getenv("MODEL_NAME", "paraformer-zh"),
        "device": get_device(),
        "gpu_available": torch.cuda.is_available()
    }


@app.post("/asr")
async def asr_recognize(
    file: UploadFile = File(...),
    hotword: str = Form("")
):
    """语音识别接口"""
    global model
    
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Model not loaded"}
        )
    
    # 保存上传的音频文件
    temp_file = f"/tmp/{file.filename}"
    with open(temp_file, "wb") as f:
        content = await file.read()
        f.write(content)
    
    try:
        # 执行识别
        result = model.generate(
            input=temp_file,
            batch_size_s=300,
            hotword=hotword
        )
        
        return {
            "text": result[0]["text"],
            "timestamp": result[0].get("timestamp", [])
        }
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
    
    finally:
        # 清理临时文件
        if os.path.exists(temp_file):
            os.remove(temp_file)


if __name__ == "__main__":
    import uvicorn
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--use_gpu", action="store_true", default=True, help="Use GPU")
    args = parser.parse_args()
    
    uvicorn.run(app, host=args.host, port=args.port)
