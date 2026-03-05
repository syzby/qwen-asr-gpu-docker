# Qwen ASR GPU Docker Image
# 基于 NVIDIA CUDA + Python 镜像

FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV MODEL_DIR=/models

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# 设置Python 3.11为默认
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# 创建工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 创建模型目录
RUN mkdir -p ${MODEL_DIR}

# 暴露端口 (如果需要HTTP服务)
EXPOSE 8000

# 下载模型 (可选，默认不下载，运行时按需下载)
# 可以通过环境变量指定模型
ENV MODEL_NAME="paraformer-zh"

# 启动命令
CMD ["python3", "app.py", "--use_gpu"]
