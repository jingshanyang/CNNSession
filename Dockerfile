# 使用完整的Python 3.10.13镜像
FROM python:3.10.13

# 设置工作目录
WORKDIR /workspace

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 从GitHub克隆LLaMA Factory
RUN git clone https://github.com/hiyouga/LLaMA-Factory.git /workspace/llama_factory

# 安装Python依赖，先安装setuptools
RUN pip install --upgrade pip setuptools wheel && \
    cd /workspace/llama_factory && pip install --no-cache-dir -e . && \
    pip install --no-cache-dir flask numpy pillow torchvision mlflow

# 设置环境变量
ENV PYTHONPATH=/workspace:/workspace/llama_factory
ENV HF_HOME=/workspace/huggingface

# 禁用CUDA，强制使用CPU
ENV CUDA_VISIBLE_DEVICES=""
ENV FORCE_CPU=1
ENV USE_CUDA=False

# 明确使用fp32，禁用bf16，适应CPU环境
ENV FP16=false
ENV BF16=false
ENV USE_FP16=false
ENV USE_BF16=false
ENV DEFAULT_PRECISION=fp32

# 暴露端口
EXPOSE 8000
EXPOSE 7860
EXPOSE 5000

# 设置默认命令
CMD ["bash", "-c", "echo '服务启动完成！' && tail -f /dev/null"]