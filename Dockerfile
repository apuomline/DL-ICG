FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

# 设置工作目录
WORKDIR /DL-ICG

# 先复制requirements.txt并安装依赖（利用Docker缓存）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 然后复制其余文件
COPY . .

# 设置脚本文件为可执行
RUN chmod +x run.sh pipeline.sh

# 环境变量设置
ENV PYTHONPATH=/DL-ICG
ENV PYTHONUNBUFFERED=1

# 清理缓存
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# 验证环境
RUN python -c "import torch; print(f'环境验证: PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# 设置启动命令 - 运行run.sh
CMD ["./run.sh"]