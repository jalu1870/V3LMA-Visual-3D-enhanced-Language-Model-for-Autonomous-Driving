# Use the official PyTorch image with CUDA support
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# FROM pytorch/pytorch:2.5.0-cuda12.1-cudnn9-devel

# ENV CUDA_HOME=/usr/local/cuda

# Set the working directory
WORKDIR /mnt

# Install any necessary packages (e.g., for file manipulation)
RUN apt-get update && apt-get install -y \
    # Add any required packages here
    ffmpeg libsm6 libxext6 git unzip docker.io\
    bash \
    && rm -rf /var/lib/apt/lists/*

# Optionally, clean up to reduce image size
RUN rm -rf /var/lib/apt/lists/*
# RUN apt-get install libcairo2-dev pkg-config python3-dev
COPY qwen/requirements_qwen.txt .
# RUN apt-get update && apt-get install  -y
RUN pip install --no-cache-dir -r requirements_qwen.txt
RUN pip install flash-attn==2.6.1 --no-build-isolation
RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
# RUN sh Mask2Former/mask2former/modeling/pixel_decoder/ops/make.sh

RUN pip install natsort

RUN pip install easydict
RUN pip install nltk
RUN pip install rouge_score
RUN pip install tf-keras
RUN pip install openpyxl
RUN pip install supervision

# Command to run your application (update this according to your app)
CMD ["bash"]
# python train.py