#!/bin/bash
export OLLAMA_HOST=0.0.0.0
export OLLAMA_ORIGINS="*"
export OLLAMA_MODELS_PATH=$HOME/.ollama/models

# 确保模型目录存在
mkdir -p $HOME/.ollama/models

# 启动 Ollama
ollama serve