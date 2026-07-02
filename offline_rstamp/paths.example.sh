#!/usr/bin/env bash
# Copy this file to paths.local.sh and edit it on the offline server.

# Root where datasets were downloaded by MLLM-SEG-data/download.sh.
export MLLM_SEG_DATA_ROOT="/data/MLLM-SEG"

# Clean experiment workspace. Scripts will create data/code/models/outputs/logs here.
export MLLM_SEG_WORKSPACE="/work/MLLM-SEG-exp"

# Upstream code directories inside the workspace.
export STAMP_CODE_DIR="${MLLM_SEG_WORKSPACE}/code/STAMP"
export RSTAMP_CODE_DIR="${MLLM_SEG_WORKSPACE}/code/R-STAMP"

# Model/checkpoint root copied manually to the offline server.
export MODEL_ROOT="${MLLM_SEG_WORKSPACE}/models"

# Example model paths. Edit to match your local files.
export STAMP_MODEL_PATH="${MODEL_ROOT}/STAMP-7B-lora"
export STAMP_2B_MODEL_PATH="${MODEL_ROOT}/STAMP-2B-uni"
export SAM_CKPT="${MODEL_ROOT}/sam_vit_h_4b8939.pth"

# Output roots.
export OUTPUT_ROOT="${MLLM_SEG_WORKSPACE}/outputs"
export LOG_ROOT="${MLLM_SEG_WORKSPACE}/logs"

# GPU setup for 2×80GB.
export CUDA_VISIBLE_DEVICES="0,1"
export NPROC_PER_NODE="2"

# Optional Hugging Face cache already copied to the offline server.
export HF_HOME="${MODEL_ROOT}/hf_home"
export TRANSFORMERS_OFFLINE="1"
export HF_DATASETS_OFFLINE="1"

