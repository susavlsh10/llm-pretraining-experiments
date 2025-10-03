#!/bin/bash
export TRITON_CACHE_DIR=/pscratch/sd/s/susav/.triton_cache
export TRANSFORMERS_CACHE=/pscratch/sd/s/susav/transformers_cache
export MPLCONFIGDIR=/pscratch/sd/s/susav/matplotlib_config
export HF_HOME=/pscratch/sd/s/susav/huggingface/cache
export HF_DATASETS_TRUST_REMOTE_CODE=1
export HOME=/global/homes/s/susav
export HF_TOKEN=HF_TOKEN_PLACEHOLDER
export PATH="$HOME/.local/bin:$PATH"
# export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

if [ -z "$MASTER_ADDR" ]; then
    export MASTER_ADDR=$(hostname)
fi

export MASTER_PORT=29501

cd /global/homes/s/susav/workspace/llm-pretraining-experiments

torchrun --nnodes=4 --nproc_per_node=4 --rdzv-backend=c10d --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT train_llama3.py --model llama-pro-mini --input_bin '/pscratch/sd/s/susav/fineweb10B/fineweb_train_*.bin' --input_val_bin '/pscratch/sd/s/susav/fineweb10B/fineweb_val_*.bin' --batch_size 16 --sequence_length 1024 --total_batch_size 524288 --num_iterations 16000 --output_dir /pscratch/sd/s/susav/llm_training --export_hf 0 --run_id training_interactive_2

# torchrun --nnodes=4 --nproc_per_node=4 \
#     --rdzv-backend=c10d \
#     --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
#     train_llama3.py \
#     --model llama-pro-mini \
#     --input_bin '/pscratch/sd/s/susav/fineweb10B/fineweb_train_*.bin' \
#     --input_val_bin '/pscratch/sd/s/susav/fineweb10B/fineweb_val_*.bin' \
#     --batch_size 16 \
#     --sequence_length 1024 \
#     --total_batch_size 524288 \
#     --num_iterations 16000 \
#     --output_dir /pscratch/sd/s/susav/llm_training \
#     --export_hf 0 \
#     --run_id training_interactive_1