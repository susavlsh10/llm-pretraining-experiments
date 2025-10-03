# llm-pretraining
LLM Pre-training Experiments 

TODO: 

+ Find ideal batch size, sequence_length, total_batch_size, num_iterations to maximize throughput; 24 A100 GPU hours training job.
+ Find ideal model size 

Single GPU Training 

```bash

python train_llama3.py --model llama-pro-mini --input_bin "/home/grads/s/sls7161/nvme/datasets/fineweb10B/fineweb_train_*.bin" --input_val_bin "/home/grads/s/sls7161/nvme/datasets/fineweb10B/fineweb_val_*.bin" --batch_size 4 --sequence_length 1024 --total_batch_size 8192 --num_iterations 40 --output_dir /home/grads/s/sls7161/nvme/llm_training --resume_from_checkpoint /home/grads/s/sls7161/nvme/llm_training/checkpoint_20.pt

python train_llama3.py --model llama-pro-mini --input_bin "/home/grads/s/sls7161/nvme/datasets/fineweb10B/fineweb_train_*.bin" --input_val_bin "/home/grads/s/sls7161/nvme/datasets/fineweb10B/fineweb_val_*.bin" --batch_size 16 --sequence_length 1024 --total_batch_size 524288 --num_iterations 40 --output_dir /home/grads/s/sls7161/nvme/llm_training --resume_from_checkpoint /home/grads/s/sls7161/nvme/llm_training/checkpoint_20.pt

```

Multi GPU Training
```bash
torchrun --nproc_per_node=2 train_llama3.py --model llama-pro-mini --input_bin "/home/grads/s/sls7161/nvme/datasets/fineweb10B/fineweb_train_*.bin" --input_val_bin "/home/grads/s/sls7161/nvme/datasets/fineweb10B/fineweb_val_*.bin" --batch_size 4 --sequence_length 1024 --total_batch_size 8192 --num_iterations 40 --output_dir /home/grads/s/sls7161/nvme/llm_training
--resume_from_checkpoint /home/grads/s/sls7161/nvme/llm_training/checkpoint_20.pt

```

A100 Runs 

Single GPU

```bash
python train_llama3.py --model llama-pro-mini --input_bin "/pscratch/sd/s/susav/fineweb10B/fineweb_train_*.bin" --input_val_bin "/pscratch/sd/s/susav/fineweb10B/fineweb_val_*.bin" --batch_size 8 --sequence_length 1024 --total_batch_size 16384 --num_iterations 20 --output_dir /pscratch/sd/s/susav/llm_training 
--resume_from_checkpoint /pscratch/sd/s/susav/llm_training/checkpoint_20.pt

```

Multi GPU

```bash
torchrun --nproc_per_node=4 train_llama3.py --model llama-pro-mini --input_bin "/pscratch/sd/s/susav/fineweb10B/fineweb_train_*.bin" --input_val_bin "/pscratch/sd/s/susav/fineweb10B/fineweb_val_*.bin" --batch_size 16 --sequence_length 1024 --total_batch_size 524288 --num_iterations 20 --output_dir /pscratch/sd/s/susav/llm_training --export_hf 0 --run_id 0 --resume_from_checkpoint /pscratch/sd/s/susav/llm_training/checkpoint_latest

```
