"""
Reference code for LLaMA-3.1 training and inference.
Will save the model weights into files, to be read from C as initialization.

This code differs from GPT-2 very slightly, there are three main differences:
1) RoPE: LLaMA uses a different positional encoding scheme called Relative Positional Encoding (RoPE).
2) GQA: Grouped Query Attention (GQA) is used to reduce the number of attention heads.
3) SwiGLU: Swish-Gated Linear Unit (SwiGLU) is used as the activation function in the MLP.

References:
# 1) https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/api/tokenizer.py
# 2) https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/api/model.py
# 3) https://github.com/meta-llama/llama3/blob/11817d47e1ba7a4959b025eb1ca308572e0e3963/llama/generation.py

Example launches to only benchmark the speed of bfloat16 compiled GPU training:

Single GPU training 
python train_llama3.py --input_bin /path/to/fineweb_train_*.bin --input_val_bin /path/to/fineweb_val.bin --output_dir ./out_llama3 --model meta-llama/Meta-Llama-3.1-8B --batch_size 4 --sequence_length 64 --total_batch_size 256 --num_iterations 1000 --learning_rate 1e-5 --warmup_iters 1000 --val_loss_every 100 --val_max_steps 20 --sample_every 200 --overfit_single_batch 0 --tensorcores 1 --dtype bfloat16 --device cuda:0 --compile 1 --zero_stage 0 --write_tensors 0 --checkpoint_every 500 --use_hf 0 --ckpt_dir /path/to/llama3_checkpoint/ --tokenizer_path /path/to/llama3_tokenizer/ --export_hf 1 --export_hf_dir ./hf_llama3_model/ --push_to_hub 0 --test_exported_model 1 --custom_n_layer 12 --custom_n_embd 1024

torchrun --nproc_per_node=2 train_llama3.py --input_bin /path/to/fineweb_train_*.bin --input_val_bin /path/to/fineweb_val.bin --output_dir ./out_llama3 --model meta-llama/Meta-Llama-3.1-8B --batch_size 2 --sequence_length 64 --total_batch_size 256 --num_iterations 1000 --learning_rate 1e-5 --warmup_iters 1000 --val_loss_every 100 --val_max_steps 20 --sample_every 200 --overfit_single_batch 0 --tensorcores 1 --dtype bfloat16 --device cuda --compile 1 --zero_stage 0 --write_tensors 0 --checkpoint_every 500 --use_hf 0 --ckpt_dir /path/to/llama3_checkpoint/ --tokenizer_path /path/to/llama3_tokenizer/ --export_hf 1 --export_hf_dir ./hf_llama3_model/ --push_to_hub 0 --test_exported_model 1 --custom_n_layer 12 --custom_n_embd 1024

"""

import argparse
import torch
from models.llama import (LLaMA, LlamaConfig,
                         DistributedShardedDataLoader, write_model, write_state,
                         export_to_huggingface,
                         test_exported_model, save_checkpoint, load_checkpoint)

from models.llama import (detect_gpu_model, calculate_model_flops,
                          get_gpu_peak_flops, calculate_mfu_from_throughput, 
                          calculate_model_parameters, print_model_info)

from utils.llama_utils import setup_logging, print0

from typing import List

import os
import time
import math
import numpy as np
import logging
import sys
from datetime import datetime
from contextlib import nullcontext

from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch._inductor.config as config



def save_to_huggingface(args, raw_model, master_process):
    if args.export_hf and master_process:
        print0("=" * 80)
        print0("Exporting model to HuggingFace format...")
        
        # Gather custom config overrides for export
        custom_config = {}
        if args.custom_n_layer is not None:
            custom_config['n_layer'] = args.custom_n_layer
        if args.custom_n_embd is not None:
            custom_config['n_embd'] = args.custom_n_embd
        if args.custom_n_head is not None:
            custom_config['n_head'] = args.custom_n_head
        if args.custom_n_kv_head is not None:
            custom_config['n_kv_head'] = args.custom_n_kv_head
        if args.custom_vocab_size is not None:
            custom_config['vocab_size'] = args.custom_vocab_size
        if args.custom_block_size is not None:
            custom_config['block_size'] = args.custom_block_size
        
        try:
            hf_model, export_path = export_to_huggingface(
                model=raw_model,
                tokenizer=raw_model.tokenizer if hasattr(raw_model, 'tokenizer') else None,
                output_dir=args.export_hf_dir,
                model_name=args.hf_model_name if args.hf_model_name else None,
                push_to_hub=bool(args.push_to_hub),
                private_repo=bool(args.private_repo),
                custom_config=custom_config if custom_config else None
            )
            
            print0(f"Model exported successfully to: {export_path}")
            
            # Test the exported model if requested
            if args.test_exported_model:
                print0("Testing exported model...")
                test_exported_model(export_path)
                
        except Exception as e:
            print0(f"Error during export: {e}")
            print0("Make sure transformers library is installed: pip install transformers")

def setup_ddp_and_device(args):
    """
    Set up distributed data parallel and determine device.
    
    Returns:
        tuple: (ddp, ddp_rank, ddp_local_rank, ddp_world_size, master_process, device, device_type, zero_stage)
    """
    ddp = int(os.environ.get('RANK', -1)) != -1
    
    if ddp:
        assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        zero_stage = args.zero_stage
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        zero_stage = 0
        ddp_world_size = 1
        master_process = True
        
        # Auto-detect device
        if args.device:
            device = args.device
        else:
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
    
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    assert device_type in {'cuda'}, "GPU required to run LLaMA 3"
    print(f"using device: {device}")
    
    # update output_dir to include run_id
    if args.output_dir:
        args.output_dir = os.path.join(args.output_dir, f"run_{args.run_id}")
        if master_process:
            os.makedirs(args.output_dir, exist_ok=True)
    
    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, master_process, device, device_type, zero_stage

def initialize_model(args):
    """
    Initialize the LLaMA model based on configuration.
    
    Args:
        args: parsed command line arguments
        
    Returns:
        LLaMA model
    """
    if args.use_hf:
        # Apply custom configuration if provided
        if any([args.custom_n_layer, args.custom_n_embd, args.custom_n_head, 
                args.custom_n_kv_head, args.custom_vocab_size, args.custom_block_size]):
            print0("Applying custom model configuration...")
            
            # Create custom config
            custom_config = LlamaConfig()
            if args.custom_n_layer is not None:
                custom_config.n_layer = args.custom_n_layer
                print0(f"Custom n_layer: {args.custom_n_layer}")
            if args.custom_n_embd is not None:
                custom_config.n_embd = args.custom_n_embd
                print0(f"Custom n_embd: {args.custom_n_embd}")
            if args.custom_n_head is not None:
                custom_config.n_head = args.custom_n_head
                print0(f"Custom n_head: {args.custom_n_head}")
            if args.custom_n_kv_head is not None:
                custom_config.n_kv_head = args.custom_n_kv_head
                print0(f"Custom n_kv_head: {args.custom_n_kv_head}")
            if args.custom_vocab_size is not None:
                custom_config.vocab_size = args.custom_vocab_size
                print0(f"Custom vocab_size: {args.custom_vocab_size}")
            if args.custom_block_size is not None:
                custom_config.block_size = args.custom_block_size
                print0(f"Custom block_size: {args.custom_block_size}")
            
            # Create new model with custom config
            print0("Creating model with custom configuration...")
            model = LLaMA(custom_config)
            print0(model)
            
            # Initialize weights properly
            model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
        else:
            # Use default HF model (if needed - currently not implemented)
            raise NotImplementedError("Default HF model loading not implemented")
    else:  # use Meta's checkpoint
        assert args.ckpt_dir is not None and os.path.exists(args.ckpt_dir), f"llama3 ckpt dir {args.ckpt_dir} does not exist"
        assert args.tokenizer_path is not None and os.path.exists(args.tokenizer_path), f"llama3 tokenizer path {args.tokenizer_path} does not exist"
        model = LLaMA.from_pretrained_llama3_meta(args.ckpt_dir, args.tokenizer_path)

    print(f"Model initialized.")
    print_model_info(model)
    return model

def setup_dataloaders(args, ddp_rank, ddp_world_size):
    """
    Set up training and validation data loaders.
    
    Args:
        args: parsed command line arguments
        ddp_rank: DDP rank
        ddp_world_size: DDP world size
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    B, T = args.batch_size, args.sequence_length
    
    train_loader = DistributedShardedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
    val_loader = None
    if args.input_val_bin:
        val_batch_size = 4 #min(B, B // 2)  # Use half the training batch size
        val_loader = DistributedShardedDataLoader(args.input_val_bin, val_batch_size, T, ddp_rank, ddp_world_size)
        print0(f"Validation loader initialized with batch size {val_batch_size}.")
    return train_loader, val_loader

def write_debug_tensors(args, model, train_loader, device, master_process):
    """
    Write tensors for C bridge debugging if requested.
    
    Args:
        args: parsed command line arguments
        model: the model
        train_loader: training data loader
        device: compute device
        master_process: whether this is the master process
    """
    if master_process and args.write_tensors and (not args.inference_only):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y)
        loss.backward()
        
        # save model params, in bfloat16
        model_to_size = {"meta-llama/Meta-Llama-3.1-8B": "8B"}
        model_size_str = model_to_size[args.model]
        write_model(model, os.path.join(args.output_dir, f"llama3.1_{model_size_str}_bf16.bin"), dtype="bfloat16")
        
        # save x, y, logits, loss, and parameter gradients, for debugging C
        write_state(model, x, y, logits, loss, os.path.join(args.output_dir, f"llama3_{model_size_str}_debug_state.bin"))
        
        # reset the train_loader for the optimization below
        train_loader.reset()

def train(args, model, train_loader, val_loader=None, logfile=None):
    """
    Main training function that handles the training loop.
    
    Args:
        args: parsed command line arguments
        model: the LLaMA model to train (already wrapped in DDP if needed)
        train_loader: training data loader
        val_loader: validation data loader (optional)
        logfile: path to log file (optional)
    
    Returns:
        trained model (unwrapped from DDP)
    """
    # Get device info
    device = next(model.parameters()).device
    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
    
    # DDP setup
    ddp = isinstance(model, DDP)
    if ddp:
        ddp_rank = int(os.environ['RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        master_process = ddp_rank == 0
    else:
        ddp_world_size = 1
        master_process = True

    # Calculate gradient accumulation steps
    B, T = args.batch_size, args.sequence_length
    tokens_per_fwdbwd = B * T * ddp_world_size
    grad_accum_steps = args.total_batch_size // tokens_per_fwdbwd

    # Set up autocast context
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if (device_type == "cuda") else nullcontext()

    # Get raw model (unwrapped from DDP)
    raw_model = model.module if ddp else model

    # Initialize optimizer
    zero_stage = args.zero_stage if ddp else 0
    optimizer = raw_model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate, 
        betas=(0.9, 0.95),
        device_type=device_type, 
        zero_stage=zero_stage
    )

    # FLOPS calculation setup
    gpu_model = detect_gpu_model()
    peak_flops = get_gpu_peak_flops(gpu_model)
    flops_per_forward = calculate_model_flops(raw_model, B, T)
    flops_per_step = flops_per_forward * grad_accum_steps * 3  # 3x for forward + 2x backward
    
    print0(f"GPU: {gpu_model}")
    print0(f"Peak FLOPS: {peak_flops/1e12:.1f} TFLOPS")
    print0(f"Model FLOPS per forward pass: {flops_per_forward/1e12:.3f} TFLOPS")
    print0(f"Model FLOPS per training step: {flops_per_step/1e12:.3f} TFLOPS")

    # Handle checkpoint resuming
    start_step = 0
    if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
        start_step = load_checkpoint(args.resume_from_checkpoint, raw_model, optimizer, train_loader)
        print0(f"Resuming training from step {start_step}")

    # Learning rate scheduler
    def get_lr(it):
        min_lr = args.learning_rate * args.learning_rate_decay_frac
        if it < args.warmup_iters:
            return args.learning_rate * (it+1) / args.warmup_iters
        if it > args.num_iterations:
            return min_lr
        decay_ratio = (it - args.warmup_iters) / (args.num_iterations - args.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (args.learning_rate - min_lr)

    # Reset memory stats
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    
    timings = []
    norm = -1.0

    # Main training loop
    # for step in range(args.num_iterations + 1):
    for step in range(start_step, args.num_iterations + 1):
        t0 = time.time()
        last_step = (step == args.num_iterations)

        # Validation evaluation
        if (args.val_loss_every > 0 and 
            (step % args.val_loss_every == 0 or last_step) and 
            val_loader is not None):
            
            # clear memory 
            # torch.cuda.empty_cache()
            
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss = 0.0
                for _ in range(args.val_max_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    _, loss = model(x, y, return_logits=False)
                    val_loss += loss.item()
                val_loss /= args.val_max_steps
            
            print0(f"val loss {val_loss}")
            if master_process and logfile is not None:
                with open(logfile, "a") as f:
                    f.write("s:%d tel:%f\n" % (step, val_loss))
            
            # clear memory 
            # torch.cuda.empty_cache()

        # Model sampling/inference
        if (args.sample_every > 0 and 
            (step % args.sample_every == 0 or last_step) and 
            master_process):
            model.eval()
            prompts = [
                "Clearly, the meaning of life is",
                "Simply put, the theory of relativity states that",
                """The repo llm.c on GitHub is""",
                """Translate English to French:

                sea otter => loutre de mer
                peppermint => menthe poivrée
                plush girafe => girafe peluche
                cheese =>""",
            ]
            
            if args.use_hf:
                prompt_tokens = [model.tokenizer(x).input_ids for x in prompts]
            else:
                prompt_tokens = [model.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

            generation_tokens = model.generate(prompt_tokens, max_gen_len=64, temperature=0.6, top_p=0.9, echo=False)
            results = [{"generation": model.tokenizer.decode(t)} for t in generation_tokens]
            for prompt, result in zip(prompts, results):
                print(prompt, end="")
                print(f"{result['generation']}")
                print("\n==================================\n")

        if last_step:
            break

        # Training step
        model.train()
        optimizer.zero_grad(set_to_none=True)
        
        if args.overfit_single_batch:
            train_loader.reset()
        
        # Gradient accumulation loop
        lossf = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            
            with ctx:
                _, loss = model(x, y, return_logits=False)
                loss = loss / grad_accum_steps
                lossf += loss.detach()
            
            if not args.inference_only:
                loss.backward()
        
        if ddp:
            dist.all_reduce(lossf, op=dist.ReduceOp.AVG)
        
        lossf = lossf.item()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        # Update learning rate and step optimizer
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()

        # Synchronize and timing
        if str(device) == "mps":
            torch.mps.synchronize()
        elif str(device).startswith("cuda"):
            torch.cuda.synchronize()
        
        t1 = time.time()
        dt = t1 - t0
        total_tokens_per_second = grad_accum_steps * ddp_world_size * B * T / (t1-t0)
        if ddp:
            tokens_per_second_gpu = total_tokens_per_second // ddp_world_size
        else:
            tokens_per_second_gpu = total_tokens_per_second
        # Calculate FLOPS metrics
        # model_flops_per_sec = flops_per_step * ddp_world_size / dt
        # mfu = calculate_mfu(model_flops_per_sec, peak_flops * ddp_world_size)

        mfu = calculate_mfu_from_throughput(model, tokens_per_second_gpu)
        # if ddp:
        
        # print0(f"step {step+1:4d}/{args.num_iterations} | train loss {lossf:.6f} | norm {norm:.4f} | lr {lr:.2e} | ({(t1-t0)*1000:.2f} ms | {total_tokens_per_second:.0f} tok/s) | {mfu:.2f}% MFU")
        
        # Also log structured metrics for easier parsing
        if master_process:
            training_logger.log_metrics(step+1, {
                'train_loss': f"{lossf:.6f}",
                'grad_norm': f"{norm:.4f}",
                'learning_rate': f"{lr:.2e}",
                'step_time_ms': f"{(t1-t0)*1000:.2f}",
                'tokens_per_sec': f"{total_tokens_per_second:.0f}",
                'mfu_percent': f"{mfu:.2f}"
            })

        # Save checkpoint
        if (step > 0 and step % (args.checkpoint_every-1) == 0) or (step == args.num_iterations-1):
            if master_process:
                print0(f"saving checkpoint at step {step+1} to {args.output_dir}")
                save_checkpoint(step+1, args, raw_model, optimizer, lossf, train_loader)
        
        # Log to file
        if master_process and logfile is not None:
            with open(logfile, "a") as f:
                f.write("s:%d trl:%f\n" % (step, lossf))

        # Track timings
        if step > 0 and step > args.num_iterations - 20:
            timings.append(t1-t0)

    # Final timing report
    timings = timings[-20:]
    print0(f"final {len(timings)} iters avg: {np.mean(timings)*1000:.3f}ms")
    if str(device).startswith("cuda"):
        print0(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    return raw_model

def arg_parser():
    # default settings will overfit a tiny batch of data
    # and save model weights and debug state to disk on the first iteration
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_hf", type=int, default=1, help="use HuggingFace (default) or use Meta's model")
    parser.add_argument("--ckpt_dir", type=str, default=None, help="path to llama3 model checkpoint (needed if use_hf=0)")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="path to llama3 tokenizer (needed if use_hf=0)")
    # file system input / output
    # parser.add_argument("--input_bin", type=str, default="/home/grads/s/sls7161/Documents/MLSystems/ProgrammingLLMs/llm-pretraining/llm.c/dev/data/fineweb10B/fineweb_train_*.bin", help="input .bin to train on")
    # parser.add_argument("--input_val_bin", type=str, default="/home/grads/s/sls7161/Documents/MLSystems/ProgrammingLLMs/llm-pretraining/llm.c/dev/data/fineweb10B/fineweb_val_*.bin", help="input .bin to eval validation loss on")
    parser.add_argument("--input_bin", type=str, default="/home/grads/s/sls7161/Documents/MLSystems/ProgrammingLLMs/llm-pretraining/llm-pretraining-experiments/dev/data/tinyshakespeare/tiny_shakespeare_train.bin", help="input .bin to train on")
    parser.add_argument("--input_val_bin", type=str, default="/home/grads/s/sls7161/Documents/MLSystems/ProgrammingLLMs/llm-pretraining/llm-pretraining-experiments/dev/data/tinyshakespeare/tiny_shakespeare_val.bin", help="input .bin to eval validation loss on")
    
    parser.add_argument("--output_dir", type=str, default="", help="output directory to which to write logs and checkpoints")
    parser.add_argument("--model", type=str, default="llama-pro-mini", help="chose the llama model, meta-llama/Meta-Llama-3.1-8B, llama-pro-mini")
    # token layout for each step of the optimization
    parser.add_argument("--batch_size", type=int, default=4, help="batch size, in units of #batch dimensions")
    parser.add_argument("--sequence_length", type=int, default=1024, help="sequence length")
    parser.add_argument("--total_batch_size", type=int, default=524288, help="total desired batch size, in units of #tokens")
    # workload (number of steps)
    parser.add_argument("--num_iterations", type=int, default=10, help="number of iterations to run")
    parser.add_argument("--inference_only", type=int, default=0, help="only run inference")
    # optimization
    parser.add_argument("--learning_rate", type=float, default=6e-4, help="learning rate warmup iterations")
    parser.add_argument("--warmup_iters", type=int, default=0, help="learning rate warmup iterations")
    parser.add_argument("--learning_rate_decay_frac", type=float, default=1.0, help="learning rate warmup iterations")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="maximum gradient magnitude")
    # evaluation
    parser.add_argument("--val_loss_every", type=int, default=10, help="every how mant steps to evaluate val loss?")
    parser.add_argument("--val_max_steps", type=int, default=20, help="how many batches of val to average?")
    parser.add_argument("--sample_every", type=int, default=0, help="how often to sample from the model?")
    # debugging
    parser.add_argument("--overfit_single_batch", type=int, default=1, help="overfit just one batch of data")
    # numerics
    parser.add_argument("--tensorcores", type=int, default=1, help="use tensorcores")
    # memory management
    parser.add_argument("--device", type=str, default="", help="by default we autodetect, or set it here")
    parser.add_argument("--compile", type=int, default=0, help="torch.compile the model")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="float32|float16|bfloat16")
    parser.add_argument("--zero_stage", type=int, default=0, help="zero redundancy optimizer stage (0/1/2/3)")
    # python -> C bridge
    parser.add_argument("--write_tensors", type=int, default=0, help="write tensors to disk")
    parser.add_argument("--checkpoint_every", type=int, default=500, help="save checkpoint every N steps")
    parser.add_argument("--resume_from_checkpoint", type=str, default="", help="path to checkpoint file to resume from")
    # HuggingFace export arguments
    parser.add_argument("--export_hf", type=int, default=1, help="export model to HuggingFace format after training")
    parser.add_argument("--export_hf_dir", type=str, default="./hf_model", help="directory to save HuggingFace model")
    parser.add_argument("--push_to_hub", type=int, default=0, help="push model to HuggingFace Hub")
    parser.add_argument("--hf_model_name", type=str, default="", help="model name on HuggingFace Hub (e.g., username/model-name)")
    parser.add_argument("--private_repo", type=int, default=0, help="create private repository on HuggingFace Hub")
    parser.add_argument("--test_exported_model", type=int, default=1, help="test exported model with sample generation")
    
    # Custom model configuration arguments
    parser.add_argument("--custom_n_layer", type=int, default=14, help="override number of layers")
    parser.add_argument("--custom_n_embd", type=int, default=768, help="override embedding dimension")
    parser.add_argument("--custom_n_head", type=int, default=12, help="override number of attention heads")
    parser.add_argument("--custom_n_kv_head", type=int, default=12, help="override number of key-value heads")
    parser.add_argument("--custom_vocab_size", type=int, default=None, help="override vocabulary size")
    parser.add_argument("--custom_block_size", type=int, default=1024, help="override maximum sequence length")
    
    # Logging arguments
    parser.add_argument("--log_level", type=str, default="INFO", help="logging level (DEBUG, INFO, WARNING, ERROR)")
    parser.add_argument("--log_to_console", type=int, default=1, help="whether to also print logs to console")
    
    # training run id 
    parser.add_argument("--run_id", type=str, default="1", help="an optional id for this training run, for logging")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Parse arguments and basic setup
    args = arg_parser()
    
    # Setup DDP and device first to get master_process
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, master_process, device, device_type, zero_stage = setup_ddp_and_device(args)
    
    # Setup comprehensive logging
    training_logger, log_file_path = setup_logging(args, master_process, ddp_rank)
    
    print0(f"Running pytorch {torch.version.__version__}")
    print0(f"args: {args}")
    
    if master_process and log_file_path:
        print0(f"All output will be logged to: {log_file_path}")
    
    # Args validation
    B, T = args.batch_size, args.sequence_length
    assert 1 <= T <= 8192, "sequence length must be between 1 and 8192"
    assert args.dtype in {"float32", "float16", "bfloat16"}
    assert args.model in {"meta-llama/Meta-Llama-3.1-8B", "llama-pro-mini"} 
    args.warmup_iters = int(0.1 * args.num_iterations) if args.warmup_iters == 0 else args.warmup_iters

    # Setup logging for backward compatibility (loss metrics)
    logfile = None
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        logfile = os.path.join(args.output_dir, "main.log")
        with open(logfile, "a") as f:
            pass

    # Calculate gradient accumulation
    tokens_per_fwdbwd = B * T * ddp_world_size
    assert args.total_batch_size % tokens_per_fwdbwd == 0
    grad_accum_steps = args.total_batch_size // tokens_per_fwdbwd
    print0(f"total desired batch size: {args.total_batch_size}")
    print0(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    # Set RNG seeds
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Configure tensor cores
    if args.tensorcores:
        torch.set_float32_matmul_precision('high')

    # Initialize model
    print0("Initializing model...")
    model = initialize_model(args)
    model = model.to(device)
    model.train()
    
    if args.compile:
        if hasattr(config, "coordinate_descent_tuning"):
            config.coordinate_descent_tuning = True
        print0("compiling the model...")
        model = torch.compile(model)

    # Setup data loaders
    print0("Setting up data loaders...")
    train_loader, val_loader = setup_dataloaders(args, ddp_rank, ddp_world_size)

    # Write tensors for C bridge if requested
    write_debug_tensors(args, model, train_loader, device, master_process)

    # Wrap model in DDP if needed
    if ddp:
        print0("Wrapping model in DDP...")
        model = DDP(model, device_ids=[ddp_local_rank])

    # Run training
    print0("Starting training...")
    trained_model = train(args, model, train_loader, val_loader, logfile)

    # Export to HuggingFace format if requested
    save_to_huggingface(args, trained_model, master_process)

    print0("Training completed!")

    # Cleanup
    if ddp:
        destroy_process_group()
