import os
import math
import glob
import inspect
from dataclasses import dataclass
from pathlib import Path

from typing import (
    AbstractSet,
    Collection,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from torch.distributed.optim import ZeroRedundancyOptimizer

# import tiktoken
# from tiktoken.load import load_tiktoken_bpe

# Tiktoken imports with conditional handling
try:
    import tiktoken
    from tiktoken.load import load_tiktoken_bpe
    TIKTOKEN_AVAILABLE = True
except ImportError:
    print("Warning: tiktoken library not available. Tokenizer functionality will be disabled.")
    TIKTOKEN_AVAILABLE = False

# HuggingFace imports for export functionality
try:
    from transformers import (
        LlamaConfig as HFLlamaConfig, 
        LlamaForCausalLM as HFLlamaForCausalLM,
        LlamaTokenizer as HFLlamaTokenizer,
        AutoTokenizer,
        AutoConfig
    )
    HF_AVAILABLE = True
except ImportError:
    print("Warning: transformers library not available. HuggingFace export will be disabled.")
    HF_AVAILABLE = False

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the LLaMA 3.x model

# Used in Grouped Query Attention (GQA), broadcasts the key and value tensors
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

# -----------------------------------------------------------------------------
# RoPE related

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_scaling(freqs: torch.Tensor):
    # Values obtained from grid search
    scale_factor = 8
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192  # original llama3 length

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def precompute_freqs_cis(
    dim: int, end: int, theta: float = 10000.0, use_scaled: bool = False
):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    if use_scaled:
        freqs = apply_scaling(freqs)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

# -----------------------------------------------------------------------------
# LLaMA building blocks

# LLaMA reference code explicitly implemented RMSNorm so we copy pasted it
# (https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/api/model.py)
# we could also use nn.RMSNorm, it has slightly different numeric properties, but equivalent
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_rep = self.n_head // self.n_kv_head
        self.hd = config.n_embd // config.n_head
        self.use_kv = config.use_kv
        self.flash = True # config.flash

        self.c_attn = nn.Linear(config.n_embd, (config.n_head + 2 * config.n_kv_head) * self.hd, bias=False)  # key, query, value projections
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)  # output projection

        # static KV cache - we could alternatively allocate it outside of the model and just pass it in when needed
        if self.use_kv:
            self.cache_k = torch.zeros((config.max_gen_batch_size, config.block_size, config.n_kv_head, self.hd))
            self.cache_v = torch.zeros((config.max_gen_batch_size, config.block_size, config.n_kv_head, self.hd))

    def forward(self, x, freqs_cis=None, start_pos=None, mask=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split([self.n_head * self.hd, self.n_kv_head * self.hd, self.n_kv_head * self.hd], dim=-1)
        q, k, v = map(lambda t: t.view(B, T, -1, self.hd), (q, k, v))  # (B, T, NH, HD)

        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)  # rotate QK (rope)  <-- 1. difference compared to GPT-2

        if self.use_kv and not self.training and start_pos >= 0:  # use kv-caching during inference
            self.cache_k[:B, start_pos : start_pos + T] = k
            self.cache_v[:B, start_pos : start_pos + T] = v
            k = self.cache_k[:B, : start_pos + T]
            v = self.cache_v[:B, : start_pos + T]

        k = repeat_kv(k, self.n_rep)  # GQA <-- 2. difference compared to GPT-2
        v = repeat_kv(v, self.n_rep)

        q, k, v = map(lambda t: t.transpose(1, 2), (q, k, v))  # (B, NH, T, HD)

        if self.flash:
            # flashattention
            # if T == 1 no need to mask, otherwise the function complains
            # scaled_dot_product_attention expects a mask where value of True indicates that the element should take part in attention
            # our mask is the opposite, so we need to invert it
            y = F.scaled_dot_product_attention(q, k, v, mask == 0 if T > 1 else None)
        else:
            # manual implementation of attention
            # this materializes the large (T,T) matrix for all the queries and keys
            scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.hd))
            if mask is not None:
                scores.masked_fill_(mask, torch.finfo(scores.dtype).min)
            att = F.softmax(scores.float(), dim=-1).type_as(q)
            y = att @ v # (B, NH, T, T) x (B, NH, T, HD) -> (B, NH, T, HD)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        # hidden_dim = 4 * config.n_embd
        # hidden_dim = int(2 * hidden_dim / 3)
        # # custom dim factor multiplier
        # if config.ffn_dim_multiplier is not None:
        #     hidden_dim = int(config.ffn_dim_multiplier * hidden_dim)
        # hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)
        raw_size = (8 * config.n_embd) / 3  # apply 2/3 scaling
        multiple = 128 
        hidden_dim = int(round(raw_size / multiple + 1) * multiple)
        self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.c_fc2 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)

    def forward(self, x):
        # SwiGLU self.c_proj(F.silu(self.c_fc2(x)) * self.c_fc(x))  <-- 3. difference compared to GPT-2
        x1 = self.c_fc(x)
        x2 = self.c_fc2(x)
        x2 = F.silu(x2)
        x = x1 * x2
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd, config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd, config.norm_eps)
        self.mlp = MLP(config)

    def forward(self, x, freqs_cis=None, start_pos=None, mask=None):
        x = x + self.attn(self.ln_1(x), freqs_cis, start_pos, mask)
        x = x + self.mlp(self.ln_2(x))
        return x

# -----------------------------------------------------------------------------
# The main LLaMA 3.1 model

@dataclass
class LlamaConfig:
    version: str = "3.1"
    block_size: int = 8192
    vocab_size: int = 128256
    n_layer: int = 32
    n_head: int = 32
    n_kv_head: int = 8
    n_embd: int = 4096
    ffn_dim_multiplier: float = 1.3
    multiple_of: int = 1024
    norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    use_scaled_rope: bool = True
    max_gen_batch_size: int = 4
    use_kv: bool = False
    flash: bool = True  # use flashattention?

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        assert self.n_kv_head <= self.n_head
        assert self.n_head % self.n_kv_head == 0
        assert self.n_embd % self.n_head == 0

class LLaMA(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd, config.norm_eps),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # self.lm_head.weight = self.transformer.wte.weight

        # init all weights, use a torch rng object to be very careful
        self.init_rng = torch.Generator()
        self.init_rng.manual_seed(42)

        self.freqs_cis = precompute_freqs_cis(
            config.n_embd // config.n_head,
            config.block_size * 2,
            config.rope_theta,
            config.use_scaled_rope,
        )

    def forward(self, idx, targets=None, return_logits=True, start_pos=0):
        _, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # forward the LLaMA model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        freqs_cis = self.freqs_cis[start_pos:start_pos+t]
        freqs_cis = freqs_cis.to(x.device)

        mask = torch.triu(torch.ones((t, t), device=next(self.parameters()).device, dtype=torch.bool), diagonal=1)

        for i, block in enumerate(self.transformer.h):
            x = block(x, freqs_cis, start_pos, mask)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x).float()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]).float() # note: using list [-1] to preserve the time dim
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss

    @staticmethod
    def adapt_llama_state_dict_keys(checkpoint, config: LlamaConfig):
        # Modify key names from Meta's LLaMA to our LLaMA
        # our key names are derived from GPT-2's key names
        checkpoint['transformer.wte.weight'] = checkpoint.pop('tok_embeddings.weight')

        for i in range(config.n_layer):
            for name in ['attention_norm', 'ffn_norm']:
                old_key = f'layers.{i}.{name}.weight'  # e.g. layers.x.attention_norm.weight -> transformer.h.x.ln_1.weight
                new_key = f'transformer.h.{i}.ln_{1 if name == "attention_norm" else 2}.weight'
                checkpoint[new_key] = checkpoint.pop(old_key)

        for i in range(config.n_layer):
            for name in ['attention.wq', 'attention.wk', 'attention.wv']:
                old_key = f'layers.{i}.{name}.weight'
                new_key = f'transformer.h.{i}.attn.c_attn.weight'
                if name == 'attention.wq':
                    checkpoint[new_key] = checkpoint.pop(old_key)
                else:  # merge 3 weights into transformer.h.x.attn.c_attn.weight
                    checkpoint[new_key] = torch.cat((checkpoint[new_key], checkpoint.pop(old_key)), dim=0)
            old_key = f'layers.{i}.attention.wo.weight'
            new_key = f'transformer.h.{i}.attn.c_proj.weight'
            checkpoint[new_key] = checkpoint.pop(old_key)

        ffn_map = {'w1': 'c_fc2', 'w2': 'c_proj', 'w3': 'c_fc'}
        for i in range(config.n_layer):
            for name in ['feed_forward.w1', 'feed_forward.w2', 'feed_forward.w3']:
                old_key = f'layers.{i}.{name}.weight'
                new_key = f'transformer.h.{i}.mlp.{ffn_map[name.split(".")[-1]]}.weight'
                checkpoint[new_key] = checkpoint.pop(old_key)

        checkpoint['transformer.ln_f.weight'] = checkpoint.pop('norm.weight')
        checkpoint['lm_head.weight'] = checkpoint.pop('output.weight')

        return checkpoint

    @staticmethod
    def adapt_llama_state_dict_keys_hf(checkpoint, config: LlamaConfig):
        # Modify key names from HuggingFace's LLaMA to our LLaMA
        # our key names are derived from GPT-2's key names
        checkpoint['transformer.wte.weight'] = checkpoint.pop('model.embed_tokens.weight')

        # We need to unpermute K and V because HF script permuted the original Meta-LLaMA weights
        # see: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py
        def unpermute(w, n_heads, dim1, dim2):
            return w.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)

        for i in range(config.n_layer):
            for name in ['input_layernorm', 'post_attention_layernorm']:
                old_key = f'model.layers.{i}.{name}.weight'  # e.g. layers.x.attention_norm.weight -> transformer.h.x.ln_1.weight
                new_key = f'transformer.h.{i}.ln_{1 if name == "input_layernorm" else 2}.weight'
                checkpoint[new_key] = checkpoint.pop(old_key)

        for i in range(config.n_layer):
            for name in ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj']:
                old_key = f'model.layers.{i}.{name}.weight'
                new_key = f'transformer.h.{i}.attn.c_attn.weight'
                if name == 'self_attn.q_proj':
                    checkpoint[new_key] = unpermute(checkpoint.pop(old_key), config.n_head, config.n_embd, config.n_embd)
                else:  # merge 3 weights into transformer.h.x.attn.c_attn.weight
                    tensor = checkpoint.pop(old_key)
                    if name == 'self_attn.k_proj':
                        tensor = unpermute(tensor, config.n_kv_head, config.n_kv_head * (config.n_embd // config.n_head), config.n_embd)
                    checkpoint[new_key] = torch.cat((checkpoint[new_key], tensor), dim=0)
            old_key = f'model.layers.{i}.self_attn.o_proj.weight'
            new_key = f'transformer.h.{i}.attn.c_proj.weight'
            checkpoint[new_key] = checkpoint.pop(old_key)

        ffn_map = {'gate_proj': 'c_fc2', 'down_proj': 'c_proj', 'up_proj': 'c_fc'}
        for i in range(config.n_layer):
            for name in ['gate_proj', 'down_proj', 'up_proj']:
                old_key = f'model.layers.{i}.mlp.{name}.weight'
                new_key = f'transformer.h.{i}.mlp.{ffn_map[name]}.weight'
                checkpoint[new_key] = checkpoint.pop(old_key)

        checkpoint['transformer.ln_f.weight'] = checkpoint.pop('model.norm.weight')

        return checkpoint

    @classmethod
    def from_pretrained_llama3_hf(cls, model_id):
        """Loads pretrained LLaMA model weights from HuggingFace"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        assert model_id == "meta-llama/Meta-Llama-3.1-8B", "Only the 8B-base model is supported for now"
        model_args = LlamaConfig()

        model = AutoModelForCausalLM.from_pretrained(model_id)
        checkpoint = LLaMA.adapt_llama_state_dict_keys_hf(model.state_dict(), model_args)

        original_default_type = torch.get_default_dtype()  # save the default type
        torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)  # much faster loading
        model = LLaMA(model_args)
        model.load_state_dict(checkpoint, strict=False)
        torch.set_default_tensor_type(torch.tensor([], dtype=original_default_type, device="cpu").type())  # restore default type

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_id = 128004  # this is the pad token id for LLaMA 3.1 base, we need to set this explicitly as our generate func expects it
        tokenizer.stop_tokens = [tokenizer.eos_token_id]
        model.tokenizer = tokenizer
        return model

    @classmethod
    def from_pretrained_llama3_meta(cls, ckpt_dir, tokenizer_path):
        """Loads pretrained LLaMA model weights from a checkpoint directory"""
        model_args = LlamaConfig()

        ckpt_path = sorted(Path(ckpt_dir).glob("*.pth"))[0]
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        checkpoint = LLaMA.adapt_llama_state_dict_keys(checkpoint, model_args)

        original_default_type = torch.get_default_dtype()  # save the default type
        torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)  # much faster loading
        model = LLaMA(model_args)
        model.load_state_dict(checkpoint, strict=False)
        torch.set_default_tensor_type(torch.tensor([], dtype=original_default_type, device="cpu").type())  # restore default type

        tokenizer = Tokenizer(model_path=tokenizer_path)
        model.tokenizer = tokenizer
        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, zero_stage):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print0(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print0(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        print0(f"using fused AdamW: {use_fused}")
        if zero_stage == 1:
            print0("using ZeroRedundancyOptimizer")
            optimizer = ZeroRedundancyOptimizer(**optim_groups[0], optimizer_class=torch.optim.AdamW,
                                                lr=learning_rate, betas=betas, fused=use_fused)
            optimizer.add_param_group(optim_groups[1])
        else:
            print0("using regular AdamW")
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=use_fused)
        return optimizer

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        echo: bool = False,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """
        Generate text sequences based on provided prompts using the language generation model.

        Args:
            prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
            max_gen_len (int): Maximum length of the generated text sequence.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            Tuple[List[List[int]], Optional[List[List[float]]]]: A tuple containing generated token sequences.

        Note:
            This method uses the provided prompts as a basis for generating text. It employs nucleus sampling to produce text with controlled randomness.

        """
        bsz = len(prompt_tokens)
        assert bsz <= self.config.max_gen_batch_size, f"Batch size {bsz} exceeds the maximum generation batch size {self.config.max_gen_batch_size}"
        device = next(self.parameters()).device

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= self.config.block_size, f"Prompt length {max_prompt_len} exceeds the maximum block size {self.config.block_size}"
        total_len = min(self.config.block_size, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=device)
        for idx, t in enumerate(prompt_tokens):
            tokens[idx, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device=device)
        input_text_mask = tokens != pad_id

        if min_prompt_len == total_len:
            logits, _ = self.forward(tokens, start_pos=prev_pos)

        stop_tokens = torch.tensor(list(self.tokenizer.stop_tokens)).to(device)

        for cur_pos in range(min_prompt_len, total_len):
            logits, _ = self.forward(tokens[:, prev_pos:cur_pos], start_pos=prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token
            eos_reached |= ~input_text_mask[:, cur_pos] & torch.isin(next_token, stop_tokens)
            prev_pos = cur_pos
            if all(eos_reached):
                break

        out_tokens = []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            # cut to after eos tok if any
            for stop_token in self.tokenizer.stop_tokens:
                try:
                    eos_idx = toks.index(stop_token)
                    toks = toks[:eos_idx]
                except ValueError:
                    pass
            out_tokens.append(toks)
        return out_tokens

# -----------------------------------------------------------------------------
# sampling utils

def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

# -----------------------------------------------------------------------------
# Llama 3.1 Tokenizer

# The tiktoken tokenizer can handle <=400k chars without
# pyo3_runtime.PanicException.
TIKTOKEN_MAX_ENCODE_CHARS = 400_000

# https://github.com/openai/tiktoken/issues/195
# Here we iterate over subsequences and split if we exceed the limit
# of max consecutive non-whitespace or whitespace characters.
MAX_NO_WHITESPACES_CHARS = 25_000


class Tokenizer:
    """
    Tokenizing and encoding/decoding text using the Tiktoken tokenizer.
    """

    special_tokens: Dict[str, int]

    num_reserved_special_tokens = 256

    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501

    def __init__(self, model_path: str):
        """
        Initializes the Tokenizer with a Tiktoken model.

        Args:
            model_path (str): The path to the Tiktoken model file.
        """
        if not TIKTOKEN_AVAILABLE:
            raise ImportError("tiktoken library is required for Tokenizer. Install with: pip install tiktoken")
        
        assert os.path.isfile(model_path), model_path

        mergeable_ranks = load_tiktoken_bpe(model_path)
        num_base_tokens = len(mergeable_ranks)
        special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|finetune_right_pad_id|>",
            "<|step_id|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|eom_id|>",  # end of message
            "<|eot_id|>",  # end of turn
            "<|python_tag|>",
        ]
        reserved_tokens = [
            f"<|reserved_special_token_{2 + i}|>"
            for i in range(self.num_reserved_special_tokens - len(special_tokens))
        ]
        special_tokens = special_tokens + reserved_tokens

        self.special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens)
        }
        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )

        self.n_words: int = num_base_tokens + len(special_tokens)
        # BOS / EOS token IDs
        self.bos_id: int = self.special_tokens["<|begin_of_text|>"]
        self.eos_id: int = self.special_tokens["<|end_of_text|>"]
        self.eot_id: int = self.special_tokens["<|eot_id|>"]
        self.eom_id: int = self.special_tokens["<|eom_id|>"]
        self.python_tag_id = self.special_tokens["<|python_tag|>"]
        self.pad_id: int = self.special_tokens["<|finetune_right_pad_id|>"]
        # hardcoded stop tokens for the base model
        self.stop_tokens = [
            self.special_tokens["<|begin_of_text|>"],
            self.special_tokens["<|end_of_text|>"],
        ]

    def encode(
        self,
        s: str,
        *,
        bos: bool,
        eos: bool,
        allowed_special: Optional[Union[Literal["all"], AbstractSet[str]]] = None,
        disallowed_special: Union[Literal["all"], Collection[str]] = (),
    ) -> List[int]:
        """
        Encodes a string into a list of token IDs.

        Args:
            s (str): The input string to be encoded.
            bos (bool): Whether to prepend the beginning-of-sequence token.
            eos (bool): Whether to append the end-of-sequence token.
            allowed_tokens ("all"|set[str]): allowed special tokens in string
            disallowed_tokens ("all"|set[str]): special tokens that raise an error when in string

        Returns:
            list[int]: A list of token IDs.

        By default, setting disallowed_special=() encodes a string by ignoring
        special tokens. Specifically:
        - Setting `disallowed_special` to () will cause all text corresponding
          to special tokens to be encoded as natural text (insteading of raising
          an error).
        - Setting `allowed_special` to "all" will treat all text corresponding
          to special tokens to be encoded as special tokens.
        """
        if allowed_special is None:
            allowed_special = set()
        assert type(s) is str

        substrs = (
            substr
            for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS)
            for substr in self._split_whitespaces_or_nonwhitespaces(
                s[i : i + TIKTOKEN_MAX_ENCODE_CHARS], MAX_NO_WHITESPACES_CHARS
            )
        )
        t: List[int] = []
        for substr in substrs:
            t.extend(
                self.model.encode(
                    substr,
                    allowed_special=allowed_special,
                    disallowed_special=disallowed_special,
                )
            )
        if bos:
            t.insert(0, self.bos_id)
        if eos:
            t.append(self.eos_id)
        return t

    def decode(self, t: Sequence[int]) -> str:
        # Typecast is safe here. Tiktoken doesn't do anything list-related with the sequence.
        return self.model.decode(cast(List[int], t))

    @staticmethod
    def _split_whitespaces_or_nonwhitespaces(
        s: str, max_consecutive_slice_len: int
    ) -> Iterator[str]:
        """
        Splits the string `s` so that each substring contains no more than `max_consecutive_slice_len`
        consecutive whitespaces or consecutive non-whitespaces.
        """
        current_slice_len = 0
        current_slice_is_space = s[0].isspace() if len(s) > 0 else False
        slice_start = 0

        for i in range(len(s)):
            is_now_space = s[i].isspace()

            if current_slice_is_space ^ is_now_space:
                current_slice_len = 1
                current_slice_is_space = is_now_space
            else:
                current_slice_len += 1
                if current_slice_len > max_consecutive_slice_len:
                    yield s[slice_start:i]
                    slice_start = i
                    current_slice_len = 1
        yield s[slice_start:]

# -----------------------------------------------------------------------------
# HuggingFace Export Functionality

def export_to_huggingface(model, tokenizer, output_dir, model_name=None, push_to_hub=False, 
                         private_repo=False, custom_config=None):
    """
    Export trained LLaMA model to HuggingFace format and optionally push to Hub.
    
    Args:
        model: Trained LLaMA model
        tokenizer: Model tokenizer 
        output_dir: Local directory to save the model
        model_name: Name for the model on HuggingFace Hub
        push_to_hub: Whether to push to HuggingFace Hub
        private_repo: Whether to create a private repository
        custom_config: Optional custom configuration overrides
    """
    if not HF_AVAILABLE:
        raise ImportError("transformers library is required for HuggingFace export. Install with: pip install transformers")
    
    print0(f"Exporting model to HuggingFace format in {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get model configuration
    config = model.config if hasattr(model, 'config') else model.module.config
    
    # Apply custom config overrides if provided
    if custom_config:
        for key, value in custom_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
                print0(f"Updated config.{key} = {value}")
    sample_mlp = model.transformer.h[0].mlp if hasattr(model, 'transformer') else model.module.transformer.h[0].mlp
    actual_intermediate_size = sample_mlp.c_fc.out_features

    # Create HuggingFace config
    hf_config = HFLlamaConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.n_embd,
        intermediate_size=actual_intermediate_size,
        num_hidden_layers=config.n_layer,
        num_attention_heads=config.n_head,
        num_key_value_heads=config.n_kv_head,
        max_position_embeddings=config.block_size,
        rms_norm_eps=config.norm_eps,
        rope_theta=config.rope_theta,
        rope_scaling=None,
        tie_word_embeddings=False,
        torch_dtype="bfloat16"
    )
    
    # Create HuggingFace model
    hf_model = HFLlamaForCausalLM(hf_config)
    
    # Convert state dict from our format to HuggingFace format
    our_state_dict = model.state_dict() if not hasattr(model, 'module') else model.module.state_dict()
    hf_state_dict = convert_state_dict_to_hf(our_state_dict, config)
    
    # Load the converted state dict
    hf_model.load_state_dict(hf_state_dict, strict=False)
    
    # Save model locally
    hf_model.save_pretrained(output_dir, max_shard_size="5GB", safe_serialization=True)
    hf_config.save_pretrained(output_dir)
    
    # Handle tokenizer
    # if hasattr(tokenizer, 'save_pretrained'):
    #     # HuggingFace tokenizer
    #     tokenizer.save_pretrained(output_dir)
    # else:
    #     # Our custom tokenizer - use a base LLaMA tokenizer as fallback
    #     try:
    #         # base_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    #         base_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
    #         base_tokenizer.save_pretrained(output_dir)
    #         print0("Warning: Using base LLaMA 3.1 tokenizer as fallback")
    #     except:
    #         print0("Warning: Could not save tokenizer. You may need to manually add a tokenizer.")
    
    # Always use Meta-Llama-3.1-8B tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
        print0("Using Meta-Llama-3.1-8B tokenizer")
    except Exception as e:
        print0(f"Error loading Meta-Llama-3.1-8B tokenizer: {e}")
        raise Exception("Could not load LLaMA-3.1-8B tokenizer")
    
    # Save the tokenizer (now guaranteed to be not None)
    tokenizer.save_pretrained(output_dir)
    print0("Tokenizer saved successfully")
    
    # Push to HuggingFace Hub if requested
    if push_to_hub and model_name:
        try:
            print0(f"Pushing model to HuggingFace Hub as {model_name}")
            hf_model.push_to_hub(model_name, private=private_repo)
            if hasattr(tokenizer, 'push_to_hub'):
                tokenizer.push_to_hub(model_name, private=private_repo)
            else:
                base_tokenizer.push_to_hub(model_name, private=private_repo)
            print0(f"Successfully pushed model to https://huggingface.co/{model_name}")
        except Exception as e:
            print0(f"Error pushing to Hub: {e}")
            print0("Make sure you're logged in with: huggingface-cli login")
    
    return hf_model, output_dir

def convert_state_dict_to_hf(our_state_dict, config):
    """Convert our LLaMA state dict to HuggingFace format"""
    hf_state_dict = {}
    
    # Embedding layer
    hf_state_dict['model.embed_tokens.weight'] = our_state_dict['transformer.wte.weight']
    
    # We need to permute K and V projections for HuggingFace format
    def permute(w, n_heads, dim1, dim2):
        return w.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)
    
    # Transform each layer
    for i in range(config.n_layer):
        layer_prefix = f'transformer.h.{i}'
        hf_layer_prefix = f'model.layers.{i}'
        
        # Layer norms
        hf_state_dict[f'{hf_layer_prefix}.input_layernorm.weight'] = our_state_dict[f'{layer_prefix}.ln_1.weight']
        hf_state_dict[f'{hf_layer_prefix}.post_attention_layernorm.weight'] = our_state_dict[f'{layer_prefix}.ln_2.weight']
        
        # Attention weights - need to split and potentially permute
        qkv_weight = our_state_dict[f'{layer_prefix}.attn.c_attn.weight']
        head_dim = config.n_embd // config.n_head
        
        # Split QKV
        q_weight = qkv_weight[:config.n_head * head_dim]
        k_weight = qkv_weight[config.n_head * head_dim:config.n_head * head_dim + config.n_kv_head * head_dim]
        v_weight = qkv_weight[config.n_head * head_dim + config.n_kv_head * head_dim:]
        
        # Permute for HuggingFace format
        hf_state_dict[f'{hf_layer_prefix}.self_attn.q_proj.weight'] = permute(
            q_weight, config.n_head, config.n_embd, config.n_embd)
        hf_state_dict[f'{hf_layer_prefix}.self_attn.k_proj.weight'] = permute(
            k_weight, config.n_kv_head, config.n_kv_head * head_dim, config.n_embd)
        hf_state_dict[f'{hf_layer_prefix}.self_attn.v_proj.weight'] = v_weight
        
        # Output projection
        hf_state_dict[f'{hf_layer_prefix}.self_attn.o_proj.weight'] = our_state_dict[f'{layer_prefix}.attn.c_proj.weight']
        
        # MLP weights
        hf_state_dict[f'{hf_layer_prefix}.mlp.gate_proj.weight'] = our_state_dict[f'{layer_prefix}.mlp.c_fc2.weight']
        hf_state_dict[f'{hf_layer_prefix}.mlp.up_proj.weight'] = our_state_dict[f'{layer_prefix}.mlp.c_fc.weight']  
        hf_state_dict[f'{hf_layer_prefix}.mlp.down_proj.weight'] = our_state_dict[f'{layer_prefix}.mlp.c_proj.weight']
    
    # Final layer norm
    hf_state_dict['model.norm.weight'] = our_state_dict['transformer.ln_f.weight']
    
    # LM head (output layer)
    hf_state_dict['lm_head.weight'] = our_state_dict['lm_head.weight']
    
    return hf_state_dict

def test_exported_model(model_path, prompt="The meaning of life is", max_length=50):
    """Test the exported HuggingFace model"""
    if not HF_AVAILABLE:
        print0("transformers library not available for testing")
        return
        
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print0(f"Testing exported model from {model_path}")
        print0(f"Prompt: {prompt}")
        print0("-" * 50)
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16, 
            device_map="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Generate
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode and print
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print0(response)
        print0("-" * 50)
        
    except Exception as e:
        print0(f"Error testing model: {e}")

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
    if header[0] != 20240801:
        print("ERROR: magic number mismatch in the data .bin file!")
        exit(1)
    assert header[1] == 7, "unsupported version"
    ntok = header[2] # number of tokens (claimed)
    return ntok # for now just return the number of tokens

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        assert header[0] == 20240801, "magic number mismatch in the data .bin file"
        assert header[1] == 7, "unsupported version"
        ntok = header[2] # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint32)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens

class DistributedShardedDataLoader:
    """
    This DataLoader is both:
    - distributed (works correctly in case of multiple processes in DDP)
    - sharded (supports datasets that are broken up into multiple data shards)
    It is not *permuted*, meaning that it itearates over the data in the order
    of the dataset on disk, so the user should make sure to shuffle their examples
    during the creation of their data shards for best performance.
    """
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += shard_ntok
        self.ntok_total = ntok_total
        print0(f"DataLoader: total number of tokens: {ntok_total:,} across {len(self.files)} files")

        # kick things off
        self.current_shard = None
        self.reset()

    def reset(self):
        # we're being a bit clever here: if we already had shard 0 loaded,
        # then don't do the work to reload it, just reset the pointer
        if self.current_shard != 0:
            self.current_shard = 0
            self.tokens = _load_data_shard(self.files[self.current_shard])
        self.current_position = self.process_rank * self.B * self.T

    def advance(self): # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        buf = torch.tensor(buf, dtype=torch.long)
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the start pointer in current shard
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds advance the shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x, y

    # NEW METHODS FOR CHECKPOINT SUPPORT
    def get_state(self):
        """Get the current state of the data loader for checkpointing"""
        return {
            'current_shard': self.current_shard,
            'current_position': self.current_position,
            'files': self.files,  # Save file list for consistency check
        }
    
    def set_state(self, state):
        """Restore the data loader state from a checkpoint"""
        # Verify file consistency
        if state['files'] != self.files:
            print0("Warning: Data files have changed since checkpoint. This may affect training.")
        
        self.current_shard = state['current_shard']
        
        # Load the correct shard if needed
        if self.current_shard != getattr(self, '_loaded_shard', None):
            self.tokens = _load_data_shard(self.files[self.current_shard])
            self._loaded_shard = self.current_shard
        
        self.current_position = state['current_position']
        
        print0(f"Restored data loader state: shard {self.current_shard}, position {self.current_position}")
        
# -----------------------------------------------------------------------------
# Python -> C bridge utilities for saving params/grads/activations to .bin files

def write_fp32(tensor, file):
    t = tensor.detach().cpu().to(torch.float32)
    b = t.numpy().tobytes()
    file.write(b)

def write_bf16(tensor, file):
    t = tensor.detach().cpu().to(torch.bfloat16)
    # numpy doesn't have bf16 datatype so we have to trick it
    t = t.view(torch.int16) # trick: reinterpret as int16
    b = t.numpy().tobytes()
    file.write(b)

def write_tensors(model_tensors, L, file, dtype):
    # writes LLaMA 3 model's weights to a binary file
    assert dtype in {"float32", "bfloat16"}
    write_fun = write_fp32 if dtype == "float32" else write_bf16
    write_fun(model_tensors["transformer.wte.weight"], file) # (V, C)
    for i in range(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_1.weight"], file)
    for i in range(L): # (L, 3C, C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_attn.weight"], file)
    for i in range(L): # (L, C, C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_proj.weight"], file)
    for i in range(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_2.weight"], file)
    for i in range(L): # (L, 4C, C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_fc.weight"], file)
    for i in range(L): # (L, 4C, C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_fc2.weight"], file)
    for i in range(L): # (L, C, 4C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_proj.weight"], file)
    write_fun(model_tensors["transformer.ln_f.weight"], file) # (C, )
    write_fun(model_tensors["lm_head.weight"], file) # (V, C)

def write_model(model, filename, dtype):
    # everything we need to instantiate the model
    # 1) header is: version int, LLaMAConfig ints, padding to 1024 bytes
    assert dtype in {"float32", "bfloat16"}
    version = {
        "float32": 3, # 3: all tensors are fp32
        "bfloat16": 5, # 5: all tensors are bf16
    }[dtype]
    header = torch.zeros(256, dtype=torch.int32)
    header[0] = 20240803 # magic
    header[1] = version # checkpoint version
    header[2] = model.config.block_size
    header[3] = model.config.vocab_size
    header[4] = model.config.n_layer
    header[5] = model.config.n_head
    header[6] = model.config.n_kv_head
    header[7] = model.config.n_embd
    header[8] = model.config.ffn_dim_multiplier
    header[9] = model.config.multiple_of
    header[10] = model.config.norm_eps
    header[11] = model.config.rope_theta
    header[12] = model.config.use_scaled_rope
    header[13] = model.config.max_gen_batch_size
    header[14] = int(model.config.version.split('.')[0]) # major version
    header[15] = int(model.config.version.split('.')[1]) # minor version
    # 2) the parameters follow the header
    params = {name: param.cpu() for name, param in model.named_parameters()}
    # now write to file
    with open(filename, "wb") as file:
        file.write(header.numpy().tobytes()) # header
        write_tensors(params, model.config.n_layer, file, dtype) # params
    print(f"wrote {filename}")

def write_state(model, x, y, logits, loss, filename):
    # the state is used for debugging.
    # it contains information about the input, logits, loss, and the parameter gradients
    # this can be used for checking the computation correctness in C
    header = torch.zeros(256, dtype=torch.int32)
    header[0] = 20240803 # magic
    header[1] = x.size(0) # batch size of the batch, B
    header[2] = x.size(1) # temporal extent of the batch, T
    grads = {name: param.grad.cpu() for name, param in model.named_parameters()}
    with open(filename, "wb") as file:
        # header
        file.write(header.numpy().tobytes())
        # input x
        file.write(x.cpu().numpy().astype("int32").tobytes()) # (B, T)
        # targets y
        file.write(y.cpu().numpy().astype("int32").tobytes()) # (B, T)
        # logits (result of the model forward pass)
        write_fp32(logits.cpu(), file)
        # loss (single float, result of the cross entropy loss)
        write_fp32(loss.cpu(), file)
        # gradients
        write_tensors(grads, model.config.n_layer, file, "float32")
    print(f"wrote {filename}")

# -----------------------------------------------------------------------------
# int main

def print0(*args, **kwargs):
    """
    Print function that only prints from the master process.
    This is overridden by the TrainingLogger during training.
    """
    # Default behavior - just check environment for rank
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)
        
def save_checkpoint(step, args, raw_model, optimizer, lossf, train_loader=None):
    """Helper function to save a checkpoint"""
    
    # Handle ZeroRedundancyOptimizer state consolidation
    # optimizer_state_dict = None
    # if optimizer is not None:
    #     if hasattr(optimizer, 'consolidate_state_dict'):
    #         # For ZeroRedundancyOptimizer, consolidate state on rank 0
    #         optimizer.consolidate_state_dict(to=0)
    #         if print0.rank == 0:
    #             optimizer_state_dict = optimizer.state_dict()
    #     else:
    #         # For regular optimizers
    #         optimizer_state_dict = optimizer.state_dict()
    
    checkpoint = {
        'model_state_dict': raw_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        'train_loss': lossf,
        'config': raw_model.config,
        'args': vars(args),  # Save training arguments
        'rng_state': torch.get_rng_state(),
    }
    
    # Save data loader state if provided
    if train_loader is not None:
        checkpoint['train_loader_state'] = train_loader.get_state()
    
    # Save CUDA RNG state if available
    if torch.cuda.is_available():
        checkpoint['cuda_rng_state'] = torch.cuda.get_rng_state()
    
    checkpoint_path = os.path.join(args.output_dir, f"checkpoint_{step}.pt")
    torch.save(checkpoint, checkpoint_path)
    
    # Also save as latest checkpoint
    latest_path = os.path.join(args.output_dir, "checkpoint_latest.pt")
    torch.save(checkpoint, latest_path)
    
    print0(f"Saved checkpoint to {checkpoint_path}")

def load_checkpoint(checkpoint_path, model, optimizer, train_loader=None):
    """
    Load checkpoint and return the step number to resume from.
    
    Args:
        checkpoint_path: path to checkpoint file
        model: model to load state into
        optimizer: optimizer to load state into
        train_loader: data loader to restore state into (optional)
        
    Returns:
        resume_step: step number to resume training from
    """
    print0(f"Loading checkpoint from {checkpoint_path}")
    # checkpoint = torch.load(checkpoint_path, map_location='cpu')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Restore data loader state
    if train_loader is not None and 'train_loader_state' in checkpoint:
        train_loader.set_state(checkpoint['train_loader_state'])
    
    # Restore RNG states for reproducibility
    if 'rng_state' in checkpoint:
        torch.set_rng_state(checkpoint['rng_state'])
    if 'cuda_rng_state' in checkpoint and torch.cuda.is_available():
        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
    
    # Get resume step
    resume_step = checkpoint.get('step', 0)
    
    print0(f"Resumed from step {resume_step}")
    return resume_step


def calculate_model_flops(model, batch_size, sequence_length):
    """
    Calculate FLOPs for one forward pass of the LLaMA model.
    
    Args:
        model: the LLaMA model
        batch_size: batch size
        sequence_length: sequence length
        
    Returns:
        flops_per_forward: FLOPs for one forward pass
    """
    config = model.config
    n_layer = config.n_layer
    n_embd = config.n_embd
    n_head = config.n_head
    n_kv_head = getattr(config, 'n_kv_head', n_head)  # Default to n_head if not specified
    vocab_size = config.vocab_size
    
    B, T = batch_size, sequence_length
    
    # Embedding layer: B * T * vocab_size * n_embd
    embedding_flops = B * T * vocab_size * n_embd
    
    # Transformer layers
    layer_flops = 0
    
    for _ in range(n_layer):
        # Attention
        # QKV projection: B * T * n_embd * (n_head + 2 * n_kv_head) * (n_embd // n_head)
        qkv_flops = B * T * n_embd * (n_head + 2 * n_kv_head) * (n_embd // n_head)
        
        # Attention computation: B * n_head * T * T * (n_embd // n_head)
        attn_flops = B * n_head * T * T * (n_embd // n_head)
        
        # Output projection: B * T * n_embd * n_embd
        out_proj_flops = B * T * n_embd * n_embd
        
        # MLP (SwiGLU): 3 linear layers
        # Gate & Up projections: 2 * (B * T * n_embd * 4 * n_embd)
        # Down projection: B * T * 4 * n_embd * n_embd
        mlp_flops = 2 * (B * T * n_embd * 4 * n_embd) + (B * T * 4 * n_embd * n_embd)
        mlp_flops = B * T * n_embd * 4 * n_embd * 3  # Simplified
        
        layer_flops += qkv_flops + attn_flops + out_proj_flops + mlp_flops
    
    # Output layer: B * T * n_embd * vocab_size
    output_flops = B * T * n_embd * vocab_size
    
    total_flops = embedding_flops + layer_flops + output_flops
    
    return total_flops

def detect_gpu_model():
    """Detect the GPU model for FLOPS calculation."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        
        if "A100" in gpu_name:
            return "A100"
        elif "A5000" in gpu_name:
            return "A5000"
        elif "V100" in gpu_name:
            return "V100"
        elif "H100" in gpu_name:
            return "H100"
        elif "RTX 4090" in gpu_name or "4090" in gpu_name:
            return "RTX4090"
        elif "RTX 3090" in gpu_name or "3090" in gpu_name:
            return "RTX3090"
        else:
            print0(f"Unknown GPU: {gpu_name}, using A100 specs as default")
            return "A100"
    
    return "A100"  # Default fallback

def get_gpu_peak_flops(device_name="A100"):
    """
    Get theoretical peak FLOPS for different GPU types.
    
    Args:
        device_name: GPU model name
        
    Returns:
        peak_flops: Peak FLOPS in operations per second
    """
    # Peak FLOPS for bfloat16 operations (using Tensor Cores)
    gpu_specs = {
        "A100": 312e12,    # 312 TFLOPS for bfloat16
        "A5000": 54e12,   # 222 TFLOPS for fp16
        "V100": 125e12,    # 125 TFLOPS for fp16  
        "H100": 989e12,    # 989 TFLOPS for bfloat16
        "RTX4090": 83e12,  # ~83 TFLOPS for bfloat16
        "RTX3090": 35e12,  # ~35 TFLOPS for fp16
    }
    
    return gpu_specs.get(device_name, 100e12)  # Default fallback

def calculate_model_flops_per_token(model):
    """
    Calculate FLOPs per token using the approximation: 6 * number of parameters.
    
    This is a common approximation where:
    - 2 * params for forward pass
    - 4 * params for backward pass (3x forward + 1x for gradient computation)
    Total = 6 * params per token
    
    Args:
        model: the LLaMA model
        
    Returns:
        flops_per_token: FLOPs required to process one token
    """
    total_params = sum(p.numel() for p in model.parameters())
    return 6 * total_params

def calculate_mfu_from_throughput(model, tokens_per_second, device_name=None):
    """
    Calculate Model FLOPS Utilization (MFU) using observed throughput.
    
    Formula: MFU = (model_flops_per_token * observed_tokens_per_second) / theoretical_peak_hardware_flops
    
    Args:
        model: PyTorch model
        tokens_per_second: Observed tokens processed per second
        device_name: GPU model name (auto-detected if None)
        
    Returns:
        mfu: Model FLOPS Utilization as percentage
    """
    if device_name is None:
        device_name = detect_gpu_model()
    
    # Calculate model FLOPs per token using 6 * num_params approximation
    model_flops_per_token = calculate_model_flops_per_token(model)
    
    # Calculate actual FLOPs per second achieved
    model_flops_per_sec = model_flops_per_token * tokens_per_second
    
    # Get theoretical peak hardware FLOPs
    peak_flops_per_sec = get_gpu_peak_flops(device_name)
    
    # Calculate MFU as percentage
    mfu = (model_flops_per_sec / peak_flops_per_sec) * 100
    
    return mfu

def calculate_mfu(model_flops_per_sec, peak_flops_per_sec):
    """
    Calculate Model FLOPS Utilization (MFU) - legacy method.
    
    Args:
        model_flops_per_sec: Actual FLOPS achieved by model
        peak_flops_per_sec: Theoretical peak FLOPS of hardware
        
    Returns:
        mfu: Model FLOPS Utilization as percentage
    """
    return (model_flops_per_sec / peak_flops_per_sec) * 100

def calculate_model_parameters(model):
    """
    Calculate the total number of parameters and trainable parameters in the model.
    
    Args:
        model: PyTorch model
        
    Returns:
        tuple: (total_params, trainable_params)
    """
    # total_params = sum(p.numel() for p in model.parameters())
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Use named_parameters to avoid double counting tied weights
    unique_params = {}
    for name, param in model.named_parameters():
        # Use the parameter's data_ptr as a unique identifier
        param_id = param.data_ptr()
        if param_id not in unique_params:
            unique_params[param_id] = param
    
    total_params = sum(p.numel() for p in unique_params.values())
    trainable_params = sum(p.numel() for p in unique_params.values() if p.requires_grad)
    
    return total_params, trainable_params

def print_model_info(model):
    """
    Print detailed information about the model including parameter counts and FLOPS info.
    
    Args:
        model: PyTorch model
    """
    total_params, trainable_params = calculate_model_parameters(model)
    model_flops_per_token = calculate_model_flops_per_token(model)
    
    print0(f"Model parameter summary:")
    print0(f"  Total parameters: {total_params:,}")
    print0(f"  Trainable parameters: {trainable_params:,}")
    print0(f"  Non-trainable parameters: {total_params - trainable_params:,}")
    print0(f"  Model size (float32): {total_params * 4 / 1024**2:.2f} MB")
    print0(f"  Model size (bfloat16): {total_params * 2 / 1024**2:.2f} MB")
    print0(f"  Model size: {total_params / 1e6:.2f} Million parameters")
    print0(f"  Model FLOPs per token: {model_flops_per_token:,} ({model_flops_per_token / 1e9:.2f} GFLOPs)")

    
    # Print parameter breakdown by module type
    # print0(f"\nParameter breakdown by module:")
    # for name, module in model.named_modules():
    #     if len(list(module.children())) == 0:  # leaf modules only
    #         module_params = sum(p.numel() for p in module.parameters())
    #         if module_params > 0:
    #             print0(f"  {name}: {module_params:,} parameters")