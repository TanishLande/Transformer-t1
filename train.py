import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import inspect

from hellaswag import render_example, iterate_examples

from dataclasses import dataclass

import math
from typing import Tuple, Optional, Literal


from kernal import act_quant, weight_dequant ,fp8_gemm


world_size = 1
rank = 0
block_size = 128
gemm_impl: Literal["bf16", "fp8"] = "bf16"
attn_impl: Literal["naive", "absorb"] = "absorb"



# ModelArgsuration class for model parameters
@dataclass
class ModelArgs:
    max_batch_size: int = 64  # 8
    max_seq_len: int = 1024  # 4096 * 4
    dtype: Literal["bf16", "fp8"] = "bf16"
    vocab_size: int = 50304  # 102400

    # Model size reduced
    dim: int = 512  # 2048
    inter_dim: int = 5048  # 10944
    moe_inter_dim: int = 256  # 1408
    n_layers: int = 6  # 12
    n_dense_layers: int = 2
    n_heads: int = 8  # 16

    # MoE params reduced
    n_routed_experts: int = 8  # 64
    n_shared_experts: int = 2  # 2
    n_activated_experts: int = 2  # 6
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.

    # MLA
    q_lora_rank: int = 0
    kv_lora_rank: int = 128  # 512
    qk_nope_head_dim: int = 64  # 128
    qk_rope_head_dim: int = 32  # 64
    v_head_dim: int = 64  # 128

    # YARN
    original_seq_len: int = 1024  # 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32  # 32
    beta_slow: int = 1
    mscale: float = 1.


def precompute_freqs_cis(dim, max_seq_len, theta=10000.0):

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # (seq_len, dim // 2)
    return torch.polar(torch.ones_like(freqs), freqs).to(torch.cfloat)  # (seq_len, dim // 2)




# normalization layer
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)


def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    if x.dtype != weight.dtype:
        x = x.to(weight.dtype)

    if weight.element_size() > 1:
        return F.linear(x, weight, bias)
    elif gemm_impl == "bf16":
        weight = weight_dequant(weight, weight.scale)
        return F.linear(x, weight, bias)
    else:
        x, scale = act_quant(x, block_size)
        y = fp8_gemm(x, scale, weight, weight.scale)
        if bias is not None:
            y += bias
        return y
    

class Linear(nn.Module):
    dtype = torch.bfloat16

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype or Linear.dtype))
        # Initialize weights using Xavier initialization
        nn.init.xavier_uniform_(self.weight)
        if self.weight.element_size() == 1:
            scale_out_features = (out_features + block_size - 1) // block_size
            scale_in_features = (in_features + block_size - 1) // block_size
            self.weight.scale = self.scale = nn.Parameter(torch.ones(scale_out_features, scale_in_features, dtype=torch.float32))
        else:
            self.register_parameter("scale", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
            nn.init.zeros_(self.bias)  # Initialize bias to zeros
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.weight, self.bias)


class ColumnParallelLinear(Linear):

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert out_features % world_size == 0, f"Output features must be divisible by world size (world_size={world_size})"
        self.part_out_features = out_features // world_size
        super().__init__(in_features, self.part_out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = linear(x, self.weight, self.bias)
        return y
    


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:

    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)


class MLA(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // world_size
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim

        if self.q_lora_rank == 0:
            self.wq = ColumnParallelLinear(self.dim, self.n_heads * self.qk_head_dim)
        else:
            self.wq_a = Linear(self.dim, self.q_lora_rank)
            self.q_norm = RMSNorm(self.q_lora_rank)
            self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.qk_head_dim)
        
        self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = ColumnParallelLinear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))
        self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)
        self.wo.NANOGPT_SCALE_INIT = 1 #  # Set scale initialization for c_proj(flag)
        self.softmax_scale = self.qk_head_dim ** -0.5
        
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        # FIX 1: Don't use buffers for cache - they maintain gradients
        # Instead, initialize as None and create fresh tensors when needed
        self.k_cache = None
        self.v_cache = None
        self.kv_cache = None
        self.pe_cache = None

    def forward(self, x: torch.Tensor, start_pos: int = 0, freqs_cis: torch.Tensor = None, mask: Optional[torch.Tensor] = None):
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        
        # Store the original dtype for consistency
        original_dtype = x.dtype
        
        # FIX: Always create fresh cache tensors to prevent memory accumulation
        if attn_impl == "naive":
            self.k_cache = torch.zeros(bsz, end_pos, self.n_local_heads, self.qk_head_dim, 
                                    dtype=original_dtype, device=x.device)
            self.v_cache = torch.zeros(bsz, end_pos, self.n_local_heads, self.v_head_dim, 
                                    dtype=original_dtype, device=x.device)
        else:
            self.kv_cache = torch.zeros(bsz, end_pos, self.kv_lora_rank, 
                                    dtype=original_dtype, device=x.device)
            self.pe_cache = torch.zeros(bsz, end_pos, self.qk_rope_head_dim, 
                                    dtype=original_dtype, device=x.device)
        
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
        
        if attn_impl == "naive":
            q = torch.cat([q_nope, q_pe], dim=-1)
            kv = self.wkv_b(self.kv_norm(kv))
            kv = kv.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
            
            # Use detach() to prevent gradient flow to cache
            self.k_cache[:bsz, start_pos:end_pos] = k.detach()
            self.v_cache[:bsz, start_pos:end_pos] = v.detach()
            
            scores = torch.einsum("bshd,bthd->bsht", q, self.k_cache[:bsz, :end_pos]) * self.softmax_scale
        else:
            wkv_b = self.wkv_b.weight if self.wkv_b.scale is None else weight_dequant(self.wkv_b.weight, self.wkv_b.scale, block_size)
            wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
            q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
            
            kv_norm = self.kv_norm(kv)
            k_pe_squeezed = k_pe.squeeze(2)
            
            # FIX: Use detach() and ensure consistent dtype
            self.kv_cache[:bsz, start_pos:end_pos, :] = kv_norm.detach().to(original_dtype)
            self.pe_cache[:bsz, start_pos:end_pos, :] = k_pe_squeezed.detach().to(original_dtype)
            
            # FIX: Ensure all tensors have the same dtype for einsum
            scores = (torch.einsum("bshc,btc->bsht", q_nope.to(original_dtype), self.kv_cache[:bsz, :end_pos]) +
                    torch.einsum("bshr,btr->bsht", q_pe.to(original_dtype), self.pe_cache[:bsz, :end_pos])) * self.softmax_scale
        
        if mask is not None:
            scores += mask.unsqueeze(1)
        
        # Apply softmax with consistent dtype
        scores = scores.softmax(dim=-1, dtype=torch.float32).to(original_dtype)
        
        if attn_impl == "naive":
            x = torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:bsz, :end_pos])
        else:
            x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:].to(original_dtype))
        
        x = self.wo(x.flatten(2))
        return x


class RowParallelLinear(Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert in_features % world_size == 0, f"Input features must be divisible by world size (world_size={world_size})"
        self.part_in_features = in_features // world_size
        super().__init__(self.part_in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        y = linear(x, self.weight)
        if world_size > 1:
            dist.all_reduce(y)
        if self.bias is not None:
            y += self.bias
        return y




# decoder only block for MLA and MLP
# Fixed Block class - remove the incorrectly placed MLA logic
class Block(nn.Module):

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.attn = MLA(args)
        self.ffn = MLP(args.dim, args.inter_dim) if layer_id < args.n_dense_layers else MoE(args)
        self.attn_norm = RMSNorm(args.dim)
        self.ffn_norm = RMSNorm(args.dim)

    def forward(self, x: torch.Tensor, start_pos: int = 0, freqs_cis: torch.Tensor = None, mask: Optional[torch.Tensor] = None):
        # Attention with residual connection
        x = x + self.attn(self.attn_norm(x), start_pos, freqs_cis, mask)
        
        # Feed-forward with residual connection
        x = x + self.ffn(self.ffn_norm(x))
        
        return x
    


    
class MLP(nn.Module):
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = ColumnParallelLinear(dim, inter_dim)
        self.w2 = RowParallelLinear(inter_dim, dim)
        self.w3 = ColumnParallelLinear(dim, inter_dim)
        self.w2.NANOGPT_SCALE_INIT = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
    

class Gate(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.n_groups = args.n_expert_groups
        self.topk_groups = args.n_limited_groups
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        nn.init.xavier_uniform_(self.weight)  # Initialize gate weights
        self.bias = nn.Parameter(torch.empty(args.n_routed_experts)) if self.dim == 7168 else None
        if self.bias is not None:
            nn.init.zeros_(self.bias)  # Initialize bias to zeros

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = linear(x, self.weight)
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            if self.bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(1, indices, False)
            scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights.type_as(x), indices
    


class Expert(nn.Module):
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = Linear(dim, inter_dim)
        self.w2 = Linear(inter_dim, dim)
        self.w3 = Linear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        assert args.n_routed_experts % world_size == 0, f"Number of experts must be divisible by world size (world_size={world_size})"
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts // world_size
        self.n_activated_experts = args.n_activated_experts
        self.experts_start_idx = rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = Gate(args)
        self.experts = nn.ModuleList([Expert(args.dim, args.moe_inter_dim) if self.experts_start_idx <= i < self.experts_end_idx else None
                                      for i in range(self.n_routed_experts)])
        self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim)
        self.shared_experts.NANOGPT_SCALE_INIT = 1

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     shape = x.size()
    #     x = x.view(-1, self.dim)
        
    #     # FIX: Add bounds checking for indices
    #     weights, indices = self.gate(x)
        
    #     # Clamp indices to valid range to prevent out-of-bounds access
    #     indices = torch.clamp(indices, 0, self.n_routed_experts - 1)
        
    #     y = torch.zeros_like(x)
    #     counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        
    #     for i in range(self.experts_start_idx, self.experts_end_idx):
    #         if counts[i] == 0:
    #             continue
    #         expert = self.experts[i]
    #         if expert is None:  # Safety check
    #             continue
                
    #         # FIX: Add error handling for expert computation
    #         try:
    #             idx, top = torch.where(indices == i)
    #             if len(idx) > 0:  # Only process if there are tokens for this expert
    #                 expert_output = expert(x[idx])
    #                 y[idx] += expert_output * weights[idx, top, None]
    #         except RuntimeError as e:
    #             print(f"Error in expert {i}: {e}")
    #             continue
        
    #     z = self.shared_experts(x)
    #     if world_size > 1:
    #         dist.all_reduce(y)
    #     return (y + z).view(shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)
        
        # Get weights and indices from gate
        weights, indices = self.gate(x)
        
        # Debug: Check indices for validity
        if torch.any(indices < 0) or torch.any(indices >= self.n_routed_experts):
            print(f"Invalid indices detected: min={indices.min().item()}, max={indices.max().item()}, n_routed_experts={self.n_routed_experts}")
            indices = torch.clamp(indices, 0, self.n_routed_experts - 1)
        
        # Ensure indices are on the correct device
        indices = indices.to(x.device)
        assert indices.device == x.device, f"Indices device ({indices.device}) does not match input device ({x.device})"
        
        y = torch.zeros_like(x)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        torch.cuda.synchronize()  # Ensure bincount is complete
        
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            if expert is None:
                continue
                
            try:
                idx, top = torch.where(indices == i)
                if len(idx) > 0:
                    expert_input = x[idx].contiguous()
                    expert_output = expert(expert_input)
                    y[idx] += expert_output * weights[idx, top, None]
                    del expert_input, expert_output
                    torch.cuda.synchronize()
            except RuntimeError as e:
                print(f"Error in expert {i}: {e}")
                continue
        
        z = self.shared_experts(x)
        if world_size > 1:
            torch.cuda.synchronize()
            dist.all_reduce(y)
        return (y + z).view(shape)



#         # Diagram of `self.transformer` Architecture
#         #
#         #           ┌────────────────────────────┐
#         #           │  Token IDs (Input)         │
#         #           └────────────┬───────────────┘
#         #                        ↓
#         #           ┌────────────────────────────┐
#         #           │  tokens                    │
#         #           │  nn.Embedding              │
#         #           │  [vocab_size × dim]        │
#         #           └────────────┬───────────────┘
#         #                        ↓
#         #        ┌────────────────────────────────────┐
#         #        │ blocks -(n_layers)                 │
#         #        │ nn.ModuleList of Transformer Blocks│
#         #        │ [Block1 → Block2 → ... → BlockN]   │
#         #        └────────────┬───────────────────────┘
#         #                     ↓
#         #        ┌────────────────────────────────────┐
#         #        │  ln_f                              │
#         #        │  RMSNorm(dim)                      │
#         #        └────────────────────────────────────┘
#         #

class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.transformer = nn.ModuleDict(dict(
            tokens=nn.Embedding(self.args.vocab_size, self.args.dim),
            blocks=nn.ModuleList([Block(i, self.args) for i in range(self.args.n_layers)]),
            ln_f=RMSNorm(self.args.dim),
        ))
        # Initialize embedding weights
        nn.init.normal_(self.transformer.tokens.weight, mean=0.0, std=0.02)
        self.lm_head = nn.Linear(self.args.dim, self.args.vocab_size, bias=False)
        # nn.init.xavier_uniform_(self.lm_head.weight)  # Initialize lm_head weights
        self.transformer.tokens.weight = self.lm_head.weight  # Tie weights between token embedding and output layer

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.args.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.args.max_seq_len, f"Cannot forward sequence of length {T}, block size is only {self.args.max_seq_len}"
        tok_emb = self.transformer.tokens(idx)
        x = tok_emb
        freqs_cis = precompute_freqs_cis(
            dim=self.args.qk_rope_head_dim,
            max_seq_len=self.args.max_seq_len,
            theta=self.args.rope_theta
        ).to(x.device)[:T]
        mask = torch.full((T, T), float("-inf"), device=x.device)
        mask = torch.triu(mask, diagonal=1)
        start_pos = 0
        for block in self.transformer.blocks:
            x = block(x, start_pos, freqs_cis, mask)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None 
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss


    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
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
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer



# -----------------------------------------------------------------
# testing 
# Run using this below command
# torchrun --standalone --nproc_per_node=8 train.py
import tiktoken
import time
from lion_pytorch import Lion
import os
import numpy as np

# DDP import 
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


# Load shards for fineWeb-Edu
def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


# dataloader which divides input token in multiple batches
class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

#-----------------------------------------------------------------------------------

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])  #   global rank of this process (Rank for a perticular GPU)
    ddp_local_rank = int(os.environ['LOCAL_RANK'])  #  local rank of this process (Rank for a perticular GPU in a node)
    ddp_world_size = int(os.environ['WORLD_SIZE'])  #   total number of processes or GPUs
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else: #no DPP simple one GPU run 
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")


# -------------------------------------------------

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm



# --------------------------------------------------

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)
device_type = "cuda" if torch.cuda.is_available() else "cpu" 

# Tokenizer
enc = tiktoken.get_encoding('gpt2')
vocab_size = enc.n_vocab

# print("Total number of tokens in GPT-2 tokenizer:", vocab_size)

# with open("input.txt", "r") as f:
#     text = f.read()
# text = text[:1000]
# tokens = enc.encode(text)
# B, T = 4, 32
# buf = torch.tensor(tokens[:B*T + 1])
# buf = buf.to(device)
# x = buf[:-1].view(B, T).to(device)  # input tokens
# y = buf[1:].view(B, T).to(device)   # target tokens

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# grad accum steps -> after some backward steps, we will update the weights to do high number of batches
total_batch_size  = 524288
B = 64
T = 1024
assert total_batch_size % (B * T * ddp_world_size) == 0, "Total batch size must be divisible by B * T * ddp_world_size"
grads_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"calculated grad accum steps: {grads_accum_steps}")



torch.set_float32_matmul_precision('high')

model_args = ModelArgs()
model = Transformer(model_args)
model.to(device)

# model = torch.compile(model)
use_compile = False # torch.compile interferes with HellaSwag eval and Generation. TODO fix
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


if ddp:
    model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)
raw_model = model.module if ddp else model  # get the raw model for optimizer configuration


train_loader = DataLoaderLite(B=4, T=524, process_rank=ddp_rank, num_processes=ddp_world_size, split="train") 
val_loader = DataLoaderLite(B=4, T=524, process_rank=ddp_rank, num_processes=ddp_world_size, split="val") 

# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
# optimizer = Lion(model.parameters(), lr=3e-4, betas=(0.9, 0.99))
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device)


#create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass


max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)  # how far you are from decay phase 
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

loss_accum = torch.tensor(0.0, device=device, requires_grad=False)

# Fixed training loop - move forward pass inside the loop
for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps -1)

    if step % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits , loss = model(x,y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validations loss: {val_loss_accum.item():.4f}" )

    # sample pool
    if step > 0 and step % 500 == 0 and master_process:
        model.eval()
        with torch.no_grad():
            num_return_sequences = 2
            max_length = 30
            tokens = enc.encode("Hello, this is an LLM ")
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
            torch.manual_seed(42 + step)
            x = tokens
            while x.size(1) < max_length:
                logits, _ = model(x)
                logits = logits[:, -1, :]
                logits = logits - torch.max(logits, dim=-1, keepdim=True)[0]
                probs = F.softmax(logits / 1.0, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                ix = torch.multinomial(topk_probs, 1)
                xcol = torch.gather(topk_indices, -1, ix)
                x = torch.cat((x, xcol), dim=1)
            for i in range(num_return_sequences):
                generated = enc.decode(x[i, :max_length].tolist())
                print(f"Step {step} Sample {i}: {generated}")
                with open(os.path.join(log_dir, "samples.txt"), "a") as f:
                    f.write(f"Step {step} Sample {i}: {generated}\n")
        model.train()

     # once in a while evaluate hellaswag
    if (step % 250 == 0 or last_step) and (not use_compile):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")


    # saving model 
    if step % 1000 == 0 and master_process:
        checkpoint_path = os.path.join(log_dir, f"model_step_{step}.pt")
        torch.save({
            'step': step,
            'model_state_dict': raw_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_accum.item(),
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

    if step % 50 == 0:
                torch.cuda.empty_cache()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

    # traning loop
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grads_accum_steps):
        x, y = train_loader.next_batch()
        x = x.to(device)
        y = y.to(device)
        with torch.autocast(device_type=device_type , dtype=torch.bfloat16):
            logits, loss = model(x, y)  # Move this INSIDE the loop
        # import code; code.interact(local=locals)
        loss = loss / grads_accum_steps  # Scale loss for gradient accumulation
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_step == grads_accum_steps - 1)  # Sync gradients only on the last micro-step
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.SUM)  # Sum losses across all processes
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # How much big step will it take || how much it will learn 
    # learning rate scheduler
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()  # Ensure all operations are complete before measuring time
    t1 = time.time()
    perplexity = torch.exp(loss)    # How much the model is confused(1 is most confident, 1000 is most confused)
    dt = (t1 - t0)*1000
    tokens_processed = train_loader.B * train_loader.T * grads_accum_steps  * ddp_world_size  # total tokens processed in this step
    tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)  # tokens per second
    if master_process:
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f} |  perplexity: {perplexity:.4} ")
        with open(log_file, "a") as f:
            f.write(f"step: {step} | train_accum_loss: {loss_accum.item():.6f} | dt: {dt*1000:.2f}ms  | perplexity: {perplexity:.4} | tok/sec: {tokens_per_sec:.2f}  \n")

if ddp:
    destroy_process_group() # Clean up DDP environment

import sys
sys.exit(0)
