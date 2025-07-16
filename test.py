import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

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
    max_batch_size: int = 1  # 8
    max_seq_len: int = 512  # 4096 * 4
    dtype: Literal["bf16", "fp8"] = "bf16"
    vocab_size: int = 50257  # 102400

    # Model size reduced
    dim: int = 256  # 2048
    inter_dim: int = 512  # 10944
    moe_inter_dim: int = 64  # 1408
    n_layers: int = 2  # 12
    n_dense_layers: int = 1
    n_heads: int = 4  # 16

    # MoE params reduced
    n_routed_experts: int = 4  # 64
    n_shared_experts: int = 1  # 2
    n_activated_experts: int = 2  # 6
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.

    # MLA
    q_lora_rank: int = 0
    kv_lora_rank: int = 64  # 512
    qk_nope_head_dim: int = 32  # 128
    qk_rope_head_dim: int = 16  # 64
    v_head_dim: int = 32  # 128

    # YARN
    original_seq_len: int = 512  # 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 8  # 32
    beta_slow: int = 1
    mscale: float = 1.


def precompute_freqs_cis(dim, max_seq_len, theta=10000.0):
    """
    Returns a tensor of complex frequencies for rotary embeddings.
    Shape: (max_seq_len, dim // 2)
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # (seq_len, dim // 2)
    return torch.polar(torch.ones_like(freqs), freqs).to(torch.cfloat)  # (seq_len, dim // 2)




# normalization layer
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Args:
        dim (int): Dimension of the input tensor.
        eps (float): Epsilon value for numerical stability. Defaults to 1e-6.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        """
        Forward pass for RMSNorm.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.
        """
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)


def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Applies a linear transformation to the incoming data: y = xA^T + b.
    This function supports specialized implementations based on quantization
    and tensor formats.

    Args:
        x (torch.Tensor): The input tensor.
        weight (torch.Tensor): The weight tensor. It may be quantized and 
            requires dequantization for certain cases.
        bias (Optional[torch.Tensor]): The bias tensor to be added. Default is None.

    Returns:
        torch.Tensor: The result of the linear transformation, which may involve 
        quantization-aware computations depending on the input parameters.

    Notes:
        - If `weight` is quantized (e.g., `element_size() == 1`), a dequantized version 
          is used for computation.
        - If `gemm_impl == "bf16"`, dequantization and a `bf16` GEMM operation are applied.
        - For other cases, the function applies quantization to `x` and uses `fp8_gemm` for computation.
    """
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
    """
    Linear layer with column parallelism, splitting output features across distributed processes.

    Args:
        in_features (int): Number of input features.
        out_features (int): Total number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert out_features % world_size == 0, f"Output features must be divisible by world size (world_size={world_size})"
        self.part_out_features = out_features // world_size
        super().__init__(in_features, self.part_out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for column parallel linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor with column-parallel computation.
        """
        y = linear(x, self.weight, self.bias)
        return y
    


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.

    Args:
        x (torch.Tensor): Input tensor with positional embeddings to be applied.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for positional embeddings.

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied.
    """
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)



# MLA (Multi-Headed Linear Attention) layer
class MLA(nn.Module):
    """
    Multi-Head Latent Attention (MLA) Layer.

    Attributes:
        dim (int): Dimensionality of the input features.
        n_heads (int): Number of attention heads.
        n_local_heads (int): Number of local attention heads for distributed systems.
        q_lora_rank (int): Rank for low-rank query projection.
        kv_lora_rank (int): Rank for low-rank key/value projection.
        qk_nope_head_dim (int): Dimensionality of non-positional query/key projections.
        qk_rope_head_dim (int): Dimensionality of rotary-positional query/key projections.
        qk_head_dim (int): Total dimensionality of query/key projections.
        v_head_dim (int): Dimensionality of value projections.
        softmax_scale (float): Scaling factor for softmax in attention computation.
    """
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
        self.softmax_scale = self.qk_head_dim ** -0.5
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        # Fix: Initialize cache with the correct dtype
        cache_dtype = Linear.dtype  # Use the same dtype as Linear layers
        if attn_impl == "naive":
            self.register_buffer("k_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.qk_head_dim, dtype=cache_dtype), persistent=False)
            self.register_buffer("v_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.v_head_dim, dtype=cache_dtype), persistent=False)
        else:
            self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank, dtype=cache_dtype), persistent=False)
            self.register_buffer("pe_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim, dtype=cache_dtype), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int = 0, freqs_cis: torch.Tensor = None, mask: Optional[torch.Tensor] = None):
        """
        Forward pass for the Multi-Head Latent Attention (MLA) Layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            start_pos (int): Starting position in the sequence for caching.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        
        # Store the original dtype for consistency
        original_dtype = x.dtype
        
        # Ensure cache tensors have the correct dtype
        if attn_impl == "naive":
            if self.k_cache.dtype != original_dtype:
                self.k_cache = self.k_cache.to(original_dtype)
                self.v_cache = self.v_cache.to(original_dtype)
        else:
            if self.kv_cache.dtype != original_dtype:
                self.kv_cache = self.kv_cache.to(original_dtype)
                self.pe_cache = self.pe_cache.to(original_dtype)
        
        # Resize cache if needed
        if attn_impl == "naive":
            if bsz > self.k_cache.size(0):
                self.k_cache = self.k_cache.resize_(bsz, self.k_cache.size(1), self.k_cache.size(2), self.k_cache.size(3))
                self.v_cache = self.v_cache.resize_(bsz, self.v_cache.size(1), self.v_cache.size(2), self.v_cache.size(3))
        else:
            if bsz > self.kv_cache.size(0):
                self.kv_cache = self.kv_cache.resize_(bsz, self.kv_cache.size(1), self.kv_cache.size(2))
                self.pe_cache = self.pe_cache.resize_(bsz, self.pe_cache.size(1), self.pe_cache.size(2))
        
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
            self.k_cache[:bsz, start_pos:end_pos] = k.to(original_dtype)
            self.v_cache[:bsz, start_pos:end_pos] = v.to(original_dtype)
            scores = torch.einsum("bshd,bthd->bsht", q, self.k_cache[:bsz, :end_pos]) * self.softmax_scale
        else:
            wkv_b = self.wkv_b.weight if self.wkv_b.scale is None else weight_dequant(self.wkv_b.weight, self.wkv_b.scale, block_size)
            wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
            q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
            
            # Ensure dtype consistency for cache operations
            kv_norm = self.kv_norm(kv).to(original_dtype)
            k_pe_squeezed = k_pe.squeeze(2).to(original_dtype)
            
            # Fix: Correct slice syntax for cache assignment
            self.kv_cache[:bsz, start_pos:end_pos, :] = kv_norm
            self.pe_cache[:bsz, start_pos:end_pos, :] = k_pe_squeezed
            
            # Cast to float32 for einsum to avoid type mismatch
            scores = (torch.einsum("bshc,btc->bsht", q_nope.to(torch.float32), self.kv_cache[:bsz, :end_pos].to(torch.float32)) +
                    torch.einsum("bshr,btr->bsht", q_pe.to(torch.float32), self.pe_cache[:bsz, :end_pos].to(torch.float32))) * self.softmax_scale
            scores = scores.to(original_dtype)  # Cast back to bf16
        
        if mask is not None:
            scores += mask.unsqueeze(1)
        
        # Apply softmax and ensure dtype consistency
        scores = scores.softmax(dim=-1, dtype=torch.float32).to(original_dtype)
        
        if attn_impl == "naive":
            x = torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:bsz, :end_pos])
        else:
            x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:].to(original_dtype))
        
        x = self.wo(x.flatten(2))
        return x

class RowParallelLinear(Linear):
    """
    Linear layer with row parallelism, splitting input features across distributed processes.

    Args:
        in_features (int): Total number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert in_features % world_size == 0, f"Input features must be divisible by world size (world_size={world_size})"
        self.part_in_features = in_features // world_size
        super().__init__(self.part_in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for row parallel linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor with row-parallel computation.
        """
        y = linear(x, self.weight)
        if world_size > 1:
            dist.all_reduce(y)
        if self.bias is not None:
            y += self.bias
        return y




# decoder only block for MLA and MLP
# Fixed Block class - remove the incorrectly placed MLA logic
class Block(nn.Module):
    """
    Transformer block combining attention and feed-forward layers.

    Attributes:
        attn (nn.Module): Attention layer (MLA).
        ffn (nn.Module): Feed-forward network (MLP or MoE).
        attn_norm (nn.Module): Layer normalization for attention.
        ffn_norm (nn.Module): Layer normalization for feed-forward network.
    """
    def __init__(self, layer_id: int, args: ModelArgs):
        """
        Initializes the Transformer block.

        Args:
            layer_id (int): Layer index in the transformer.
            args (ModelArgs): Model arguments containing block parameters.
        """
        super().__init__()
        self.attn = MLA(args)
        self.ffn = MLP(args.dim, args.inter_dim) if layer_id < args.n_dense_layers else MoE(args)
        self.attn_norm = RMSNorm(args.dim)
        self.ffn_norm = RMSNorm(args.dim)

    def forward(self, x: torch.Tensor, start_pos: int = 0, freqs_cis: torch.Tensor = None, mask: Optional[torch.Tensor] = None):
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            start_pos (int): Starting position in the sequence for caching.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        # Attention with residual connection
        x = x + self.attn(self.attn_norm(x), start_pos, freqs_cis, mask)
        
        # Feed-forward with residual connection
        x = x + self.ffn(self.ffn_norm(x))
        
        return x
    


    
class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) used as a feed-forward layer.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """
    def __init__(self, dim: int, inter_dim: int):
        """
        Initializes the MLP layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.w1 = ColumnParallelLinear(dim, inter_dim)
        self.w2 = RowParallelLinear(inter_dim, dim)
        self.w3 = ColumnParallelLinear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLP layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after MLP computation.
        """
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
    """
    Expert layer for Mixture-of-Experts (MoE) models.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """
    def __init__(self, dim: int, inter_dim: int):
        """
        Initializes the Expert layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.w1 = Linear(dim, inter_dim)
        self.w2 = Linear(inter_dim, dim)
        self.w3 = Linear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Expert layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert computation.
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
    """
    Mixture-of-Experts (MoE) module.

    Attributes:
        dim (int): Dimensionality of input features.
        n_routed_experts (int): Total number of experts in the model.
        n_local_experts (int): Number of experts handled locally in distributed systems.
        n_activated_experts (int): Number of experts activated for each input.
        gate (nn.Module): Gating mechanism to route inputs to experts.
        experts (nn.ModuleList): List of expert modules.
        shared_experts (nn.Module): Shared experts applied to all inputs.
    """
    def __init__(self, args: ModelArgs):
        """
        Initializes the MoE module.

        Args:
            args (ModelArgs): Model arguments containing MoE parameters.
        """
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MoE module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert routing and computation.
        """
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        y = torch.zeros_like(x)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        z = self.shared_experts(x)
        if world_size > 1:
            dist.all_reduce(y)
        return (y + z).view(shape)



# Transformer model class
# class Transformer(nn.Module):
#     def __init__(self, args: ModelArgs):
#         super().__init__()
#         self.args = args  #installizing all the model parameters in args

#         # ---------------------------------------------
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
#         # Output of this stack → sent to `lm_head` for logits
#         # ---------------------------------------------

#         self.transformer = nn.ModuleDict(dict(
#             tokens = nn.Embedding(self.args.vocab_size, self.args.dim),
#             blocks = nn.ModuleList([Block(i, self.args) for i in range(self.args.n_layers)]),
#             ln_f = RMSNorm(self.args.dim),
#         ))

#         self.lm_head = nn.Linear(self.args.dim, self.args.vocab_size, bias=False)

#     def forward(self, idx, targets=None):
#         B, T = idx.size()
#         assert T <= self.args.max_seq_len, f"Cannot forward sequence of length {T}, block size is only {self.args.max_seq_len}"
        
#         # Step 1: Embedding
#         tok_emb = self.transformer.tokens(idx)  # (B, T, dim)
#         x = tok_emb

#         # Step 2: Precompute RoPE frequencies
#         freqs_cis = precompute_freqs_cis(
#             dim=self.args.qk_rope_head_dim, 
#             max_seq_len=self.args.max_seq_len, 
#             theta=self.args.rope_theta
#         ).to(x.device)[:T]  # (T, dim//2)

#         # Step 3: Create causal mask
#         mask = torch.full((T, T), float("-inf"), device=x.device)
#         mask = torch.triu(mask, diagonal=1)  # Causal mask: prevent attending to future tokens

#         # Step 4: Set starting position
#         start_pos = 0

#         # Step 5: Pass through all blockss
#         for block in self.transformer.blocks:
#             x = block(x, start_pos, freqs_cis, mask)

#         # Final norm and output
#         x = self.transformer.ln_f(x)
#         logits = self.lm_head(x)
#         return logits


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
        nn.init.xavier_uniform_(self.lm_head.weight)  # Initialize lm_head weights

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
        return logits

# testing 
model_args = ModelArgs()
model = Transformer(model_args)
# Use Accelerate to split across devices
from accelerate import infer_auto_device_map, dispatch_model
import tiktoken

# device_map = infer_auto_device_map(model, max_memory={0: "4GiB", "cpu": "32GiB"}, no_split_module_classes=["Block"])
# model = dispatch_model(model)

model.to('cuda')
model.eval()


num_return_sequence = 5
max_length = 30



# Tokenizer
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("hello, this is an LLM ")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequence, 1)

# Choose device for input (use device of first module)
first_device = next(model.parameters()).device  
x = tokens.to(first_device)


# Sampling loop
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x)
        logits = logits[:, -1, :]
        
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("Invalid logits detected:", logits)
            break
        
        logits = logits - torch.max(logits, dim=-1, keepdim=True)[0]
        temperature = 1.0
        probs = F.softmax(logits / temperature, dim=-1)
        
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            print("Invalid probs detected:", probs)
            break
        
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        if torch.isnan(topk_probs).any() or torch.isinf(topk_probs).any():
            print("Invalid topk_probs detected:", topk_probs)
            break
        
        ix = torch.multinomial(topk_probs, 1)
        xcol = torch.gather(topk_indices, -1, ix)
        x = torch.cat((x, xcol), dim=1)

for i in range(num_return_sequence):
    tokens = x[i, :max_length].tolist()
    print(">", enc.decode(tokens))