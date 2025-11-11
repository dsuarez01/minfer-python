from __future__ import annotations

from minfer.kernels import KernelBackend
from minfer.utils import GGUFReaderWrapper, Params, BufPool, TensorData, LUT

import torch
import torch.nn as nn

class Model(nn.Module):
    
    blocks: nn.ModuleList
    
    params: Params
    kerns: KernelBackend
    bufs: BufPool
    luts: LUT
    compute_logits: bool

    output_norm_td: TensorData
    output_td: TensorData
    token_embd_td: TensorData

    def __init__(self, path: str, run_params: dict) -> None:
        reader = GGUFReaderWrapper(path)
        self.params = Params(reader, run_params)
        self.kerns = KernelBackend(self.params.backend)
        self.bufs = BufPool(self.params)
        self.luts = LUT(self.params.act_dtype)
        self.compute_logits = False

        # TensorData attrs
        self.token_embd_td = reader.get_tensor_req("token_embd.weight")
        self.register_buffer(self.token_embd_td.name, self.token_embd_td.data)

        output_td = reader.get_tensor_noreq("output.weight")
        if output_td:
            self.output_td = output_td
        else:
            # tied wrd embd, just clone token_embd data
            self.output_td = TensorData(
                name="output.weight",
                type=self.token_embd_td.type,
                shape=self.token_embd_td.shape,
                n_elements=self.token_embd_td.n_elements,
                n_bytes=self.token_embd_td.n_bytes,
                data=self.token_embd_td.data.clone()
            )
        self.register_buffer(self.output_td.name, self.output_td.data)

        self.output_norm_td = reader.get_tensor_req("output_norm.weight")
        self.register_buffer(self.output_norm_td.name, self.output_norm_td.data)


    @torch.compile(mode="reduce-overhead")
    def forward(self, tokens: torch.Tensor, pos: int) -> None:
        pass

class Transformer(nn.Module):
    block_idx: int
    is_moe: bool
    model: Model
    
    def __init__(self, block_idx: int, reader: GGUFReaderWrapper, params: Params, model: Model) -> None:
        super().__init__()
        self.block_idx = block_idx
        self.is_moe = params.n_exps > 0
        self.model = model
        
        # register k,v caches
        self.register_buffer(
            "k_cache", 
            torch.zeros(
                (params.batch_size, params.n_kv_heads, params.max_seq_len, params.head_dim), 
                dtype=params.act_dtype,
            )
        )
        
        self.register_buffer(
            "v_cache",
            torch.zeros(
                (params.batch_size, params.n_kv_heads, params.max_seq_len, params.head_dim), 
                dtype=params.act_dtype,
            )
        )

        # TODO: reg doesn't handle noreq tensors, reconsider
        def reg(attr_name: str, tensor_name: str):
            td = reader.get_tensor_req(tensor_name)
            setattr(self, f"{attr_name}_td", td)
            self.register_buffer(f"{attr_name}_weight", td.data) # note: created on CPU
        
        # attn (TODO: add support for q_norm and k_norm)
        # the qkv need to be fused into one
        q_td = reader.get_tensor_req(f"blk.{block_idx}.attn_q.weight")
        k_td = reader.get_tensor_req(f"blk.{block_idx}.attn_k.weight")

        if q_td.type == k_td.type:
            self.fused_qk = True
            qk_data = torch.cat([q_td.data, k_td.data], dim=0)
            self.register_buffer("attn_qk_weight", qk_data)
            self.attn_qk_td = TensorData(
                name=f"blk.{block_idx}.attn_qk.weight",
                type=q_td.type,
                shape=(2*q_td.shape[0], q_td.shape[1]),  # double d_out
                n_elements=q_td.n_elements+k_td.n_elements,
                n_bytes=q_td.n_bytes+k_td.n_bytes,
                data=qk_data
            )
        else:
            self.fused_qk = False
            setattr(self, "attn_q_td", q_td)
            setattr(self, "attn_k_td", k_td)
            self.register_buffer("attn_q_weight", q_td.data)
            self.register_buffer("attn_k_weight", k_td.data)

        reg("attn_v", f"blk.{block_idx}.attn_v.weight")
        reg("attn_output", f"blk.{block_idx}.attn_output.weight")
        reg("attn_norm", f"blk.{block_idx}.attn_norm.weight")
        
        # ffn
        reg("ffn_norm", f"blk.{block_idx}.ffn_norm.weight")
        
        # TODO: expert-level parallelism? (doesn't fit current design)
        if self.is_moe:
            reg("ffn_gate_inp", f"blk.{block_idx}.ffn_gate_inp.weight")
            reg("ffn_gate_exps", f"blk.{block_idx}.ffn_gate_exps.weight")
            reg("ffn_up_exps", f"blk.{block_idx}.ffn_up_exps.weight")
            reg("ffn_down_exps", f"blk.{block_idx}.ffn_down_exps.weight")
        else:
            reg("ffn_gate", f"blk.{block_idx}.ffn_gate.weight")
            reg("ffn_up", f"blk.{block_idx}.ffn_up.weight")
            reg("ffn_down", f"blk.{block_idx}.ffn_down.weight")
    

    @property
    def params(self) -> Params:
        return self.model.params

    @property
    def kerns(self) -> KernelBackend:
        return self.model.kerns

    @property
    def bufs(self) -> BufPool:
        return self.model.bufs
    
    @property
    def luts(self) -> LUT:
        return self.model.luts

    def forward(self, x: torch.Tensor, pos: int) -> None:
        pass