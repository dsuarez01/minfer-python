from __future__ import annotations

from minfer.kernels import KernelBackend
from minfer.utils import GGUFReaderWrapper, Params, BufPool, TensorData, LUT, GGMLQuantizationType

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
        super().__init__()
        reader = GGUFReaderWrapper(path)
        self.params = Params(reader, run_params)
        self.kerns = KernelBackend(self.params.backend)
        self.bufs = BufPool(self.params)
        self.luts = LUT(self.params.act_dtype)
        self.compute_logits = False

        # TensorData attrs
        self.token_embd_td = reader.get_tensor_req("token_embd.weight").register(parent=self)

        output_td = reader.get_tensor_noreq("output.weight")
        if output_td:
            self.output_td = output_td.register(parent=self)
        else:
            # tied wrd embd, just clone token_embd data
            self.output_td = TensorData(
                name="output.weight",
                type=self.token_embd_td.type,
                shape=self.token_embd_td.shape,
                n_elements=self.token_embd_td.n_elements,
                n_bytes=self.token_embd_td.n_bytes,
                data=self.token_embd_td.data.clone(),
                parent=self,
            ).register()

        self.output_norm_td = reader.get_tensor_req("output_norm.weight").register(parent=self)

        # block init
        for i in range(self.params.block_cnt):
            self.blocks.append(
                Transformer(
                    block_idx=i,
                    reader=reader,
                    params=self.params,
                    model=self,
                )
            )

    # @torch.compile(mode="reduce-overhead") TODO: unsure if this is useful for us
    def forward(self, tokens: torch.Tensor, pos: int) -> None:
        # embed
        # blocks
        # rmsnorm (optional?)
        # lm head
        pass

class Transformer(nn.Module):
    block_idx: int
    is_moe: bool
    fused_qk: bool
    model: Model
    
    def __init__(self, block_idx: int, reader: GGUFReaderWrapper, params: Params, model: Model) -> None:
        super().__init__()
        self.block_idx = block_idx
        self.is_moe = params.n_exps > 0
        self.model = model
        
        # k,v caches
        k_cache_data = torch.zeros((params.batch_size // params.dp_size, params.n_kv_heads, params.max_seq_len, params.head_dim), dtype=params.act_dtype)
        ggml_dtype = GGMLQuantizationType.F32 if params.act_dtype == torch.float32 else GGMLQuantizationType.F16
        self.k_cache = TensorData(
            name=f"blk.{self.block_idx}.k_cache",
            type=ggml_dtype,
            shape=k_cache_data.shape,
            n_elements=torch.numel(k_cache_data),
            n_bytes=k_cache_data.nbytes,
            data=k_cache_data,
            parent=self,
        ).register()
        
        v_cache_data = torch.zeros((params.batch_size // params.dp_size, params.n_kv_heads, params.max_seq_len, params.head_dim), dtype=params.act_dtype)
        self.v_cache = TensorData(
            name=f"blk.{self.block_idx}.v_cache",
            type=ggml_dtype,
            shape=v_cache_data.shape,
            n_elements=torch.numel(v_cache_data),
            n_bytes=v_cache_data.nbytes,
            data=v_cache_data,
            parent=self,
        ).register()
        
        # attn
        ## norms
        self.attn_norm = reader.get_tensor_req(f"blk.{block_idx}.attn_norm.weight").register(parent=self)
        q_norm_td = reader.get_tensor_noreq(f"blk.{block_idx}.attn_q_norm.weight")
        if q_norm_td: self.q_norm_td = q_norm_td.register(parent=self)

        k_norm_td = reader.get_tensor_noreq(f"blk.{block_idx}.attn_k_norm.weight")
        if k_norm_td: self.k_norm_td = k_norm_td.register(parent=self)

        ## projs
        q_td = reader.get_tensor_req(f"blk.{block_idx}.attn_q.weight")
        k_td = reader.get_tensor_req(f"blk.{block_idx}.attn_k.weight")

        if q_td.type == k_td.type:
            self.fused_qk = True
            qk_data = torch.cat([q_td.data, k_td.data], dim=0) # this should catch if q_td.shape[1] != k_td.shape[1]
            self.attn_qk_td = TensorData(
                name=f"blk.{block_idx}.attn_qk.weight",
                type=q_td.type,
                shape=(q_td.shape[0]+k_td.shape[0], q_td.shape[1]),  # double d_out
                n_elements=q_td.n_elements+k_td.n_elements,
                n_bytes=q_td.n_bytes+k_td.n_bytes,
                data=qk_data,
                parent=self,
            ).register()
        else:
            self.fused_qk = False
            self.q_td = q_td.register(parent=self)
            self.k_td = k_td.register(parent=self)

        self.v_td = reader.get_tensor_req(f"blk.{block_idx}.attn_v.weight").register(parent=self)
        self.attn_output = reader.get_tensor_req(f"blk.{block_idx}.attn_output.weight").register(parent=self)
        
        # moe (EP and DP)
        self.is_moe = True # figure out how to deal with dense models later
        self.ffn_norm = reader.get_tensor_req(f"blk.{block_idx}.ffn_norm.weight").register(parent=self)
        self.ffn_gate_inp = reader.get_tensor_req(f"blk.{block_idx}.ffn_gate_inp.weight").register(parent=self)

        ep_rank = params.rank % params.ep_size
        if params.exp_idxing == "contiguous":
            exps_per_rank = params.n_exps // params.ep_size
            start = ep_rank * exps_per_rank
            self.exp_ids = list(range(start, start+exps_per_rank))
        elif params.exp_idxing == "interleaved":
            self.exp_ids = list(range(ep_rank, params.n_exps, params.ep_size))

        ffn_gate_exps = reader.get_tensor_req(f"blk.{block_idx}.ffn_gate_exps.weight")
        self.ffn_gate = TensorData(
            name=f"blk.{block_idx}.ffn_gate.weight",
            type=ffn_gate_exps.type,
            shape=(len(self.exp_ids),)+ffn_gate_exps.shape[1:],
            n_elements=ffn_gate_exps.n_elements//len(self.exp_ids),
            n_bytes=ffn_gate_exps.n_bytes//len(self.exp_ids),
            data=ffn_gate_exps.data[self.exp_ids].clone(),
            parent=self,
        ).register()
        del ffn_gate_exps

        ffn_up_exps = reader.get_tensor_req(f"blk.{block_idx}.ffn_up_exps.weight")
        self.ffn_up = TensorData(
            name=f"blk.{block_idx}.ffn_up.weight",
            type=ffn_up_exps.type,
            shape=(len(self.exp_ids),)+ffn_up_exps.shape[1:],
            n_elements=ffn_up_exps.n_elements//len(self.exp_ids),
            n_bytes=ffn_up_exps.n_bytes//len(self.exp_ids),
            data=ffn_up_exps.data[self.exp_ids].clone(),
            parent=self,
        ).register()
        del ffn_up_exps

        ffn_down_exps = reader.get_tensor_req(f"blk.{block_idx}.ffn_down_exps.weight")
        self.ffn_down = TensorData(
            name=f"blk.{block_idx}.ffn_down.weight",
            type=ffn_down_exps.type,
            shape=(len(self.exp_ids),)+ffn_down_exps.shape[1:],
            n_elements=ffn_down_exps.n_elements//len(self.exp_ids),
            n_bytes=ffn_down_exps.n_bytes//len(self.exp_ids),
            data=ffn_down_exps.data[self.exp_ids].clone(),
            parent=self,
        ).register()
        del ffn_down_exps

        # TODO: non-moe case, tensor names are below
        # f"blk.{block_idx}.ffn_gate.weight"
        # f"blk.{block_idx}.ffn_up.weight"
        # f"blk.{block_idx}.ffn_down.weight"
    

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
        # attn norm
        # qkv (norm may be applied to q and k)
        # flash attn

        # ffn norm
        # gating
        # routing: all-to-all (if moe model)
        # ffn
        # gather (if moe model)

        pass