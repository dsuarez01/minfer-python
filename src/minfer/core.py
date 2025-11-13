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

    woutnorm: TensorData
    wout: TensorData
    wtoken_embd: TensorData

    def __init__(self, path: str, run_params: dict) -> None:
        super().__init__()
        reader = GGUFReaderWrapper(path)
        self.params = Params(reader, run_params)
        self.kerns = KernelBackend(self.params.backend)
        self.bufs = BufPool(self.params)
        self.luts = LUT(self.params.act_dtype)
        self.compute_logits = False

        # TensorData attrs
        self.wtoken_embd = reader.get_tensor_req("token_embd.weight").register(parent=self)

        wout = reader.get_tensor_noreq("output.weight")
        if wout:
            self.wout = wout.register(parent=self)
        else:
            # tied wrd embd, just clone token_embd data
            self.wout = TensorData(
                name="output.weight",
                type=self.wtoken_embd.type,
                shape=self.wtoken_embd.shape,
                n_elements=self.wtoken_embd.n_elements,
                n_bytes=self.wtoken_embd.n_bytes,
                data=self.wtoken_embd.data.clone(),
                parent=self,
            ).register()

        self.woutnorm = reader.get_tensor_req("output_norm.weight").register(parent=self)
        if self.woutnorm.data.dtype != self.params.act_dtype:
            self.woutnorm.data = self.woutnorm.data.to(self.params.act_dtype)

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
        self.kerns.embed()
        # blocks
        for i in range(self.params.block_cnt):
            self.blocks.forward()
        # rmsnorm
        self.kerns.rmsnorm()
        # lm head
        self.kerns.matmul()
        return


class Transformer(nn.Module):
    block_idx: int
    is_moe: bool
    model: Model
    exp_ids: list[int]

    k_cache: TensorData
    v_cache: TensorData
    
    wattn_norm: TensorData
    wattn_out: TensorData
    wq: TensorData
    wq_norm: TensorData
    wk: TensorData
    wk_norm: TensorData
    wv: TensorData
    wffn_norm: TensorData
    wffn_gate: TensorData
    wffn_up: TensorData
    wffn_down: TensorData
    
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
        self.wattn_norm = reader.get_tensor_req(f"blk.{block_idx}.attn_norm.weight").register(parent=self)
        if self.wattn_norm.data.dtype != params.act_dtype:
            self.wattn_norm.data = self.wattn_norm.data.to(params.act_dtype)
        wq_norm = reader.get_tensor_noreq(f"blk.{block_idx}.attn_q_norm.weight")
        wk_norm = reader.get_tensor_noreq(f"blk.{block_idx}.attn_k_norm.weight")
        assert (wq_norm is None) == (wk_norm is None) # both either present or absent
        if wq_norm and wk_norm:
            self.wq_norm = wq_norm.register(parent=self)
            if self.wq_norm.data.dtype != params.act_dtype:
                self.wq_norm.data = self.wq_norm.data.to(params.act_dtype)
            self.wk_norm = wk_norm.register(parent=self)
            if self.wk_norm.data.dtype != params.act_dtype:
                self.wk_norm.data = self.wk_norm.data.to(params.act_dtype)

        ## projs
        self.wq = reader.get_tensor_req(f"blk.{block_idx}.attn_q.weight").register(parent=self)
        self.wk = reader.get_tensor_req(f"blk.{block_idx}.attn_k.weight").register(parent=self)
        self.wv = reader.get_tensor_req(f"blk.{block_idx}.attn_v.weight").register(parent=self)
        self.wattn_out = reader.get_tensor_req(f"blk.{block_idx}.attn_output.weight").register(parent=self)
        
        # moe (EP and DP)
        self.is_moe = True # figure out how to deal with dense models later
        self.wffn_norm = reader.get_tensor_req(f"blk.{block_idx}.ffn_norm.weight").register(parent=self)
        if self.wffn_norm.data.dtype != params.act_dtype:
            self.wffn_norm.data = self.wffn_norm.data.to(params.act_dtype)
        self.wffn_gate_inp = reader.get_tensor_req(f"blk.{block_idx}.ffn_gate_inp.weight").register(parent=self)

        if params.exp_idxing == "contiguous":
            exps_per_rank = params.n_exps // params.ep_size
            start = params.ep_rank*exps_per_rank
            self.exp_ids = list(range(start, start+exps_per_rank))
        elif params.exp_idxing == "interleaved":
            self.exp_ids = list(range(params.ep_rank, params.n_exps, params.ep_size))

        wffn_gate_exps = reader.get_tensor_req(f"blk.{block_idx}.ffn_gate_exps.weight")
        self.wffn_gate = TensorData(
            name=f"blk.{block_idx}.ffn_gate.weight",
            type=wffn_gate_exps.type,
            shape=(len(self.exp_ids),)+wffn_gate_exps.shape[1:],
            n_elements=wffn_gate_exps.n_elements//len(self.exp_ids),
            n_bytes=wffn_gate_exps.n_bytes//len(self.exp_ids),
            data=wffn_gate_exps.data[self.exp_ids].clone(),
            parent=self,
        ).register()
        del wffn_gate_exps

        wffn_up_exps = reader.get_tensor_req(f"blk.{block_idx}.ffn_up_exps.weight")
        self.wffn_up = TensorData(
            name=f"blk.{block_idx}.ffn_up.weight",
            type=wffn_up_exps.type,
            shape=(len(self.exp_ids),)+wffn_up_exps.shape[1:],
            n_elements=wffn_up_exps.n_elements//len(self.exp_ids),
            n_bytes=wffn_up_exps.n_bytes//len(self.exp_ids),
            data=wffn_up_exps.data[self.exp_ids].clone(),
            parent=self,
        ).register()
        del wffn_up_exps

        wffn_down_exps = reader.get_tensor_req(f"blk.{block_idx}.ffn_down_exps.weight")
        self.wffn_down = TensorData(
            name=f"blk.{block_idx}.ffn_down.weight",
            type=wffn_down_exps.type,
            shape=(len(self.exp_ids),)+wffn_down_exps.shape[1:],
            n_elements=wffn_down_exps.n_elements//len(self.exp_ids),
            n_bytes=wffn_down_exps.n_bytes//len(self.exp_ids),
            data=wffn_down_exps.data[self.exp_ids].clone(),
            parent=self,
        ).register()
        del wffn_down_exps

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

    def forward(self) -> None:
        # attn norm
        self.kerns.rmsnorm()
        # fused qkv
        self.kerns.qkv()
        # (optional) rmsnorm per-head over q and k
        if self.wq_norm and self.wk_norm:
            self.kerns.rmsnorm()
            self.kerns.rmsnorm()
        # rope over q and k
        self.kerns.rope()
        self.kerns.rope()
        # flash attn
        self.kerns.flash_attn()
        # ffn norm
        self.kerns.rmsnorm()
        # moe scoring
        self.kerns.moe_scoring()
        # routing: all-to-all (if moe model)
        # unsure what is usually done here
        # ffn
        self.kerns.ffn()
        # gather (if moe model)
        return