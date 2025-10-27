from __future__ import annotations

from minfer.kernels import KernelBackend
from minfer.utils import GGUFReaderWrapper, Params, BufPool, TensorData

import torch
import torch.nn as nn

class Model:
    blocks: list[Transformer]
    buf_pools: dict[torch.device, BufPool]
    compute_logits: bool
    devices: list[torch.device]
    device_map: dict[torch.device, list[BufPool | TensorData | Transformer]]
    ks: dict
    params: Params

    output_norm_td: TensorData
    output_td: TensorData
    token_embd_td: TensorData

    def __init__(self, path: str, run_params: dict) -> None:
        reader = GGUFReaderWrapper(path)
        self.params = Params(reader, run_params)
        self.compute_logits = False
        self.devices = [torch.device("cpu")] + [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
        self.device_map = {}
        self.ks = {
            device.type: KernelBackend("cpu" if device.type == "cpu" else self.params.backend)
            for device in self.devices
        }
        self.buf_pools = {
            device: BufPool(self.params) for device in self.devices
        }

        # TensorData attrs
        self.token_embd_td = reader.get_tensor_req("token_embd.weight")
        output_td = reader.get_tensor_noreq("output.weight")
        if output_td:
            self.output_td = output_td
        else:
            # tied wrd embd, just clone token_embd data
            self.output_td = TensorData(
                name="output.weight",
                tensor_type=self.token_embd_td.tensor_type,
                shape=self.token_embd_td.shape,
                n_elements=self.token_embd_td.n_elements,
                n_bytes=self.token_embd_td.n_bytes,
                data=self.token_embd_td.data.clone()
            )

        self.output_norm_td = reader.get_tensor_req("output_norm.weight")

        # init Transformer blocks on CPU
        self.blocks = []
        for i in range(self.params.block_cnt):
            self.blocks.append(Transformer(i, reader, self.params, self))

        # device map maps everything to CPU
        cpu_device = self.devices[0]
        self.device_map[cpu_device] = [p for p in self.buf_pools.values()]
        self.device_map[cpu_device].append(self.token_embd_td)
        for block in self.blocks:
            self.device_map[cpu_device].append(block)
        self.device_map[cpu_device].append(self.output_norm_td)
        self.device_map[cpu_device].append(self.output_td)

    def dispatch(self) -> None:
        """ Tensors and blocks moved to assigned devices """
        self._assign_devices()
        for device, items in self.device_map.items():
            for item in items:
                if isinstance(item, BufPool):
                    item.to(device)
                elif isinstance(item, TensorData):
                    item.data = item.data.to(device)
                elif isinstance(item, Transformer):
                    item.to(device)
                    item.device = device
                else:
                    raise TypeError(f"Expected BufPool, tuple, or Transformer: got {type(item).__name__}")
    
    def _assign_devices(self) -> None:
        """ Greedily assign items to GPUs on avail. mem """
        mem_avail = {
            d: torch.cuda.get_device_properties(d).total_memory
            for d in self.devices
            if d.type == "cuda"
        }

        def get_size(item):
            if isinstance(item, (BufPool, Transformer)):
                return sum(p.element_size() * p.nelement() for p in item.buffers())
            elif isinstance(item, TensorData):
                return item.data.element_size() * item.data.nelement()
            else:
                raise TypeError(f"Unknown item type: {type(item)}")

        device_map = {d: [] for d in self.devices}
        cpu_device = self.devices[0]

        for item in self.device_map[cpu_device]:
            size = get_size(item)
            target = next((d for d in mem_avail if mem_avail[d] >= size), None)
            if target is None:
                target = cpu_device
            else:
                mem_avail[target] -= size
            device_map[target].append(item)
        
        self.device_map = device_map

    @torch.compile(mode="reduce-overhead")
    def forward(self, tokens: torch.Tensor, pos: int) -> None:
        pass

class Transformer(nn.Module):
    block_idx: int
    device: torch.device
    is_moe: bool
    model: Model
    
    def __init__(self, block_idx: int, reader: GGUFReaderWrapper, params: Params, model: Model) -> None:
        super().__init__()
        self.block_idx = block_idx
        self.model = model
        self.device = torch.device("cpu")
        self.is_moe = params.n_exps > 0
        
        # k,v caches
        self.register_buffer(f"k_cache", torch.zeros(params.batch_size, params.n_kv_heads, params.max_seq_len, params.head_dim))
        self.register_buffer(f"v_cache", torch.zeros(params.batch_size, params.n_kv_heads, params.max_seq_len, params.head_dim))

        def reg(attr_name: str, tensor_name: str):
            td = reader.get_tensor_req(tensor_name)
            setattr(self, f"{attr_name}_td", td)
            self.register_buffer(f"{attr_name}_weight", td.data)
        
        # attn
        reg("attn_q", f"blk.{block_idx}.attn_q.weight")
        reg("attn_k", f"blk.{block_idx}.attn_k.weight")
        reg("attn_v", f"blk.{block_idx}.attn_v.weight")
        reg("attn_output", f"blk.{block_idx}.attn_output.weight")
        reg("attn_norm", f"blk.{block_idx}.attn_norm.weight")
        
        # ffn
        reg("ffn_norm", f"blk.{block_idx}.ffn_norm.weight")
        
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
    def bufs(self) -> BufPool:
        return self.model.buf_pools[self.device]
    
    @property
    def ks(self) -> KernelBackend:
        return self.model.ks[self.device.type]

    def forward(self, x: torch.Tensor, pos: int) -> None:
        pass