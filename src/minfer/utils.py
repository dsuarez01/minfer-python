from __future__ import annotations
import re
from pathlib import Path
from dataclasses import dataclass

from gguf import ReaderField, ReaderTensor, GGUFReader

import torch
import torch.nn as nn

from .const import GGMLQuantizationType

# https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
PATTERN = re.compile(
    r"^(?P<BaseName>[A-Za-z0-9\s]*(?:(?:-(?:(?:[A-Za-z\s][A-Za-z0-9\s]*)|(?:[0-9\s]*)))*))"
    r"-(?:(?P<SizeLabel>(?:\d+x)?(?:\d+\.)?\d+[A-Za-z](?:-[A-Za-z]+(\d+\.)?\d+[A-Za-z]+)?)"
    r"(?:-(?P<FineTune>[A-Za-z0-9\s-]+))?)?"
    r"-(?:(?P<Version>v\d+(?:\.\d+)*))"
    r"(?:-(?P<Encoding>(?!LoRA|vocab)[\w_]+))?"
    r"(?:-(?P<Type>LoRA|vocab))?"
    r"(?:-(?P<Shard>\d{5}-of-\d{5}))?"
    r"\.gguf$"
)

# use this dataclass since ReaderTensor is immutable IntEnum (see gguf-py)
@dataclass
class TensorData:
    name: str
    type: GGMLQuantizationType
    shape: tuple
    n_elements: int
    n_bytes: int
    data: torch.Tensor
    parent: nn.Module | None

    def register(self, parent: nn.Module | None = None) -> TensorData:
        if parent is None:
            if self.parent is None:
                raise ValueError("No parent module provided")
            parent = self.parent
        self.parent = parent
        parent.register_buffer(self.name, self.data)
        return self

class GGUFReaderWrapper:
    """ Wrapper handles model sharding, and maps tensors by name instead of storing in list """
    
    reader: GGUFReader
    _tensor_dict: dict[str, TensorData]

    def __init__(self, path_str: str):
        
        path = Path(path_str)
        match = PATTERN.match(path.name)
        
        if not match:
            raise ValueError(
                f"Filename: '{path.name}' "
                f"Expected: <BaseName>-<SizeLabel>-<Version>[-<FineTune>][-<Encoding>][-<Type>][-<Shard>].gguf"
            )
        
        if not all([match.group("BaseName"), match.group("SizeLabel"), match.group("Version")]):
            raise ValueError(
                f"Filename '{path.name}' missing required fields (BaseName, SizeLabel, or Version)"
            )
        
        shard_info = match.group("Shard")
        shard_total = 1
        if shard_info:
            shard_match = re.match(r"(?P<shard_num>\d{5})-of-(?P<shard_total>\d{5})", shard_info)
            if not shard_match:
                raise ValueError(f"Invalid shard format: {shard_info}")
            shard_total = int(shard_match.group("shard_total"))
        
        base_name = "-".join(filter(None, [
            match.group("BaseName"),
            match.group("SizeLabel"),
            match.group("FineTune"),
            match.group("Version"),
            match.group("Encoding"),
            match.group("Type"),
        ]))
        
        # auto-loads all shards (single file is just one "shard")
        readers: list[GGUFReader] = []
        for i in range(1, shard_total+1):
            load_path = path_str if not shard_info else path.parent / f"{base_name}-{i:05d}-of-{shard_total:05d}.gguf"
            readers.append(GGUFReader(load_path))
        
        # first shard contains all model param metadata, rest contain tensors
        self.reader = readers[0]
        self._tensor_dict = {}
        
        for r in readers:
            for t in r.tensors:
                self._tensor_dict[t.name] = TensorData(
                    name=t.name,
                    type=t.tensor_type,
                    shape=tuple(t.shape),
                    n_elements=t.n_elements,
                    n_bytes=t.n_bytes,
                    data=torch.from_numpy(t.data),
                    parent=None,
                )

        # quick check: tensor count matches expected
        field = self.get_field_noreq("split.tensors.count")
        expected = field.contents() if field is not None else self.get_field_req("tensor_count").contents()
        actual = len(self._tensor_dict)

        if expected != actual:
            raise ValueError(f"Tensor count mismatch: expected {expected} tensors, got {actual}")

    def get_field_req(self, name: str) -> ReaderField:
        field = self.reader.get_field(name)
        if field is None:
            raise ValueError(f"Req. field '{name}' not found")
        return field.contents()

    def get_field_noreq(self, name: str) -> ReaderField | None:
        return self.reader.get_field(name)
    
    def get_tensor_req(self, name: str) -> TensorData:
        tensor = self._tensor_dict.pop(name, None)
        if tensor is None:
            raise ValueError(f"Req. tensor '{name}' not found")
        return tensor

    def get_tensor_noreq(self, name: str) -> TensorData | None:
        return self._tensor_dict.pop(name, None)

class Params:

    arch: str
    block_cnt: int
    vocab_size: int
    ctx_len: int
    hidden_dim: int
    mlp_dim: int
    norm_eps: float
    n_heads: int
    n_kv_heads: int
    head_dim: int
    rotary_dim: int
    freq_base: float
    n_exps: int
    n_act_exps: int
    backend: str
    batch_size: int
    max_seq_len: int
    act_dtype: torch.dtype
    world_size: int
    rank: int
    local_rank: int
    dp_size: int
    ep_size: int
    exp_idxing: str
    ep_rank: int
    dp_rank: int

    def __init__(self, reader: GGUFReaderWrapper, run_params: dict):
        
        self.arch = reader.get_field_req("general.architecture").contents()
        self.block_cnt = reader.get_field_req(f"{self.arch}.block_count").contents()
        
        vocab_field = reader.get_field_noreq(f"{self.arch}.vocab_size")
        tokens = reader.get_field_req("tokenizer.ggml.tokens")
        
        self.vocab_size = vocab_field.contents() if vocab_field else len(tokens.data)
        self.ctx_len = reader.get_field_req(f"{self.arch}.context_length").contents()

        # all the relevant parameters from the config are set here
        # hidden_dim, vocab_size, what else?
        self.hidden_dim = reader.get_field_req(f"{self.arch}.embedding_length").contents()
        self.mlp_dim = reader.get_field_req(f"{self.arch}.feed_forward_length").contents()
        self.norm_eps = reader.get_field_req(f"{self.arch}.attention.layer_norm_rms_epsilon").contents()

        self.n_heads = reader.get_field_req(f"{self.arch}.attention.head_count").contents()
        self.n_kv_heads = reader.get_field_req(f"{self.arch}.attention.head_count_kv").contents()
        self.head_dim = self.hidden_dim // self.n_heads
        
        self.rotary_dim = reader.get_field_req(f"{self.arch}.rope.dimension_count").contents()
        self.freq_base = reader.get_field_req(f"{self.arch}.rope.freq_base").contents()
        
        n_exps_field = reader.get_field_noreq(f"{self.arch}.expert_count")
        n_act_exps_field = reader.get_field_noreq(f"{self.arch}.expert_used_count")

        self.n_exps = n_exps_field.contents() if n_exps_field else 0
        self.n_act_exps = n_act_exps_field.contents() if n_act_exps_field else 0

        # runtime params: will eventually include temp, seed, etc.
        self.backend = run_params["backend"]
        self.batch_size = run_params["batch_size"]
        self.max_seq_len = run_params["max_seq_len"]
        assert run_params["act_dtype"] in ["float32", "float16"], "Only supported activation dtypes are float32 and float16"
        self.act_dtype = getattr(torch, run_params["act_dtype"])
        
        # data + expert parallelism
        self.world_size = run_params["world_size"]
        self.rank = run_params["rank"]
        self.local_rank = run_params["local_rank"]
        self.dp_size = run_params["dp_size"]
        self.ep_size = run_params["ep_size"]
        self.exp_idxing = run_params["exp_idxing"]
        self.dp_rank = self.rank // self.ep_size
        self.ep_rank = self.rank % self.ep_size
        
        assert self.batch_size % self.dp_size == 0, "dp size must divide batch sz"
        assert self.dp_size <= self.batch_size, "dp size exceeds batch sz"

        assert self.n_exps % self.ep_size == 0, "ep size must divide # experts"
        assert self.ep_size <= self.n_exps, "ep size exceeds # experts"
        
        assert self.dp_size * self.ep_size == self.world_size, "dp and ep sizes must split world size exactly"

        if self.max_seq_len > self.ctx_len:
            print(
                f"Max seq len provided is greater than len of ctx model was trained on."
                f"Can result in degraded model performance, take note."
                f"max_seq_len: {self.max_seq_len}"
                f"context_len: {self.ctx_len}"
            )

class BufPool(nn.Module):
    """ All act bufs per device """
    
    x: torch.Tensor
    xb: torch.Tensor
    xb2: torch.Tensor
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    att_scores: torch.Tensor
    moe_scores: torch.Tensor
    act_exps: torch.Tensor
    act_exps_ws: torch.Tensor
    hb: torch.Tensor
    hb2: torch.Tensor
    logits: torch.Tensor
    
    def __init__(self, params: Params):
        super().__init__()
        
        # general act bufs
        self.register_buffer("x", torch.zeros((params.batch_size // params.dp_size, params.max_seq_len, params.hidden_dim), dtype=params.act_dtype))
        self.register_buffer("xb", torch.zeros((params.batch_size // params.dp_size, params.max_seq_len, params.hidden_dim), dtype=params.act_dtype))
        self.register_buffer("xb2", torch.zeros((params.batch_size // params.dp_size, params.max_seq_len, params.hidden_dim), dtype=params.act_dtype))
        
        # attn
        self.register_buffer("q", torch.zeros((params.batch_size // params.dp_size, params.max_seq_len, params.n_heads, params.head_dim), dtype=params.act_dtype))

        # moe routing
        self.register_buffer("moe_scores", torch.zeros((params.batch_size // params.dp_size, params.max_seq_len, params.n_exps), dtype=params.act_dtype))
        self.register_buffer("act_exps", torch.zeros((params.batch_size // params.dp_size, params.max_seq_len, params.n_act_exps), dtype=torch.uint8))
        self.register_buffer("act_exps_ws", torch.zeros((params.batch_size // params.dp_size, params.max_seq_len, params.n_act_exps), dtype=params.act_dtype))

        # experts
        self.register_buffer("hb", torch.zeros((params.batch_size // params.dp_size, params.max_seq_len, params.n_exps // params.ep_size, params.mlp_dim), dtype=params.act_dtype))
        self.register_buffer("hb2", torch.zeros((params.batch_size // params.dp_size, params.max_seq_len, params.n_exps // params.ep_size, params.mlp_dim), dtype=params.act_dtype))

        # logits
        self.register_buffer("logits", torch.zeros((params.batch_size // params.dp_size, params.max_seq_len, params.vocab_size), dtype=params.act_dtype))