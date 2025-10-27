import re
from pathlib import Path
from dataclasses import dataclass

from gguf import ReaderField, ReaderTensor, GGUFReader, LlamaFileType, GGMLQuantizationType

import torch
import torch.nn as nn

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

# since ReaderTensor is immutable
@dataclass
class TensorData:
    name: str
    tensor_type: GGMLQuantizationType
    shape: tuple
    n_elements: int
    n_bytes: int
    data: torch.Tensor

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
        
        for reader in readers:
            for tensor in reader.tensors:
                self._tensor_dict[tensor.name] = TensorData(
                    name=tensor.name,
                    tensor_type=tensor.tensor_type,
                    shape=tuple(tensor.shape),
                    n_elements=tensor.n_elements,
                    n_bytes=tensor.n_bytes,
                    data=torch.from_numpy(tensor.data)
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
        return field

    def get_field_noreq(self, name: str) -> ReaderField | None:
        return self.reader.get_field(name)
    
    def get_tensor_req(self, name: str) -> TensorData:
        tensor = self._tensor_dict.get(name, None)
        if tensor is None:
            raise ValueError(f"Req. tensor '{name}' not found")
        return tensor

    def get_tensor_noreq(self, name: str) -> TensorData | None:
        return self._tensor_dict.get(name, None)

class Params:

    arch: str
    dtype: LlamaFileType
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

    def __init__(self, reader: GGUFReaderWrapper, params: dict):
        
        self.arch = reader.get_field_req("general.architecture").contents()
        self.dtype = LlamaFileType(reader.get_field_req("general.file_type").contents())
        self.block_cnt = reader.get_field_req(f"{self.arch}.block_count").contents()
        
        vocab_field = reader.get_field_noreq(f"{self.arch}.vocab_size")
        tokens_field = reader.get_field_req("tokenizer.ggml.tokens")
        
        self.vocab_size = vocab_field.contents() if vocab_field else len(tokens_field.data)
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
        self.backend = params["backend"]
        self.batch_size = params["batch_size"]
        self.max_seq_len = params["max_seq_len"]

        if self.max_seq_len > self.ctx_len:
            print(
                f"Max seq len provided is greater than len of ctx model was trained on."
                f"Can result in degraded model performance, take note."
                f"max_seq_len: {self.max_seq_len}"
                f"context_len: {self.ctx_len}"
            )

class BufPool(nn.Module):
    """ All act bufs per device """
    def __init__(self, params: Params):
        super().__init__()
        
        # general act bufs
        self.register_buffer("x", torch.zeros(params.batch_size, params.max_seq_len, params.hidden_dim))
        self.register_buffer("xb", torch.zeros(params.batch_size, params.max_seq_len, params.hidden_dim))
        self.register_buffer("xb2", torch.zeros(params.batch_size, params.max_seq_len, params.hidden_dim))
        
        # attn
        self.register_buffer("q", torch.zeros(params.batch_size, params.n_heads, params.max_seq_len, params.head_dim))
        self.register_buffer("k", torch.zeros(params.batch_size, params.n_kv_heads, params.max_seq_len, params.head_dim))
        self.register_buffer("v", torch.zeros(params.batch_size, params.n_kv_heads, params.max_seq_len, params.head_dim))
        
        self.register_buffer("att_scores", torch.zeros(params.batch_size, params.n_heads, params.max_seq_len, params.max_seq_len))

        if params.n_heads * params.head_dim != params.hidden_dim: # applies to some models like qwen
            self.register_buffer("att_out", torch.zeros(params.batch_size, params.max_seq_len, params.n_heads, params.head_dim))

        # moe routing
        self.register_buffer("moe_scores", torch.zeros(params.batch_size, params.max_seq_len, params.n_exps))
        self.register_buffer("act_exps", torch.zeros(params.batch_size, params.max_seq_len, params.n_act_exps))
        self.register_buffer("act_exps_ws", torch.zeros(params.batch_size, params.max_seq_len, params.n_act_exps))

        # ffn
        self.register_buffer("hb", torch.zeros(params.batch_size, params.max_seq_len, params.mlp_dim))
        self.register_buffer("hb2", torch.zeros(params.batch_size, params.max_seq_len, params.mlp_dim))

        # logits
        self.register_buffer("logits", torch.zeros(params.batch_size, params.max_seq_len, params.vocab_size))