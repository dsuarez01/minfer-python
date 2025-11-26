from ._C import dequant_row as _dequant_row, quant_row as _quant_row # only exposed for testing
from .kernels import rmsnorm, il_rope, neox_rope, matmul, embed, qkv, flash_attn, moe_scoring, ffn

# NOTE 1: the kernels above should probably be an interface (compatible w the Triton and CUDA kernels 
# in terms of the call-site logic, so (1) they take grid arguments though they're unused, and (2) they take identical arguments to the
# CUDA and triton kernels even though some of them might (?) be unused) into either one or a composition of PyTorch functionals 
# that model the desired behavior of each kernel.
# NOTE 2: if you do define these kernels, make sure to use them in the tests. 
# (doesn't need to be right now, since you've not fully defined the Triton/CUDA kernels)
# NOTE 3: for now, i've defined placeholders. will fill in the logic later.