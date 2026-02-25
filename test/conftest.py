import torch
import minfer.kernels.torch_ext # registers ops in cases where we don't use KernelBackend("torch_ext")