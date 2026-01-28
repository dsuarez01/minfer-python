import os
import torch
import glob

from setuptools import setup
from torch.utils.cpp_extension import (
    CUDAExtension,
    BuildExtension,
    CUDA_HOME,
)

library_name = "minfer"

if torch.__version__ >= "2.6.0":
    py_limited_api = True
else:
    py_limited_api = False

def get_extensions():
    debug_mode = os.getenv("DEBUG", "0") == "1"

    if debug_mode:
        print("Compiling in debug mode")

    assert torch.cuda.is_available() and CUDA_HOME is not None, "CUDA not enabled or CUDA_HOME not set"
    
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join("src", library_name, "kernels", "torch_ext", "csrc")
    include_path = "-I"+os.path.join(this_dir, extensions_dir)

    extra_link_args = [
        "-fopenmp",
    ]
    extra_compile_args = {
        "cxx": [
            "-std=c++17",
            "-O3" if not debug_mode else "-O0",
            "-fopenmp",
            "-fdiagnostics-color=always",
            "-DPy_LIMITED_API=0x030c0000",  # min CPython v3.12,
            "-DTORCH_TARGET_VERSION=0x020a000000000000",
            include_path,
        ],
        "nvcc": [
            "-O3" if not debug_mode else "-O0",
            "-gencode=arch=compute_89,code=compute_89",
            "-gencode=arch=compute_89,code=sm_89",
            "-DTORCH_TARGET_VERSION=0x020a000000000000", # min pytorch 2.10 for stable ABI
            "-DUSE_CUDA",
            "--ptxas-options=-v",
            include_path,
        ],
    }
    
    if debug_mode:
        extra_compile_args["cxx"].append("-g")
        extra_compile_args["nvcc"].extend(["-G","-g"])
        extra_link_args.extend(["-O0", "-g"])
    else:
        extra_compile_args["nvcc"].extend(["-lineinfo", "-use_fast_math"])

    sources = glob.glob(os.path.join(extensions_dir, "**", "*.cpp"), recursive=True) + [os.path.join(extensions_dir, "cuda", "reg.cu")]
    
    return [
        CUDAExtension(
            name=f"{library_name}.kernels.torch_ext._C",
            sources=sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            py_limited_api=py_limited_api,
        )
    ]

setup(
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp312"}} if py_limited_api else {},
)