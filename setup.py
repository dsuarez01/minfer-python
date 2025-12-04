import os
import torch
import glob

from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    CppExtension,
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

    extra_link_args = []
    extra_compile_args = {
        "cxx": [
            "-std=c++17",
            "-O3" if not debug_mode else "-O0",
            "-fdiagnostics-color=always",
            "-DPy_LIMITED_API=0x030d0000",  # min CPython v3.13,
            include_path,
        ],
        "nvcc": [
            "-O3" if not debug_mode else "-O0",
            "--use_fast_math",
            "-rdc=true",
            "-gencode=arch=compute_70,code=lto_70",
            include_path,
        ],
    }
    
    if debug_mode:
        extra_compile_args["cxx"].append("-g")
        extra_compile_args["nvcc"].append("-g")
        extra_link_args.extend(["-O0", "-g"])

    sources = list(glob.glob(os.path.join(extensions_dir, "*.cpp")))
    
    extensions_cuda_dir = os.path.join(extensions_dir, "cuda")
    cuda_sources = list(glob.glob(os.path.join(extensions_cuda_dir, "*.cu")))
    sources += cuda_sources
    
    return [
        CUDAExtension(
            name=f"{library_name}.kernels.torch_ext._C",
            sources=sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            py_limited_api=py_limited_api,
            dlink=True,
        )
    ]

setup(
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp313"}} if py_limited_api else {},
)