from setuptools import setup
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension

setup(
    ext_modules=[
        CppExtension( # reference kernels
            name="minfer.kernels._ref._C",
            sources=[
                "src/minfer/kernels/_ref/quants.cpp",
                "src/minfer/kernels/_ref/quants_impl.cpp",
            ],
            extra_compile_args=["-std=c++17", "-O3"],
        ),
        CUDAExtension( # CUDA kernels
            name="minfer.kernels.cuda._kernels_C",
            sources=["src/minfer/kernels/cuda/kernels.cu"],
            extra_compile_args={
                "cxx": ["-std=c++17", "-O3"],
                "nvcc": ["-O3", "--use_fast_math"]
            }
        ),
        CUDAExtension( # CUDA quant-specific kernels
            name="minfer.kernels.cuda._quants_C",
            sources = [
                "src/minfer/kernels/cuda/quants.cu", 
                "src/minfer/kernels/cuda/quants_impl.cu"
            ],
            extra_compile_args={
                "cxx": ["-std=c++17", "-O3"],
                "nvcc": ["-O3", "--use_fast_math"]
            }
        ), 
    ],
    cmdclass={"build_ext": BuildExtension}
)