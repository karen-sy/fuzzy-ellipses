#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
from setuptools.command.develop import develop
from subprocess import check_call
from os import path

os.path.dirname(os.path.abspath(__file__))


class update_submodules(develop):
    def run(self):
        if path.exists(".git"):
            check_call(["git", "submodule", "update", "--init", "--recursive"])
        develop.run(self)


setup(
    name="fuzzy_ellipses",
    packages=["fuzzy_ellipses"],
    ext_modules=[
        CUDAExtension(
            name="fuzzy_ellipses._C",
            sources=[
                "cuda_rasterizer/rasterizer_impl.cu",
                "cuda_rasterizer/forward.cu",
                "rasterize_main.cu",
                "ext.cpp",
            ],
            extra_compile_args={
                "nvcc": [
                    "-gencode arch=compute_89,code=sm_89",
                    "--extended-lambda",
                    "-lcublas",
                    "-I"
                    + os.path.join(
                        os.path.dirname(os.path.abspath(__file__)), "third_party/glm/"
                    ),
                ]
            },
        )
    ],
    cmdclass={
        "build_ext": BuildExtension,
        "develop": update_submodules,
    },
)
