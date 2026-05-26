import os
from distutils.sysconfig import get_config_vars

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

(opt,) = get_config_vars("OPT")
os.environ["OPT"] = " ".join(
    flag for flag in opt.split() if flag != "-Wstrict-prototypes"
)

src = "src"
sources = [
    os.path.join(root, file)
    for root, dirs, files in os.walk(src)
    for file in files
    if file.endswith(".cpp") or file.endswith(".cu")
]

setup(
    name="pointops2",
    version="1.0",
    install_requires=["torch", "numpy"],
    packages=["pointops2"],
    package_dir={"pointops2": "functions"},
    ext_modules=[
        CUDAExtension(
            name="pointops2_cuda",
            sources=sources,
            extra_compile_args={
                "cxx": ["-g"], 
                "nvcc": [
                    "-O2",
                    "-gencode=arch=compute_86,code=sm_86",  # Ampere (A6000)
                    "-gencode=arch=compute_89,code=sm_90",   # Ada Lovelace (L40S)
                    "-gencode=arch=compute_80,code=sm_80"   # A100
                ]
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)