# Copyright (c) Facebook, Inc. and its affiliates.

from os import path, listdir

import setuptools
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

here = path.abspath(path.dirname(__file__))


def find_sources(root_dir):
    sources = []
    for file in listdir(root_dir):
        _, ext = path.splitext(file)
        if ext in [".cpp", ".cu"]:
            sources.append(path.join(root_dir, file))

    return sources


def make_extension(name, package):
    return CUDAExtension(
        name="{}.{}._backend".format(package, name),
        sources=find_sources(path.join("src", name)),
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": ["--expt-extended-lambda"],
        },
        include_dirs=[path.join(here, "include")],
    )


with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    # Meta-data
    name="seamseg",
    author="Lorenzo Porzi",
    author_email="lorenzo@mapillary.com",
    description="Seamless Scene Segmentation for Pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mapillary/seamseg",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],

    # Versioning
    use_scm_version={"root": ".", "relative_to": __file__, "write_to": "seamseg/_version.py"},

    # Requirements
    setup_requires=["setuptools_scm"],
    python_requires=">=3, <4",

    # Package description
    packages=[
        "seamseg",
        "seamseg.algos",
        "seamseg.config",
        "seamseg.data",
        "seamseg.models",
        "seamseg.modules",
        "seamseg.modules.heads",
        "seamseg.utils",
        "seamseg.utils.bbx",
        "seamseg.utils.nms",
        "seamseg.utils.parallel",
        "seamseg.utils.roi_sampling",
    ],
    ext_modules=[
        make_extension("nms", "seamseg.utils"),
        make_extension("bbx", "seamseg.utils"),
        make_extension("roi_sampling", "seamseg.utils")
    ],
    cmdclass={"build_ext": BuildExtension},
    include_package_data=True,
)
