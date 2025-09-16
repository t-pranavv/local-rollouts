# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from setuptools import find_packages, setup

dependencies = [
    "accelerate>=1.6.0",
    "azure-identity",
    "azure-keyvault-secrets",
    "azure-storage-blob",
    "datasets>=3.5.0",
    "deepspeed>=0.16.4",
    "dion @ git+https://github.com/microsoft/dion",
    "einops",
    "evaluate>=0.4.3",
    "flash-attn>=2.7.0",
    "huggingface-hub",
    "ipython",
    "lightning>=2.5.0",
    "lm-eval>=0.4.8",
    "math-verify[antlr4_9_3]",
    "mlflow",
    "natsort",
    "nbsphinx",
    "omegaconf",
    "pandas",
    "peft>=0.15.1",
    "pre-commit",
    "pydata-sphinx-theme",
    "pylatexenc",
    "pytest",
    "ray",
    "redis",
    "safetensors",
    "sphinx",
    "sphinx-book-theme",
    "tensorboard",
    "tokenizers>=0.21.1",
    "torchao",
    "transformers>=4.51.3",
    "trl>=0.17.0",
    "typing-extensions",
    "vllm>=0.7.3",
    "wandb",
]


def filter_dependencies(*pkgs):
    return [x for x in dependencies if any(y in x for y in pkgs)]


install_requires = filter_dependencies(
    "accelerate",
    "azure-identity",
    "azure-keyvault-secrets",
    "azure-storage-blob",
    "datasets",
    "deepspeed",
    "dion",
    "einops",
    "evaluate",
    "huggingface-hub",
    "lightning",
    "math-verify[antlr4_9_3]",
    "mlflow",
    "natsort",
    "omegaconf",
    "pandas",
    "peft",
    "pre-commit",
    "pylatexenc",
    "pytest",
    "ray",
    "redis",
    "safetensors",
    "tensorboard",
    "tokenizers",
    "transformers",
    "trl",
    "typing-extensions",
    "wandb",
)

extras_require = {}

extras_require["docs"] = filter_dependencies(
    "ipython", "nbsphinx", "pydata-sphinx-theme", "sphinx", "sphinx-book-theme"
)
extras_require["eval"] = filter_dependencies("lm-eval")
extras_require["flash-attn"] = filter_dependencies("flash-attn")
extras_require["rl"] = filter_dependencies("torchao", "vllm")

extras_require["all"] = (
    extras_require["docs"] + extras_require["eval"] + extras_require["flash-attn"] + extras_require["rl"]
)

with open("README.md", "r", encoding="utf_8") as f:
    long_description = f.read()

setup(
    name="phyagi",
    version="3.2.2.dev",  # Expected format is x.y.z.dev or x.y.z (when adding a release tag)
    description="Physics of AGI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Microsoft",
    url="https://github.com/microsoft/phyagi-sdk",
    license="MIT",
    python_requires=">=3.10.0",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": ["phyagi-cli=phyagi.cli.interface:main"],
    },
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Development Status :: 5 - Production/Stable",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
