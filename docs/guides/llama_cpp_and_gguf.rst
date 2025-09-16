Llama.cpp and GGUF
==================

`llama.cpp <https://github.com/ggerganov/llama.cpp>`_ is a lightweight, high-performance C++ inference engine originally designed for running LLaMA models efficiently on CPUs. Over time, it has expanded to support a wider variety of transformer architectures and GPU acceleration, making it a versatile tool for local inference workloads.

Despite its broad support, Llama.cpp does not run models in their native Hugging Face format. Instead, it relies on a specialized binary format called **GGUF**.

What is GGUF?
-------------

**GGUF (GGML Universal Format)** is a compact, self-contained format that packages all the components required for inference—model weights, architecture metadata, tokenizer configuration, and more—into a single binary file. This design simplifies deployment and distribution.

A key benefit of GGUF is its support for **memory-mapped inference**, which enables efficient streaming of model layers during execution. As a result, even very large models (e.g., LLaMA-70B) can be run on systems with limited RAM or VRAM, as only the required parts of the model are loaded on demand.

Installation
------------

We provide a convenient script that installs all dependencies and builds ``llama.cpp`` from source. This includes compiling binaries, installing Python requirements, and placing executables in a user-local path:

.. code-block:: bash

    cd phyagi-sdk
    bash scripts/tools/install/llama-cpp.sh

This script ensures ``llama.cpp`` and its tools (like ``llama-quantize``, ``llama-cli``, and ``llama-bench``) are accessible system-wide for the current user.

Model conversion and quantization
---------------------------------

Once ``llama.cpp`` is installed, models must be converted from their Hugging Face format into GGUF format before use. To streamline this process, we provide a helper script that performs the following steps:

1. Converts a Hugging Face checkpoint to GGUF format.
2. Applies quantization to reduce model size and improve inference speed.
3. Optionally benchmarks the quantized model.

To run the full conversion pipeline:

.. code-block:: bash

    cd phyagi-sdk
    bash scripts/tools/convert_and_quantize_to_gguf.sh \
      --checkpoint=<path/to/checkpoint> \
      --outdir=<path/to/output-dir> \
      --quantization-type=Q2_K \
      --run-llama-bench

Supported quantization types include: ``Q2_K``, ``Q3_K_S``, ``Q4_0``, ``Q4_K``, ``Q5_K``, ``Q6_K``, ``IQ4_XS``, and others.

This script also validates the resulting GGUF files using ``llama-bench`` benchmark tool.

Running inference
-----------------

With a quantized GGUF model in hand, you can run inference directly using the ``llama-cli`` interface:

.. code-block:: bash

    llama-cli -m <path/to/model.gguf> -cnv -p "What is the derivative of x^2?"

The ``-cnv`` flag enables conversational mode, allowing you to interact with the model in a chat-like manner.
