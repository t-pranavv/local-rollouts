# Docker

This folder contains Docker build configurations for running PhyAGI in consistent, isolated environments on both NVIDIA and AMD GPUs.

## Quick Start

1. **Login to your Azure Container Registry (ACR):**

```bash
az acr login -n <acr-name>
```

2. **Build a Docker image:**

- For **NVIDIA** GPUs:

```bash
bash nvidia/build_image.sh
```

- For **AMD** GPUs:

```bash
bash amd/build_image.sh
```

3. **Push the image:**

```bash
docker push <acr-name>.azurecr.io/<image-name>:<tag>
```

4. **Run the container:**

```bash
bash run_container.sh
```

*Remember to adjust the `-it` argument in the `run_container.sh` script to match your built image.*

- The base Docker images are tailored for PyTorch with Flash-Attention and DeepSpeed.
- Compatibility and performance may vary between NVIDIA (CUDA) and AMD (ROCm) environments.
- For advanced usage, including installation tweaks and version control, refer to the [full documentation](../docs/getting_started/build_docker.rst).