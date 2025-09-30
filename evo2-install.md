# Install Evo2 on EIDF

We specify a base docker image from the docker hub (or similar). The dependencies include Python 3.11 and Nvidia's Transformer Engine (for FP8 in some of the Evo2 layers).

## A docker image including Transformer Engine

From: https://github.com/NVIDIA/TransformerEngine/pkgs/container/transformer-engine

```
docker pull ghcr.io/nvidia/transformer-engine:pytorch_te_hf_24.01-py3
```
