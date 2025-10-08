# Learning Model of Life (LML)

> For decades, the high-throughput era of biology has generated data faster than it can be interpreted. Advances in artificial intelligence and engineering biology make it possible not only to collate this data and decode the rules of biology, but to create a model that designs and conducts its own experiments—a biological singularity.

Overview:

- evo2
- savanna
- bionemo-framework

## evo2

[Evo2](https://github.com/ArcInstitute/evo2) is a multi-hybrid foundation model for genome generation and understanding across all domains of life. The [Learning Model of Life (LML)](https://lml.ac.uk/) investigates alignment and adaptation of evo2 to various local datasets and the generation of novel genoms. Kubernetes yaml configurations and scripts for installation on the GAIL partition of EIDF are provided in the [evo2](evo2/) directory. Task included:
- sequence generation with evo2
- zero-shot inference of brca1 variant effects with evo2

## savanna

Evo2 was pre-trained and fine-tuned using [savanna](https://github.com/Zymrael/savanna), a training infrastructure for multi-hybrid models. Scripts for installing savanna are provided in the [savanna](savanna/) directory.

## bionemo-framework

[NVIDIA BioNeMo Framework](https://github.com/NVIDIA/bionemo-framework) 

> is a comprehensive suite of programming tools, libraries, and models designed for computational drug discovery

which includes `bionemo-evo2`, a subpackage building on Megatron-LM parallelism and NeMo2 algorithms. More information on running `bionemo-evo2` are provided in the [`bionemo-evo2`](bionemo-framework/) directory. Tasks included:
- fine-tuning evo2 to a local dataset
- zero-shot inference with evo2
