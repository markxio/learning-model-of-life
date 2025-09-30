# bionemo-evo2

Overview:

- Get a container up running on kubernetes
- Data to train/fine-tune
- Converting examples as .ipynb notebooks to ipython
- Available evo2 modles with bionemo
- Check the memory and hw requirements per model

## Get a container up running on kubernetes

Use the job-bionemo.yaml config file to create a container:

```
kubectl create -f job-bionemo.yaml
```

Check the container status and get its name:

```
kubectl get po
```

Confirm the container's status has switched to `running` (might take a while for kubernetes to download the image and set up the container). Then copy the name (something like `tklaisoo-eidf107-job-gtqqw-k2fc6 `) and create a session in the container to "log on":

```
kubectl exec -it tklaisoo-eidf107-job-wm9fs-n7g8l -- bash
```

You should be in the container now. We need a writable location that we can clone the bionemo github repository to:

```
git clone https://github.com/NVIDIA/bionemo-framework.git
cd bionemo-framework/sub-packages/bionemo-evo2
```

## Data to train/fine-tune

- 158 MB of fasta data, genomes, human chromosomes
- from Genomics institute, University of California Santa Cruz (USCZ), download: https://hgdownload.soe.ucsc.edu/goldenpath/hg38/chromosomes/chr20.fa.gz
- fasta is a text-based format for representing either nucleotide sequences or amino acid (protein) sequences in bioinformatics, see https://www.ncbi.nlm.nih.gov/genbank/fastaformat/
- split train/validation/test: 90%/5%/5% 
- preprocess once: from text-based sequences to binary tokens compatible with evo2 (padding shorter sequences/append end-of-sequence tokens)
- evo2 preprocessing script: https://github.com/Zymrael/savanna/blob/main/tools/preprocess_data.py

## Converting examples as .ipynb notebooks to ipython

The bionemo framework provides two tutorials as starting point, which are provided as .ipynb:
- Finetuning
- Zeroshot BRCA1 Variant Effect Prediction

We could run these notebooks in interactive mode but alternatively can translate it to python that is executable on the command line. 

<!--- See: https://stackoverflow.com/questions/35545402/how-to-run-an-ipynb-jupyter-notebook-from-terminal -->

```
cd examples
jupyter nbconvert --to python fine-tuning-tutorial.ipynb
jupyter nbconvert --to python zeroshot_brca1.ipynb
```

If you try to run the generate .py files with e.g., `python fine-tuning-tutorial.py`, python will complain about `get_ipython is undefined`. That's because we have to run the .py files with `ipython fine-tuning-tutorial.py`.

## Available evo2 modles with bionemo

Check: https://huggingface.co/arcinstitute

- savanna_evo2_1b_base (used in the `fine-tuning-tutorial.ipynb`)
- savanna_evo2_7b_base
- savanna_evo2_40b_base
- savanna_evo2_7b
- savanna_evo2_40b

<!--- Try replacing the model in `fine-tuning-tutorial.ipynb` with the 40b_base model. -->

## Check the memory and hw requirements per model

<!---

### evo2-7b

See: https://github.com/NVIDIA/bionemo-framework/issues/986
Question: `How many H100s and how much memory are needed to fine-tune the evo2-7b model?`
Answer: Run `torchrun --nproc-per-node 4 --nnodes 1 /usr/local/bin/train_evo2 -d ./sub-packages/bionemo-evo2/examples/training_data_config.yaml  --dataset-dir ./preprocessed_data --result-dir pretraining_demo --experiment-name evo2 --model-size 7b --devices 4 --num-nodes 1 --seq-length 8192 --micro-batch-size 1 --lr 0.000015 --min-lr 0.0000149 --warmup-steps 100 --grad-acc-batches 4 --max-steps 100  --clip-grad 250 --wd 0.001 --attention-dropout 0.01 --hidden-dropout 0.01 --val-check-interval 50  --create-tensorboard-logger --ckpt-async-save`


The data was taken by pulling from the notebook ./sub-packages/bionemo-evo2/examples/fine-tuning-tutorial.ipynb

<details>

<summar>Some relevant information from the logs:</summary>

```
┏━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┓
┃   ┃ Name                                ┃ Type                   ┃ Params ┃ Mode  ┃
┡━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━┩
│ 0 │ module                              │ DDP                    │  6.5 B │ train │
│ 1 │ module.module                       │ Float16Module          │  6.5 B │ train │
│ 2 │ module.module.module                │ HyenaModel             │  6.5 B │ train │
│ 3 │ module.module.module.embedding      │ LanguageModelEmbedding │  2.1 M │ train │
│ 4 │ module.module.module.rotary_pos_emb │ RotaryEmbedding        │      0 │ train │
│ 5 │ module.module.module.decoder        │ HyenaStack             │  6.5 B │ train │
│ 6 │ module.module.module.output_layer   │ ColumnParallelLinear   │      0 │ train │
└───┴─────────────────────────────────────┴────────────────────────┴────────┴───────┘
Trainable params: 6.5 B
Non-trainable params: 0
Total params: 6.5 B
Total estimated model params size (MB): 25.9 K
Modules in train mode: 452
Modules in eval mode: 0

...

Training epoch 0, iteration 32/99 | lr: 4.8e-06 | global_batch_size: 16 | global_step: 32 | reduced_train_loss: 1.436 | train_step_timing in s: 4.929 | consumed_samples: 528
Training epoch 0, iteration 33/99 | lr: 4.95e-06 | global_batch_size: 16 | global_step: 33 | reduced_train_loss: 1.483 | train_step_timing in s: 4.938 | consumed_samples: 544
Training epoch 0, iteration 34/99 | lr: 5.1e-06 | global_batch_size: 16 | global_step: 34 | reduced_train_loss: 1.58 | train_step_timing in s: 4.93 | consumed_samples: 560
```

nvidia-smi output:
```
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.216.03             Driver Version: 535.216.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA H100 80GB HBM3          On  | 00000000:19:00.0 Off |                    0 |
| N/A   68C    P0             689W / 700W |  66809MiB / 81559MiB |    100%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   1  NVIDIA H100 80GB HBM3          On  | 00000000:2D:00.0 Off |                    0 |
| N/A   60C    P0             686W / 700W |  66861MiB / 81559MiB |    100%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   2  NVIDIA H100 80GB HBM3          On  | 00000000:3F:00.0 Off |                    0 |
| N/A   59C    P0             691W / 700W |  66595MiB / 81559MiB |    100%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   3  NVIDIA H100 80GB HBM3          On  | 00000000:66:00.0 Off |                    0 |
| N/A   54C    P0             688W / 700W |  66595MiB / 81559MiB |    100%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
```
</details>
-->

## evo2-1b

<details>
<summary>evo2-1b on 1x H200 GPU</summary>

Run:
```
torchrun --nproc-per-node 1 --nnodes 1 /usr/local/bin/train_evo2 -d ./sub-packages/bionemo-evo2/examples/training_data_config.yaml  --dataset-dir ./preprocessed_data --result-dir pretraining_demo --experiment-name evo2 --model-size 1b --devices 1 --num-nodes 1 --seq-length 8192 --micro-batch-size 1 --lr 0.000015 --min-lr 0.0000149 --warmup-steps 100 --grad-acc-batches 4 --max-steps 100  --clip-grad 250 --wd 0.001 --attention-dropout 0.01 --hidden-dropout 0.01 --val-check-interval 50  --create-tensorboard-logger --ckpt-async-save
```

some output:
```
┏━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┓
┃   ┃ Name                                ┃ Type                   ┃ Params ┃ Mode  ┃
┡━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━┩
│ 0 │ module                              │ DDP                    │  1.1 B │ train │
│ 1 │ module.module                       │ Float16Module          │  1.1 B │ train │
│ 2 │ module.module.module                │ HyenaModel             │  1.1 B │ train │
│ 3 │ module.module.module.embedding      │ LanguageModelEmbedding │  983 K │ train │
│ 4 │ module.module.module.rotary_pos_emb │ RotaryEmbedding        │      0 │ train │
│ 5 │ module.module.module.decoder        │ HyenaStack             │  1.1 B │ train │
│ 6 │ module.module.module.output_layer   │ ColumnParallelLinear   │      0 │ train │
└───┴─────────────────────────────────────┴────────────────────────┴────────┴───────┘
Trainable params: 1.1 B                                                                                                                 
Non-trainable params: 0                                                                                                                 
Total params: 1.1 B                                                                                                                     
Total estimated model params size (MB): 4.4 K                                                                                           
Modules in train mode: 356                                                                                                              
Modules in eval mode: 0
```

</details>

## evo2-7b

<details>

<summary>evo2-7b on 1x H200 GPU</summary>

Run:
```
torchrun --nproc-per-node 1 --nnodes 1 /usr/local/bin/train_evo2 -d ./sub-packages/bionemo-evo2/examples/training_data_config.yaml  --dataset-dir ./preprocessed_data --result-dir pretraining_demo --experiment-name evo2 --model-size 7b --devices 1 --num-nodes 1 --seq-length 8192 --micro-batch-size 1 --lr 0.000015 --min-lr 0.0000149 --warmup-steps 100 --grad-acc-batches 4 --max-steps 100  --clip-grad 250 --wd 0.001 --attention-dropout 0.01 --hidden-dropout 0.01 --val-check-interval 50  --create-tensorboard-logger --ckpt-async-save
```

Error:
```
[rank0]: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 344.00 MiB. GPU 0 has a total capacity of 139.81 GiB of which 2.00 MiB is free. Process 3131906 has 139.80 GiB memory in use. Of the allocated memory 137.97 GiB is allocated by PyTorch, and 704.56 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
```

</details>

<details>

<summary>evo2-7b on 2x H200 GPU</summary>

Run:
```
torchrun --nproc-per-node 2 --nnodes 1 /usr/local/bin/train_evo2 -d ./sub-packages/bionemo-evo2/examples/training_data_config.yaml  --dataset-dir ./preprocessed_data --result-dir pretraining_demo --experiment-name evo2 --model-size 7b --devices 2 --num-nodes 1 --seq-length 8192 --micro-batch-size 1 --lr 0.000015 --min-lr 0.0000149 --warmup-steps 100 --grad-acc-batches 4 --max-steps 100  --clip-grad 250 --wd 0.001 --attention-dropout 0.01 --hidden-dropout 0.01 --val-check-interval 50  --create-tensorboard-logger --ckpt-async-save
```

```
┏━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┓
┃   ┃ Name                                ┃ Type                   ┃ Params ┃ Mode  ┃
┡━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━┩
│ 0 │ module                              │ DDP                    │  6.5 B │ train │
│ 1 │ module.module                       │ Float16Module          │  6.5 B │ train │
│ 2 │ module.module.module                │ HyenaModel             │  6.5 B │ train │
│ 3 │ module.module.module.embedding      │ LanguageModelEmbedding │  2.1 M │ train │
│ 4 │ module.module.module.rotary_pos_emb │ RotaryEmbedding        │      0 │ train │
│ 5 │ module.module.module.decoder        │ HyenaStack             │  6.5 B │ train │
│ 6 │ module.module.module.output_layer   │ ColumnParallelLinear   │      0 │ train │
└───┴─────────────────────────────────────┴────────────────────────┴────────┴───────┘
Trainable params: 6.5 B
Non-trainable params: 0
Total params: 6.5 B
Total estimated model params size (MB): 25.9 K
Modules in train mode: 452
Modules in eval mode: 0
```

Runs ok!

</details>

## evo2-40b

<details>

<summary>evo2-40b on 2x H200 GPU</summary>

Run:
```
torchrun --nproc-per-node 2 --nnodes 1 /usr/local/bin/train_evo2 -d ./sub-packages/bionemo-evo2/examples/training_data_config.yaml  --dataset-dir ./preprocessed_data --result-dir pretraining_demo --experiment-name evo2 --model-size 40b --devices 2 --num-nodes 1 --seq-length 8192 --micro-batch-size 1 --lr 0.000015 --min-lr 0.0000149 --warmup-steps 100 --grad-acc-batches 4 --max-steps 100  --clip-grad 250 --wd 0.001 --attention-dropout 0.01 --hidden-dropout 0.01 --val-check-interval 50  --create-tensorboard-logger --ckpt-async-save
```

We run out of storage on eidf, try to not create checkpoints during testing with `--disable-checkpointing`

Errors:
```
[rank0]: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 75.13 GiB. GPU 0 has a total capacity of 139.81 GiB of which 63.24 GiB is free. Process 3299062 has 76.56 GiB memory in use. Of the allocated memory 75.13 GiB is allocated by PyTorch, and 18.76 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
[rank1]: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 75.13 GiB. GPU 1 has a total capacity of 139.81 GiB of which 63.24 GiB is free. Process 3299063 has 76.56 GiB memory in use. Of the allocated memory 75.13 GiB is allocated by PyTorch, and 18.76 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
```

</details>

<details>

<summary>evo2-40b on 3x H200 GPU</summary>

Run:
```
torchrun --nproc-per-node 3 --nnodes 1 /usr/local/bin/train_evo2 -d training_data_config.yaml  --dataset-dir preprocessed_data --result-dir pretraining_demo --experiment-name evo2 --model-size 40b --devices 3 --num-nodes 1 --seq-length 8192 --micro-batch-size 1 --lr 0.000015 --min-lr 0.0000149 --warmup-steps 100 --grad-acc-batches 4 --max-steps 100  --clip-grad 250 --wd 0.001 --attention-dropout 0.01 --hidden-dropout 0.01 --val-check-interval 50  --create-tensorboard-logger --disable-checkpointing
```

Errors:
```
[rank0]: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 75.13 GiB. GPU 0 has a total capacity of 139.81 GiB of which 63.19 GiB is free. Process 3314890 has 76.61 GiB memory in use. Of the allocated memory 75.13 GiB is allocated by PyTorch, and 18.76 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
[rank1]: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 75.13 GiB. GPU 1 has a total capacity of 139.81 GiB of which 63.19 GiB is free. Process 3314891 has 76.61 GiB memory in use. Of the allocated memory 75.13 GiB is allocated by PyTorch, and 18.76 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
[rank2]: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 75.13 GiB. GPU 2 has a total capacity of 139.81 GiB of which 63.19 GiB is free. Process 3314892 has 76.61 GiB memory in use. Of the allocated memory 75.13 GiB is allocated by PyTorch, and 18.76 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
```

</details>

<details>

<summary>evo2-40b on 4x H200 GPU</summary>

Run:
```
torchrun --nproc-per-node 4 --nnodes 1 /usr/local/bin/train_evo2 -d training_data_config.yaml  --dataset-dir preprocessed_data --result-dir pretraining_demo --experiment-name evo2 --model-size 40b --devices 4 --num-nodes 1 --seq-length 8192 --micro-batch-size 1 --lr 0.000015 --min-lr 0.0000149 --warmup-steps 100 --grad-acc-batches 4 --max-steps 100  --clip-grad 250 --wd 0.001 --attention-dropout 0.01 --hidden-dropout 0.01 --val-check-interval 50  --create-tensorboard-logger --disable-checkpointing
```
pending... 

</details>

<details>

<summary>evo2-40b on 4x H100 80 GB GPUs</summary>

Run:
```
torchrun --nproc-per-node 4 --nnodes 1 /usr/local/bin/train_evo2 -d training_data_config.yaml  --dataset-dir preprocessed_data --result-dir pretraining_demo --experiment-name evo2 --model-size 40b --devices 4 --num-nodes 1 --seq-length 8192 --micro-batch-size 1 --lr 0.000015 --min-lr 0.0000149 --warmup-steps 100 --grad-acc-batches 4 --max-steps 100  --clip-grad 250 --wd 0.001 --attention-dropout 0.01 --hidden-dropout 0.01 --val-check-interval 50  --create-tensorboard-logger --disable-checkpointing
```

Results:
had to increase mem from 256GiB to 512 GiB

Errors:
```
[rank3]: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 75.13 GiB. GPU 3 has a total capacity of 79.25 GiB of which 3.02 GiB is free. Process 1142987 has 76.22 GiB memory in use. Of the allocated memory 75.13 GiB is allocated by PyTorch, and 18.76 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
[rank1]: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 75.13 GiB. GPU 1 has a total capacity of 79.25 GiB of which 3.02 GiB is free. Process 1142985 has 76.22 GiB memory in use. Of the allocated memory 75.13 GiB is allocated by PyTorch, and 18.76 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
[rank2]: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 75.13 GiB. GPU 2 has a total capacity of 79.25 GiB of which 3.02 GiB is free. Process 1142986 has 76.22 GiB memory in use. Of the allocated memory 75.13 GiB is allocated by PyTorch, and 18.76 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
[rank0]: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 75.13 GiB. GPU 0 has a total capacity of 79.25 GiB of which 3.02 GiB is free. Process 1142984 has 76.22 GiB memory in use. Of the allocated memory 75.13 GiB is allocated by PyTorch, and 18.76 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
```

</details>

<details>

<summary>evo2-40b on 8x H100 80 GB GPUs</summary>

pending...

</details>


## Experiment

1. inference: generate USCZ sequences based on ???, test for accuracy
2. fine-tune on USZC data
3. inference: compare predictive accuracy to 1. (how to compare generated data? exact match? similarity? conservative/scrict metric vs relaxed threshold metric)