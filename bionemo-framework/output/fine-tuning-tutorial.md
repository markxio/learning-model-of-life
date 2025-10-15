## Fine-tuning tutorial for Evo2: Adapt the 1b evo2 checkpoint for your hardware
Deploy tutorial on brev.dev:
[![ Click here to deploy.](https://brev-assets.s3.us-west-1.amazonaws.com/nv-lb-dark.svg)](https://console.brev.dev/launchable/deploy?launchableID=env-2uGqijcTiNxv3V8LZJxXAa7KlKC)

### Background and motivation
To motivate this tutorial, we have noticed that the public
evo2 checkpoint in hugging face for the 1b model is sensitive to `--fp8` status in training, the zero shot inference
task, as demonstrated in the zero shot BRCA-1 notebook, produces near random AUCs if you do not use `--fp8`. 
If you want to infer or score new data, you need FP8 enabled since it was trained that way. Interestingly the `7b` checkpoint does not suffer from this
limitation and seems robust to FP8 being activated or not. The consequence of this is that if you have older GPUs with
a compute capability less than 8.9, which do not support FP8, then the output that you get from scoring sequences with
sensitive checkpoints may not be biologically meaningful. 

We plan on making
a `1b` parameter evo2 checkpoint available soon that has been fine-tuned to be robust to FP8 or BF16 inference in bionemo
on NGC, but in the meantime this notebook tutorial outlines the steps for fine-tuning. The only difference between this
notebook and what we did in production was to run these steps on more data on a slurm cluster to increase the global
batch size. That said, if you run this for enough steps to get loss on the 1b checkpoint to the 1.08 range, you should 
have good luck with downstream sequence scoring tasks. 

### Requirements

This is a tutorial demonstrating how you can fine-tune Evo2 on new data and/or hardware. The tutorial should take 
slightly under 1 hour to run on an RTX A6000 in bf16 precision.

As configured, this tutorial requires an NVIDIA GPU with approximately 45GB of ram. If you have multiple GPUs with less
memory, or you are having trouble with CUDA OOM at the training step below, try reducing the `--micro-batch-size` and/or
increasing the number of `--devices [int]` to match your setup and also setting `--tensor-parallel-size [int]` to
the number of devices. This should split up most of the model evenly between your devices, which will require much less
memory. When we train the 1b model in practice we typically have the micro batch size set to 8, and run without model 
parallelism on available devices to achieve the largest possible global batch size.


```python
import os


# This variable should be used in the notebooks to run a subset of the model layers or a smaller model/dataset
FAST_CI_MODE: bool = os.environ.get("FAST_CI_MODE", False)
# Clean up any prior runs
CLEANUP: bool = False
if CLEANUP:
    !rm -rf preprocessed_data
    !rm -rf preatraining_demo
    !rm -rf pretraining_demo
    !rm -rf training_data_config.yaml
    !rm -rf preprocess_config.yaml
    !rm -f chr20.fa.gz
    !rm -f chr21.fa.gz
    !rm -f chr22.fa.gz
    !rm -f chr20_21_22.fa
```

### Setup training data
Evo2 uses megatron style datasets behind the scenes with advanced support for randomly indexing into documents, and
packing documents together into batches at scale. The file-formats backing these datasets is not a standard biological
format like fasta for representing genomes. First we show how you can start from a fasta file and preprocess them into
the required data format for downstream handling. High level the steps are as follows:
1. Acquire fasta files locally, ideally in some shared cluster storage
2. Write a config script defining how you want the processed files to be generated from the fasta files. This is where
  you specify top level train/validation/test splitting decisions.
3. Call the actual `preprocess_evo2` script to generate the results.

The next 4 cells go through this process on a set of smaller human chromosomes. At least 3 fasta records need to be present,
one for the train, validation, and test split.


```python
%%capture
import os

from bionemo.core.utils.subprocess_utils import run_subprocess_safely


concat_path = "chr20_21_22.fa"
if not os.path.exists(concat_path):
    !wget https://hgdownload.soe.ucsc.edu/goldenpath/hg38/chromosomes/chr20.fa.gz
    !wget https://hgdownload.soe.ucsc.edu/goldenpath/hg38/chromosomes/chr21.fa.gz
    !wget https://hgdownload.soe.ucsc.edu/goldenpath/hg38/chromosomes/chr22.fa.gz
    !zcat chr20.fa.gz > chr20.fa
    !zcat chr21.fa.gz > chr21.fa
    !zcat chr22.fa.gz > chr22.fa
    !cat chr20.fa chr21.fa chr22.fa > chr20_21_22.fa
```


```python
full_fasta_path = os.path.abspath(concat_path)
output_dir = os.path.abspath("preprocessed_data")
output_yaml = f"""
- datapaths: ["{full_fasta_path}"]
  output_dir: "{output_dir}"
  output_prefix: chr20_21_22_uint8_distinct
  train_split: 0.9
  valid_split: 0.05
  test_split: 0.05
  overwrite: True
  embed_reverse_complement: true
  random_reverse_complement: 0.0
  random_lineage_dropout: 0.0
  include_sequence_id: false
  transcribe: "back_transcribe"
  force_uppercase: false
  indexed_dataset_dtype: "uint8"
  tokenizer_type: "Byte-Level"
  vocab_file: null
  vocab_size: null
  merges_file: null
  pretrained_tokenizer_model: null
  special_tokens: null
  fast_hf_tokenizer: true
  append_eod: true
  enforce_sample_length: null
  ftfy: false
  workers: 1
  preproc_concurrency: 100000
  chunksize: 25
  drop_empty_sequences: true
  nnn_filter: false  # If you split your fasta on NNN (in human these are contigs), then you should set this to true.
  seed: 12342  # Not relevant because we are not using random reverse complement or lineage dropout.
"""
with open("preprocess_config.yaml", "w") as f:
    print(output_yaml, file=f)
```


```python
%%capture
!preprocess_evo2 --config preprocess_config.yaml
```


```python
# There should be a collection of bin/idx files created in the preprocessed_data directory.
!ls -lh preprocessed_data/
```

    total 309M
    drwxr-xr-x 3 root root    1 Aug 20 13:22 chr20_21_22_uint8_distinct_byte-level_test
    -rw-r--r-- 1 root root 123M Oct 15 15:33 chr20_21_22_uint8_distinct_byte-level_test.bin
    -rw-r--r-- 1 root root   82 Oct 15 15:33 chr20_21_22_uint8_distinct_byte-level_test.idx
    drwxr-xr-x 3 root root    1 Aug 20 13:22 chr20_21_22_uint8_distinct_byte-level_train
    -rw-r--r-- 1 root root  97M Oct 15 15:33 chr20_21_22_uint8_distinct_byte-level_train.bin
    -rw-r--r-- 1 root root   82 Oct 15 15:33 chr20_21_22_uint8_distinct_byte-level_train.idx
    drwxr-xr-x 3 root root    1 Aug 20 13:22 chr20_21_22_uint8_distinct_byte-level_val
    -rw-r--r-- 1 root root  90M Oct 15 15:33 chr20_21_22_uint8_distinct_byte-level_val.bin
    -rw-r--r-- 1 root root   82 Oct 15 15:33 chr20_21_22_uint8_distinct_byte-level_val.idx


### [Optional] specify or convert initial checkpoint
The main difference between pre-training and fine-tuning is whether or not you decide to start training the model with
weights from a prior training run. For this tutorial we want to tune a `1b` checkpoint from hugging face that is known
(at the time of this writing) to be sensitive to GPU architecture so that it will work with your architecture. We have a
script that will download and convert a savanna format evo2 checkpoint from hugging face, and output that into a NeMo2
format checkpoint directory that can be used as the starting point for a fine-tuning run.


```python
%%capture
if not os.path.exists("nemo2_evo2_1b_8k"):
    !evo2_convert_to_nemo2 \
      --model-path hf://arcinstitute/savanna_evo2_1b_base \
      --model-size 1b --output-dir nemo2_evo2_1b_8k
```

### Configure the training dataset
The next step is to configure your training dataset, in this case configuring the simple single-file example we output
two steps ago in this tutorial. 


```python
from pathlib import Path


output_pfx = str(Path(os.path.abspath("preprocessed_data")) / "chr20_21_22_uint8_distinct_byte-level")
output_yaml = f"""
- dataset_prefix: {output_pfx}_train
  dataset_split: train
  dataset_weight: 1.0
- dataset_prefix: {output_pfx}_val
  dataset_split: validation
  dataset_weight: 1.0
- dataset_prefix: {output_pfx}_test
  dataset_split: test
  dataset_weight: 1.0
"""
with open("training_data_config.yaml", "w") as f:
    print(output_yaml, file=f)
```

This next cell takes approximately 25 minutes to run on an RTX A6000 with `MAX_STEPS=100`. Each step takes about 9.5 seconds with the 
following configuration, so you can budget a desired number of max steps to try.


```python
%%capture
MAX_STEPS: int = 10 if FAST_CI_MODE else 100
val_check_interval = min(int(MAX_STEPS // 2), 50)
warmup_steps = min(MAX_STEPS, 100)
# For evo2 training and fine-tuning follow the same set of steps, so we use the same train_evo2 command.
#  the big difference is the --ckpt-dir argument which points to a pre-existing checkpoint from some other training run.

if FAST_CI_MODE:
    model_subset_option = (
        "--num-layers 4 --hybrid-override-pattern SDH* --activation-checkpoint-recompute-num-layers 2"
    )
else:
    # By default do 5 layers of activation checkpointing
    model_subset_option = "--activation-checkpoint-recompute-num-layers 5"
train_cmd = f"""train_evo2 \
    -d training_data_config.yaml \
    --dataset-dir ./preprocessed_data \
    --result-dir pretraining_demo \
    --experiment-name evo2 \
    --model-size 1b \
    --devices 1 \
    --num-nodes 1 \
    --seq-length 8192 \
    --micro-batch-size 2 \
    --lr 0.000015 \
    --min-lr 0.0000149 \
    --warmup-steps {warmup_steps} \
    --grad-acc-batches 4 \
    --max-steps {MAX_STEPS} \
    --ckpt-dir nemo2_evo2_1b_8k \
    --clip-grad 250 \
    --wd 0.001 \
    --attention-dropout 0.01 \
    --hidden-dropout 0.01 \
    --val-check-interval {val_check_interval} \
    {model_subset_option} \
    --create-tensorboard-logger \
    --ckpt-async-save"""

print(f"Running command: {train_cmd}")

result = run_subprocess_safely(train_cmd)
```


```python
assert result["returncode"] == 0, result
```

The plotting code is hidden in documentation for brevity. You can view the notebook on github, run it in jupyter-lab or launch the tutorial on brev.dev if you want to view the source.


```python
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorboard.backend.event_processing.event_accumulator as event_accumulator


# Function to extract data from TensorBoard event files and convert to DataFrame
def tensorboard_to_dataframe(event_file):
    """Given a TensorBoard event file, return a pandas DataFrame with the training metrics."""
    # Load the event file
    ea = event_accumulator.EventAccumulator(
        event_file,
        size_guidance={
            event_accumulator.SCALARS: 0,  # 0 means load all
        },
    )
    ea.Reload()

    # Get list of all available tags
    tags = ea.Tags()["scalars"]

    # First, find the union of all steps
    all_steps = set()
    for tag in tags:
        events = ea.Scalars(tag)
        steps = [event.step for event in events]
        all_steps.update(steps)

    # Sort steps for proper ordering
    all_steps = sorted(all_steps)

    # Initialize the dataframe with steps
    df = pd.DataFrame({"step": all_steps})

    # Add each metric as a column
    for tag in tags:
        events = ea.Scalars(tag)
        # Create a dictionary mapping steps to values
        step_to_value = {event.step: event.value for event in events}
        # Add the values to the dataframe, using NaN for missing steps
        df[tag] = df["step"].map(step_to_value)

    return df


# Example of creating a multi-metric plot with seaborn
def plot_multiple_training_metrics(df, metrics_to_plot, figsize=(15, 10)):
    """Given a pandas DataFrame with the training metrics, plot the metrics."""
    n = len(metrics_to_plot)
    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True)

    if n == 1:  # Handle the case of a single plot
        axes = [axes]

    sns.set_style("whitegrid")

    for i, metric in enumerate(metrics_to_plot):
        if metric in df.columns:
            sns.lineplot(x="step", y=metric, data=df, ax=axes[i], linewidth=2.5, errorbar="sd")
            axes[i].set_title(metric, fontsize=14)
            axes[i].set_ylabel("Value", fontsize=12)
    axes[-1].set_xlabel("Steps", fontsize=14)
    plt.tight_layout()
    plt.show()
```

The following figures show various training metrics per step.
* `reduced_train_loss` captures the training loss. On larger runs you want to see the loss drop to about 1.08 consistently
  for the 1b checkpoint.
* `lr` shows the learning rate schedule for training. Typically we do a linear warmup schedule followed by an cosine decay.
  this small notebook tutorial just goes through the initial warmup period.
* `grad_norm` shows the gradient norm of the full model. As the model fits the data better you should see this value drop
  down below 1.0 consistently. 
* `val_loss` shows the same kind of loss shown in `reduced_train_loss` but for a held-out set of validation samples. If you
  ever train the model a very long time and see this start to go up while the training loss continues to drop that's a sign
  of over-fitting. We have not yet seen this happen. Small fluctuations up and down are expected during training.


```python
# Get the TensorBoard event file for the training run
log_dirs = !find pretraining_demo/evo2/dev -name "events.out.tfevents*"
tf_event_file = log_dirs[0]

# Extract data from your event file
df = tensorboard_to_dataframe(tf_event_file)
# You can uncomment and modify this to plot multiple metrics once you see what's available
plot_multiple_training_metrics(df, ["reduced_train_loss", "lr", "grad_norm", "val_loss"])
```


    
![png](fine-tuning-tutorial_files/fine-tuning-tutorial_17_0.png)
    


Now you have a checkpoint that you can try out in place of the converted evo2 checkpoint in the BRCA-1 tutorial 
(the path is displayed in the next code cell). To test your checkpoint, please supply the following path to the saved 
checkpoint produced by this notebook as the `--ckpt-dir {checkpoint_path}`
argument to the `predict_evo2` command in the zero shot BRCA tutorial. For the 1b checkpoint you should see AUC above
0.73 if you successfully fine-tuned the checkpoint for your hardware, or to check that your hardware works with the 
converted checkpoint from hugging face as is.

In our experience running this notebook for up to an hour on a single GPU is not sufficient to recover BF16 accuracy. We
have more details about what did work in the Next Steps section below.


```python
final_ckpt_paths = !ls -d pretraining_demo/evo2/checkpoints/*-last
final_ckpt_path = final_ckpt_paths[-1]
final_ckpt_path
```




    'pretraining_demo/evo2/checkpoints/epoch=0-step=99-consumed_samples=800.0-last'



### Next steps
On a small number of devices, or with the small demo fasta we provided in this tutorial, it's possible you are not at the needed
1.08 loss level to get good downstream accuracy out of this checkpoint. You can try increasing the `MAX_STEPS` parameter in the training cell,
or running a larger cluster with more GPUs. The following loss curve was generated with a global batch size of 256 at 8192 context or approximately
2 million tokens per step. With that configuration we see a good loss of 1.08 after approximately 100 steps. The following figure shows our
learning rate across the first 500 steps of fine-tuning with a global batch size of 256. Later on in this notebook we also show the slurm script
to replicate this on your cluster.



```python
# Display the example loss curve from a larger training run
from IPython.display import Image, display


# Load and display the image
display(Image("../assets/1b_finetuning_train_curve_500_steps_256gbs.png", width=800))
```


    
![png](fine-tuning-tutorial_files/fine-tuning-tutorial_21_0.png)
    


#### How we fine-tuned the 1b checkpoint for bf16 accuracy
An example of the full slurm script to run the above training curve on our infrastructure is as follows:

First make a `~/.netrc` file with your wandb login info. You can also accomplish this by setting wandb ENV variables,
assuming you want to log to wandb. If not you can pass the `--no-wandb` argument as part of the args to `train_evo2`:

```ini
machine api.wandb.ai
  login user
  password PASSWORD_HERE
```

Next, paste/edit the following sbatch script for your own configuration:

```bash
# TODO: You may need to add more SBATCH configuration here specific to your cluster.
#SBATCH --nodes=4                       # number of nodes
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8                 # n tasks per machine (one task per gpu) <required>
#SBATCH --time=04:00:00                     # wall time  (8 for batch, backfill, 2 for batch_short)
#SBATCH --mem=0                             # all mem avail
#SBATCH --exclusive
set -x
# You may want to edit this file and/or add your own version to your mounts.
CONFIG_PATH_IN_CONTAINER=/workspace/bionemo2/sub-packages/bionemo-evo2/examples/configs/full_pretrain_shortphase_config.yaml
# You can build a `.sqsh` file with enroot which may be faster to load on each node rather than pulling down from NGC
IMAGE_PATH=nvcr.io/nvidia/clara/bionemo-framework:nightly
WANDB_PROJECT_NAME= # Set you wandb project here, or leave blank and add --no-wandb to the image
MODEL_SIZE=1b  # change this to 7b_arc_longcontext etc. This version is different.
CP_SIZE=1
TP_SIZE=1
PP_SIZE=1
MICRO_BATCH_SIZE=8
GRAD_ACC_BATCHES=1
SEQ_LEN=8192
MAX_STEPS=580000 # 8T tokens given 1024 nodes and 8192 seq length
VAL_CHECK=500
CLIP_GRAD=250  # Arc trained without gradient clipping. Set to a large value so megatron still logs grad_norm.
# The following arguments will remove the EOD/PAD tokens from the loss, unlike how the original Evo2 model was trained.
#  this does not impact downstream accuracy in our experience and is more standard.
EXTRA_ARGS="--enable-preemption --ckpt-async-save --overlap-grad-reduce --clip-grad $CLIP_GRAD --eod-pad-in-loss-mask"
LR=0.000015
MIN_LR=0.0000015
WU_STEPS=100
SEED=1234 
WD=0.001
ADO=0.01
HDO=0.01
EXPERIMENT_NAME=fine_tune_evo2_1b_on_bf16
# NCCL performance parameters
# =========================
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

# Mounts
# =========================
DATA_PATH= # PATH to the directory that stores your data that you want to mount into the container
DATA_MOUNT=/workspace/bionemo2/data  # or if you configure your data with a different base dir in the config, use that here
RESULTS_PATH_CLUSTER= # Where do you want the results to land on your shared cluster storage
RESULTS_PATH_IMAGE=/results/
CKPT_MOUNT_CLUSTER= # Path to shared location on your cluster where the checkpoint files can be found
CKPT_MOUNT_IMAGE=/checkpoints/  # pragma: allowlist secret  (for some reason this line flags a high entropy string check in CI)
NETRC_PATH=$HOME/.netrc
NETRC_MOUNT=/root/.netrc
# TODO either move your config to one of the mounted paths or add your own mount to a location with your configs

mkdir -p $RESULTS_PATH_CLUSTER
MOUNTS=${DATA_PATH}:${DATA_MOUNT},${RESULTS_PATH_CLUSTER}:${RESULTS_PATH_IMAGE},${NETRC_PATH}:${NETRC_MOUNT},${CKPT_MOUNT_CLUSTER}:${CKPT_MOUNT_IMAGE},$HOME/.cache:/root/.cache
# Generate (or retrieve) a unique, shared ID per run to handle restarts in W&B and Tensorboard
# =========================
mkdir -p ${RESULTS_PATH_CLUSTER}
if [ -f ${RESULTS_PATH_CLUSTER}/run.id ];
then
    RUN_ID=$(<${RESULTS_PATH_CLUSTER}/run.id)
else
    array=()
    for i in {a..z} {A..Z} {0..9};
    do
    array[$RANDOM]=$i
    done
    RUN_ID=$(printf %s ${array[@]::8})
    echo $RUN_ID > ${RESULTS_PATH_CLUSTER}/run.id
fi
# =========================
read -r -d '' COMMAND <<EOF
echo "*******STARTING********" \
&& echo "---------------" \
&& echo "Starting training" \
&&  \
train_evo2 \
    -d $CONFIG_PATH_IN_CONTAINER \
    --num-nodes=${SLURM_JOB_NUM_NODES} \
    --ckpt-dir $CKPT_MOUNT_IMAGE/nemo2_evo2_1b_8k \
    --devices=${SLURM_NTASKS_PER_NODE} \
    --grad-acc-batches $GRAD_ACC_BATCHES \
    --max-steps=$MAX_STEPS \
    --seed $SEED \
    ${EXTRA_ARGS} \
    --wandb-run-id $RUN_ID \
    --wandb-project $WANDB_PROJECT_NAME \
    --lr $LR \
    --wd $WD \
    --activation-checkpoint-recompute-num-layers 5 \
    --min-lr $MIN_LR \
    --warmup-steps $WU_STEPS \
    --attention-dropout $ADO \
    --hidden-dropout $HDO \
    --limit-val-batches=20 \
    --val-check-interval=${VAL_CHECK} \
    --result-dir=$RESULTS_PATH_IMAGE \
    --seq-length=${SEQ_LEN} \
    --tensor-parallel-size=${TP_SIZE} \
    --context-parallel-size=${CP_SIZE} \
    --pipeline-model-parallel-size=${PP_SIZE} \
    --workers 8 \
    --micro-batch-size=${MICRO_BATCH_SIZE} \
    --model-size=${MODEL_SIZE}
EOF
srun \
    --output ${RESULTS_PATH_CLUSTER}/slurm-%j.out \
    --error ${RESULTS_PATH_CLUSTER}/error-%j.out \
    --container-image=$IMAGE_PATH \
    --container-mounts ${MOUNTS} \
    bash -c "${COMMAND}"
set +x

```
