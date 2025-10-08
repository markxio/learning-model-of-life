# Sequence Generation and Alignment Analysis with Evo2
This notebook demonstrates how to generate biological sequences using the Evo2 model and analyze them using Biopython alignments.

## Setup and Dependencies

First, let's import our required libraries and set up our environment. Note you need Jupyter to run notebooks.



```python
import os
import argparse
import csv
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
from Bio.Seq import Seq

from evo2 import Evo2

# Set random seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)


```

    /workspace/writeable/evo2/venv_python3.11/lib/python3.11/site-packages/Bio/pairwise2.py:278: BiopythonDeprecationWarning: Bio.pairwise2 has been deprecated, and we intend to remove it in a future release of Biopython. As an alternative, please consider using Bio.Align.PairwiseAligner as a replacement, and contact the Biopython developers if you still need the Bio.pairwise2 module.
      warnings.warn(
    /workspace/writeable/evo2/venv_python3.11/lib/python3.11/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm


## Model Initialization
Let's initialize our Evo2 model. We'll use the 7B parameter version as a default.


```python
model_name = 'evo2_7b'

model = Evo2(model_name)
```

    Fetching 4 files:   0%|                                                        | 0/4 [00:00<?, ?it/s]

    Fetching 4 files: 100%|█████████████████████████████████████████████| 4/4 [00:00<00:00, 18787.48it/s]

    


    Found complete file in repo: evo2_7b.pt


      0%|                                                                         | 0/32 [00:00<?, ?it/s]

     12%|████████▏                                                        | 4/32 [00:00<00:00, 39.61it/s]

     78%|█████████████████████████████████████████████████▏             | 25/32 [00:00<00:00, 139.35it/s]

    100%|███████████████████████████████████████████████████████████████| 32/32 [00:00<00:00, 138.57it/s]

    


    Extra keys in state_dict: {'blocks.27.mixer.mixer.filter.t', 'blocks.17.mixer.dense._extra_state', 'blocks.6.mixer.mixer.filter.t', 'blocks.13.mixer.mixer.filter.t', 'unembed.weight', 'blocks.20.mixer.mixer.filter.t', 'blocks.10.mixer.dense._extra_state', 'blocks.3.mixer.dense._extra_state', 'blocks.2.mixer.mixer.filter.t', 'blocks.31.mixer.dense._extra_state', 'blocks.24.mixer.dense._extra_state', 'blocks.24.mixer.attn._extra_state', 'blocks.31.mixer.attn._extra_state', 'blocks.30.mixer.mixer.filter.t', 'blocks.16.mixer.mixer.filter.t', 'blocks.17.mixer.attn._extra_state', 'blocks.3.mixer.attn._extra_state', 'blocks.23.mixer.mixer.filter.t', 'blocks.9.mixer.mixer.filter.t', 'blocks.10.mixer.attn._extra_state'}


    /workspace/writeable/evo2/venv_python3.11/lib/python3.11/site-packages/transformer_engine/pytorch/module/base.py:630: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
      state = torch.load(state, map_location="cuda")


    /workspace/writeable/evo2/vortex/vortex/model/utils.py:153: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
      return torch_load(state, map_location=device)


## Data Loading
Next we'll create functions to load our example sequences



```python
def read_sequences(input_file: Path) -> Tuple[List[str], List[str]]:
    """
    Read input and target sequences from CSV file.
    
    Expected CSV format:
    input_sequence,target_sequence
    ACGTACGT,ACGTACGTAA
    ...
    """
    input_seqs: List[str] = []
    names: List[str] = []
    
    with open(input_file, encoding='utf-8-sig', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            input_seqs.append(row[0])
            if len(row) > 1:
                names.append(row[1])
    
    return input_seqs, names

# Load example data

sequences, names = read_sequences('../../vortex/test/data/prompts.csv')

# For 'autocomplete', we split the data into input and target sequences

input_seqs = [seq[:500] for seq in sequences]
target_seqs = [seq[500:1000] for seq in sequences]

print(f"Loaded {len(sequences)} sequence pairs")
```

    Loaded 4 sequence pairs


### Now it's time to generate!


```python
generations = model.generate(
    input_seqs,
    n_tokens=500,
    temperature=1.0,
)

generated_seqs = generations.sequences
print(generated_seqs)
```

    Initializing inference params with max_seqlen=1000


    /workspace/writeable/evo2/vortex/vortex/model/engine.py:559: UserWarning: Casting complex values to real discards the imaginary part (Triggered internally at ../aten/src/ATen/native/Copy.cpp:308.)
      inference_params.state_dict[layer_idx] = state[..., L - 1].to(dtype=state_dtype)


    Prompt: "GAATAGGAACAGCTCCGGTCTACAGCTCCCAGCGTGAGCGACGCAGAAGACGGTGATTTCTGCATTTCCATCTGAGGTACCGGGTTCATCTCACTAGGGAGTGCCAGACAGTGGGCGCAGGCCAGTGTGTGTGCGCACCGTGCGCGAGCCGAAGCAGGGCGAGGCATTGCCTCACCTGGGAAGCGCAAGGGGTCAGGGAGTTCCCTTTCCGAGTCAAAGAAAGGGGTGATGGACGCACCTGGAAAATCGGGTCACTCCCACCCGAATATTGCGCTTTTCAGACCGGCTTAAGAAACGGCGCACCACGAGACTATATCCCACACCTGGCTCAGAGGGTCCTACGCCCACGGAATCTCGCTGATTGCTAGCACAGCAGTCTGAGATCAAACTGCAAGGCGGCAACGAGGCTGGGGGAGGGGCGCCCGCCATTGCCCAGGCTTGCTTAGGTAAACAAAGCAGCCGGGAAGCTCGAACTGGGTGGAGCCCACCACAGCTCAAGG",	Output: "AGGCCTGCCTGCCTCTGTAGGCTCCACCTCTGGGGGCAGGGCACAGACAAACAAAAAGACATCACAAACCTCTGCAGACTTAAATGTCCCTGTCTGACAGCTTTGAAGAGAGCAGTGGTTCTCCCAGCACGCAGCTGGAGATCTGAGAACGGGCAGACTGCCTCCTCAAGTAGGTCCCTGGCCCCTGACCCCCAAACAGCCTAACAGGACAGTGACTCCAGGGACTCCATGCGGCTCGCCCGCACTGGTGAACTTAGAAGTGAAACCTACCATGAGGCTCACCGCGCTGCAGCCACCCTAAGGTTAACAAAAAAAACAGAGAACTGGCTCCGGCCAGCAAAAGAACACCGAAAACAGGAGCTCAGCGGTAAATCAGCACCGCGCGCCTCAGTAAAGCAGCCGGGTTGCTCCTCCACAGGCAGAGCAACAAAGCCGGCTACCTAGCAGTCCTAAATGGCAGATCCAAATCCCCACACCTCTAGGACAAAGCACCAATTCAA",	Score: -0.46965208649635315
    Prompt: "GACACCATCGAATGGCGCAAAACCTTTCGCGGTATGGCATGATAGCGCCCGGAAGAGAGTCAATTCAGGGTGGTGAATGTGAAACCAGTAACGTTATACGATGTCGCAGAGTATGCCGGTGTCTCTTATCAGACCGTTTCCCGCGTGGTGAACCAGGCCAGCCACGTTTCTGCGAAAACGCGGGAAAAAGTGGAAGCGGCGATGGCGGAGCTGAATTACATTCCCAACCGCGTGGCACAACAACTGGCGGGCAAACAGTCGTTGCTGATTGGCGTTGCCACCTCCAGTCTGGCCCTGCACGCGCCGTCGCAAATTGTCGCGGCGATTAAATCTCGCGCCGATCAACTGGGTGCCAGCGTGGTGGTGTCGATGGTAGAACGAAGCGGCGTCGAAGCCTGTAAAGCGGCGGTGCACAATCTTCTCGCGCAACGCGTCAGTGGGCTGATCATTAACTATCCGCTGGATGACCAGGATGCCATTGCTGTGGAAGCTGCCTGCAC",	Output: "TAATGTTCCGGCGTTATTTCTTGATGTCTCTGACCAGACTCCCATCAACAGTATTATTTTCTCCCATGAAGACGGTACGCGACTGGGCGTGGAGCATCTGGTCGCATTAGGTCACCAGCAAATCGCGCTGTTAGCGGGCCCATTAAGTTCTGTCTCGGCGCGTCTACGTCTGGCGGGCTGGCATAAATATCTCATACGCAATCAAATTCAGCCGATAGCGGTACTGGAAGGTGACTGGAGTGCGCAGTCCGGTTTTGCCCAGACCATGCAAATGCTGAATGAGACGCCGCCGCCCACCGCGCTGCTGGTGGCGAATGATGTGATGGCGGTGGGCGCACTGCGCGCACTGGAACAAGCCAAAATCAGCGTCCCGCAGGAGATGTCGATCATCGGTTATGACGATACTCAGGACAGCTCATATTATATCCCGCCGTTAACCACCGTCAGGCAGGATTTTCGTCTACTGGGGAAAACCGCCGTGGACCGGTTGATCAGCCTGA",	Score: -0.19441889226436615
    Prompt: "GTTAATGTAGCTTAAAACAAAAGCAAGGTACTGAAAATACCTAGACGAGTATATCCAACTCCATAAACAACAAAGGTTTGGTCCCGGCCTTCTTATTGGTTACTAGGAAACTTATACATGCAAGTATCCGCCCGCCAGTGAATACGCCTTCTAAATCATCACTGATCAAAGAGAGCTGGCATCAAGCACACACCCCAAGTGTAGCTCATGACGTCTCGCCTAGCCACACCCCCACGGGAAACAGCAGTAGTAAATATTTAGCAATTAACAAAAGTTAGACTAAGTTATCCTAATAAAGGACTGGTCAATTTCGTGCCAGCAACCGCGGCCATACGATTAGTCCAAATTAATAAGCATACGGCGTAAAGCGTATTAGAAGAATTAAAAAAATAAAGTTAAATCTTATACTAGCTGTTTAAAGCTCAAGATAAGACATAAATAGCCTACGAAAGTGACTTTAATAATCCTAAACATACGATAGCTAGGGTACAAACTGAGAT",	Output: "TAGATACCTCACTATGCCTAGCCATAAACCTAAGTAGTCCCATTAACAAAACTACTCGCCAGAGTACTACAAGCAACAGCTTAAAATTCAAAGGACTTGGCGGTGCTTTATACCCACCTAGAGGAGCCTGTTCTATAATCGATAAACCCCGATAAACCTTACCACCCATTGCTATTCCAGTCTATATACCGCCATCTTCAGCAAACCCTTAAAAGGGCAAGAAGTAAGCAAAAGTATATAACATAAAAAAGTTAGGTCAAGGTGTAACTAATTAGGTGGGAAGAAATGGGCTACATTTTCTATCACAGAATACACTACGAATGACAACCTGAAACAAGGTCATCTGAAGGCGGATTTAGTAGTAAATTAAGACTAGAGTGCTTAATTGAACAAGGCCATGAAGCACGCACACACCGCCCGTCACCCTCCTCAAATATCAACTATAATTACACAATACTTAAACACCAAACAAACCAGAGGAGATAAGTCGTAACAAGGTA",	Score: -0.34271344542503357
    Prompt: "GATCACAGGTCTATCACCCTATTAACCACTCACGGGAGCTCTCCATGCATTTGGTATTTTCGTCTGGGGGGTATGCACGCGATAGCATTGCGAGACGCTGGAGCCGGAGCACCCTATGTCGCAGTATCTGTCTTTGATTCCTGCCTCATCCTATTATTTATCGCACCTACGTTCAATATTACAGGCGAACATACTTACTAAAGTGTGTTAATTAATTAATGCTTGTAGGACATAATAATAACAATTGAATGTCTGCACAGCCACTTTCCACACAGACATCATAACAAAAAATTTCCACCAAACCCCCCCTCCCCCGCTTCTGGCCACAGCACTTAAACACATCTCTGCCAAACCCCAAAAACAAAGAACCCTAACACCAGCCTAACCAGATTTCAAATTTTATCTTTTGGCGGTATGCACTTTTAACAGTCACCCCCCAACTAACACATTATTTTCCCCTCCCACTCCCATACTACTAATCTCATCAATACAACCCCCGC",	Output: "AGACATACCCCAATCTTCCCCATCCATCCCATCTCCCTCTACCCATTTTTAGCTAATTGCCACTCATCGGCCACTGAGTTACCCAATTTCCCTCTCCCATCCATCAACTCATCCAAAGCTAACATTTTCCCCAAAACAGGCACCCTAACAACAGACATGGCCATGGTGGACGTGAATCCCTTCGGGAAAGGTCTGGCCGCGCTGGACCTGCGAGGCAAGGTGCGAGTGCTGGAATACGACCCCTCAGCGCAGCGTTTCCTGATTGCCTACGCCAATGGCGACGTTGCACTCTTTGAATCTTTCACCTCCCTCACTCCAAATAACGCTATATTGCCTGGCCTATTCCTGACCGAGTCTGCCCTGCCGAATTTTGGCGGTCCCGTCGTGACAGCCGCACACACCGTCTCCCTGTCGCCAGATGGCCGTTACCTCGTCGCCAATGCAACATCAAGCAGCGAGGTTGCTGTCGTGCCCGTGAACGCCACTGCGCCGGGTTCGCT",	Score: -1.2539472579956055
    ['AGGCCTGCCTGCCTCTGTAGGCTCCACCTCTGGGGGCAGGGCACAGACAAACAAAAAGACATCACAAACCTCTGCAGACTTAAATGTCCCTGTCTGACAGCTTTGAAGAGAGCAGTGGTTCTCCCAGCACGCAGCTGGAGATCTGAGAACGGGCAGACTGCCTCCTCAAGTAGGTCCCTGGCCCCTGACCCCCAAACAGCCTAACAGGACAGTGACTCCAGGGACTCCATGCGGCTCGCCCGCACTGGTGAACTTAGAAGTGAAACCTACCATGAGGCTCACCGCGCTGCAGCCACCCTAAGGTTAACAAAAAAAACAGAGAACTGGCTCCGGCCAGCAAAAGAACACCGAAAACAGGAGCTCAGCGGTAAATCAGCACCGCGCGCCTCAGTAAAGCAGCCGGGTTGCTCCTCCACAGGCAGAGCAACAAAGCCGGCTACCTAGCAGTCCTAAATGGCAGATCCAAATCCCCACACCTCTAGGACAAAGCACCAATTCAA', 'TAATGTTCCGGCGTTATTTCTTGATGTCTCTGACCAGACTCCCATCAACAGTATTATTTTCTCCCATGAAGACGGTACGCGACTGGGCGTGGAGCATCTGGTCGCATTAGGTCACCAGCAAATCGCGCTGTTAGCGGGCCCATTAAGTTCTGTCTCGGCGCGTCTACGTCTGGCGGGCTGGCATAAATATCTCATACGCAATCAAATTCAGCCGATAGCGGTACTGGAAGGTGACTGGAGTGCGCAGTCCGGTTTTGCCCAGACCATGCAAATGCTGAATGAGACGCCGCCGCCCACCGCGCTGCTGGTGGCGAATGATGTGATGGCGGTGGGCGCACTGCGCGCACTGGAACAAGCCAAAATCAGCGTCCCGCAGGAGATGTCGATCATCGGTTATGACGATACTCAGGACAGCTCATATTATATCCCGCCGTTAACCACCGTCAGGCAGGATTTTCGTCTACTGGGGAAAACCGCCGTGGACCGGTTGATCAGCCTGA', 'TAGATACCTCACTATGCCTAGCCATAAACCTAAGTAGTCCCATTAACAAAACTACTCGCCAGAGTACTACAAGCAACAGCTTAAAATTCAAAGGACTTGGCGGTGCTTTATACCCACCTAGAGGAGCCTGTTCTATAATCGATAAACCCCGATAAACCTTACCACCCATTGCTATTCCAGTCTATATACCGCCATCTTCAGCAAACCCTTAAAAGGGCAAGAAGTAAGCAAAAGTATATAACATAAAAAAGTTAGGTCAAGGTGTAACTAATTAGGTGGGAAGAAATGGGCTACATTTTCTATCACAGAATACACTACGAATGACAACCTGAAACAAGGTCATCTGAAGGCGGATTTAGTAGTAAATTAAGACTAGAGTGCTTAATTGAACAAGGCCATGAAGCACGCACACACCGCCCGTCACCCTCCTCAAATATCAACTATAATTACACAATACTTAAACACCAAACAAACCAGAGGAGATAAGTCGTAACAAGGTA', 'AGACATACCCCAATCTTCCCCATCCATCCCATCTCCCTCTACCCATTTTTAGCTAATTGCCACTCATCGGCCACTGAGTTACCCAATTTCCCTCTCCCATCCATCAACTCATCCAAAGCTAACATTTTCCCCAAAACAGGCACCCTAACAACAGACATGGCCATGGTGGACGTGAATCCCTTCGGGAAAGGTCTGGCCGCGCTGGACCTGCGAGGCAAGGTGCGAGTGCTGGAATACGACCCCTCAGCGCAGCGTTTCCTGATTGCCTACGCCAATGGCGACGTTGCACTCTTTGAATCTTTCACCTCCCTCACTCCAAATAACGCTATATTGCCTGGCCTATTCCTGACCGAGTCTGCCCTGCCGAATTTTGGCGGTCCCGTCGTGACAGCCGCACACACCGTCTCCCTGTCGCCAGATGGCCGTTACCTCGTCGCCAATGCAACATCAAGCAGCGAGGTTGCTGTCGTGCCCGTGAACGCCACTGCGCCGGGTTCGCT']


## Alignment Analysis
### Let's analyze our generated sequences using Biopython's alignment tools.


```python
def analyze_alignments(generated_seqs: List[str],
                       target_seqs: List[str],
                       names: Optional[List[str]] = None
                      ) -> List[dict]:
    """
    Analyze and visualize alignments between generated and target sequences.
    
    Args:
        generated_seqs: List of generated sequences
        target_seqs: List of target sequences
        names: Optional list of sequence names
        
    Returns:
        List of alignment metrics for each sequence pair
    """
    metrics = []
    print("\nSequence Alignments:")
    
    for i, (gen_seq, target_seq) in enumerate(zip(generated_seqs, target_seqs)):
        if names and i < len(names):
            print(f"\nAlignment {i+1} ({names[i]}):")
        else:
            print(f"\nAlignment {i+1}:")
        
        gen_bio_seq = Seq(gen_seq)
        target_bio_seq = Seq(target_seq)
        
        # Get alignments
        alignments = pairwise2.align.globalms(
            gen_bio_seq, target_bio_seq,
            match=2,
            mismatch=-1,
            open=-0.5,
            extend=-0.1
        )
        
        best_alignment = alignments[0]
        print(format_alignment(*best_alignment))
        
        matches = sum(a == b for a, b in zip(best_alignment[0], best_alignment[1]) 
                      if a != '-' and b != '-')
        alignment_length = len(best_alignment[0].replace('-', ''))
        similarity = (matches / len(target_seq)) * 100
        
        seq_metrics = {
            'similarity': similarity,
            'score': best_alignment[2],
            'length': len(target_seq),
            'gaps': best_alignment[0].count('-') + best_alignment[1].count('-')
        }
        
        if names and i < len(names):
            seq_metrics['name'] = names[i]
            
        metrics.append(seq_metrics)
        
        print(f"Sequence similarity: {similarity:.2f}%")
        print(f"Alignment score: {best_alignment[2]:.2f}")
    
    return metrics

# Analyze alignments
alignment_metrics = analyze_alignments(generated_seqs, target_seqs, names)
```

    
    Sequence Alignments:
    
    Alignment 1 (L1RE2):
    AGGCCTGCCTGCCTCTGTAGGCTCCACCTCTGGGGGCAGGGCACAGACAAACAAAAAG--A-CA-TCACAAACCTCTGCAGACTTAAA-TGTCCCTGTCTGACAGCTTTGAAGAGAGCAGTGGTTCTCCCAGCACGCAGCTGGAGATCTGAGAACGGGCAGACTGCCTCCTCAAGTAGG-TCCCTGG-CCCCTGACCCCCAA-A-CAGCCTAACA--GGACAGTG-ACTC------CAGGG----------AC-TC-CATGCGGC-------TCGC------CC-GCA-CTG-G-----TGAA-C--TTAGAAGTGAAAC-CT----ACCATG--AGG----CT-CACCG----CG----CTG--CAGC-CACC--C-T--AAGGTTAAC-AAAA------AAAAC----AGAGAACTGGCTCCGGCCAGCAAA---AGAACACCGAAAACA--GG--AGCTCAGCGGTAAATCA-GCACCGC-GCGCCTCAGTAAAGCAGCCGGGTTGCTCCTCCACAGGCAGAGCAACAAAGCCG-GCTA-CCTAGCAGTCCTAAATGGCAGATCCA-AATCCCCACACCT--CT--A-GGACA-AA-GCACCAATTCAA-
    ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||  | || |    ||||||||||||||| || ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| || ||||| | ||||||||||||   | |||||||||   ||  || | || |      |||||          || || ||  ||||       || |      || ||| ||| |     ||   |  ||||||| ||||  ||    |||| |  |||    || |||||    |     |||  ||   ||||  | |  |||    || ||||      |||||    | |||  |||    ||  |  |||   ||||||  ||||| |  ||  | |||     ||||  | |||  |  |||||||  |    |          |||||||| |   || | |||   ||   | |  |||  ||  || |    ||| |  || ||    || |     ||  | ||  | || |     |||    
    AGGCCTGCCTGCCTCTGTAGGCTCCACCTCTGGGGGCAGGGCACAGACAAACAAAAAGGCAGCAGT----AACCTCTGCAGACTT-AAGTGTCCCTGTCTGACAGCTTTGAAGAGAGCAGTGGTTCTCCCAGCACGCAGCTGGAGATCTGAGAACGGGCAGACTGCCTCCTCAAGT-GGGTCCCT-GACCCCTGACCCCC--GAGCAGCCTAAC-TGGG--AG-GCAC-CCCCCAGCAGGGGCACACTGACACCTCACA--CGGCAGGGTATTC-CAACAGACCTGCAGCTGAGGGTCCTG--TCTGTTAGAAG-GAAA-ACTAACAACCA-GAAAGGACATCTACACCGAAAAC-CCATCTGTACA--TCACCATCATCAAAG----ACCAAAAGTAGATAAAACCACAA-AGA--TGG----GG--A--AAAAACAGAACA--GAAAA-ACTGGAAA-CTC-----TAAA--ACGCA--G-AGCGCCTC--T----C----------CTCCTCCA-A---AG-G-AAC---GC--AG-T-TCCT--CA--CC-A----GCA-A--CAGAA----CA-A---AGCTGGATGG--AGAATG-----ATT---T
      Score=709.8
    
    Sequence similarity: 77.40%
    Alignment score: 709.80
    
    Alignment 2 (ECOLAC):


    TAATGTTCCGGCGTTATTTCTTGATGTCTCTGACCAGACT-CCCATCAACAGTATTATTTTCTCCCATGAAGACGGTACGCGACTGGGCGTGGAGCATCTGGTCGCATTAGG-TCACCAGCAAATCGCGCTGTTAGCGGGCCCATTAAGTTCTGTCTCGGCGCGTCTA-CGTCTGGCG-GGCTGGCATAAATATCTCA-TACGCAATCAAATTCAGCCGATAGCGGTA-CTGG-AAGGT-GACTGGAGTGCGCA-GTCCGGTTTTGCC--CAGA-CCATGCAAATGCTGAATGAGACGCCGC--CG--CCCACC-GCGC-TGCTGGTG-GCG-AAT-GATGT--GATGGCGG-TGGGCGCAC-TGCGCGCACTGGAACA--AGCCAAA-A-TCAGCG---TCC-CGCAG--GAGATGTCG---ATCATCGGT--T---AT--GACGATACTC--AGGACAGCTCATA-TTATATCCCGCCGTT-AACCACCG-TC--AGGCAGGATTTTCGTC-TA-CTGGGGA-AAACC-GCCGTGGACCGG-TTGA-T-CAG-C-CTGA
    |||||||||||||||||||||||||||||||||||||||  |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| || ||||||||||||||||||||||||||||||||||||||||||||||||||||||  ||||||||  ||||||||||||||||||| | ||||||||||||||||||||||||| | | || ||||  ||||||||||| || ||||||||||  |  || | ||||||||||||||||||||  |  ||  ||  |||| | |||  |||||||  ||  ||  ||  |  |||||| | ||||||||  |||||||       ||  | ||    | ||  ||   |   ||| |  | | || ||   ||| |||||  |   ||  |||||||| |  | ||||||||||  ||||||||||||| | |||||||  ||  |  ||||||||||| | |  ||||||  ||||| | |||||||| | |||  | ||  | ||  
    TAATGTTCCGGCGTTATTTCTTGATGTCTCTGACCAGAC-ACCCATCAACAGTATTATTTTCTCCCATGAAGACGGTACGCGACTGGGCGTGGAGCATCTGGTCGCATT-GGGTCACCAGCAAATCGCGCTGTTAGCGGGCCCATTAAGTTCTGTCTCGGCGCGTCT-GCGTCTGGC-TGGCTGGCATAAATATCTCACT-CGCAATCAAATTCAGCCGATAGCGG-AAC-GGGAAGG-CGACTGGAGTGC-CATGTCCGGTTTT--CAACA-AACCATGCAAATGCTGAATGAG--G--GCATCGTTCCCA-CTGCG-ATGCTGGT-TGC-CAA-CGA--TCAGATGGC-GCTGGGCGCA-ATGCGCGC-------CATTA-CC---GAGTC--CGGGCT--GCGC-GTTG-G-TG-CGGATATC-TCGGTAGTGGGATACGACGATAC-CGAA-GACAGCTCAT-GTTATATCCCGCCG-TCAACCACC-ATCAAA--CAGGATTTTCG-CCT-GCTGGGG-CAAACCAG-CGTGGACC-GCTTG-CTGCA-ACTCT--
      Score=829.8
    
    Sequence similarity: 87.80%
    Alignment score: 829.80
    
    Alignment 3 (NC_007596.2Mammuthusprimigeniusmitochondrion):


    TAGATACCTCACTATGCCTAGCCA-TAAACCT---A-AG-TAGTCCCATTA-ACAAAA-CTACTC-GCCAGAGTA-CTACA-AGCAAC--AGCTTAAAA-TTCAAAGGACTTGGCGGTGCTTTATAC-CCACCTAGAGG-AGCCTGT-TCTA-TAATC-GATA-AACCCCGATAA-ACCTTAC---CACCCATTGCTA-TTCCAGTCT-ATATACCG-CCATCTTCAGCAAACCC-TTAAA-AGGG--CAAGAAGTAAGCAAAAG--TA-T-ATAA--CATA-AAAAAGTTAGGTCA--AGGTGTAA---CTAATTAG-GTG--GG---AAGAAATGGGCTACATTTTCTATC-AC-AGAAT--ACACTA-CGA-ATGACA-AC-CTGAAACAA--GGTCA--TCTGAAGGCGGATTTAGTAGTAAAT-TAAGAC-TAGAGT-GCTTAATTGAACAAGGCCATGAAGCACGC--ACACACCGCCCGTCACC-CTCCTCAA--A--T----ATCAA-C--T-ATAATTACACA-ATACTTAAACACCAA-ACAAACCAGAGGAGAT-AAGTCGTAACAAGGTA
    |||||||||||||||||||||||  |||| ||   | || ||   || ||  || ||| ||| || ||||||| | ||||  |||  |  ||||||||| || |||||||||||||||||||||||  |||||||| || ||||||| ||   ||| | |||  ||||||||| | |||||||   |||   |||||| || |||||  |||||||  ||||||||||||||||| |   | ||||  ||| ||||  |    ||  || | ||||  |||  ||||||||||| |   ||||||     |||      |||  ||   |||  ||||||||||||||||||  |  |||||  |||  | ||  || ||  || ||||   ||  |||    | ||||||||||||||||||||||  |||||  |||||  ||||||||||||||||||||||||  |||  ||||||||||||||| | ||||||||  |  |    ||||| |  | || |||  ||| ||  |||||||  || |||    ||||||||  ||||||||||||||  
    TAGATACCTCACTATGCCTAGCC-CTAAA-CTTTGATAGCTA---CC-TT-TAC-AAAGCTA-TCCGCCAGAG-AACTAC-TAGC--CAGAGCTTAAAACTT-AAAGGACTTGGCGGTGCTTTATA-TCCACCTAG-GGGAGCCTGTCTC--GTAA-CCGAT-GAACCCCGAT-ACACCTTACCGTCAC---TTGCTAATT-CAGTC-CATATACC-ACCATCTTCAGCAAACCCCT---ATAGGGCACAA-AAGT--G----AGCTTAATCATAACCCAT-GAAAAAGTTAGG-C-CGAGGTGT--CGCCTA-----CGTGACGGTCAAAG--ATGGGCTACATTTTCTAT-TA-TAGAATAGACA--AACG-GAT-AC-CACTCTGA---AATGGGT--GGT-TGAAGGCGGATTTAGTAGTAAA-CTAAGA-ATAGAG-AGCTTAATTGAACAAGGCCATGAAG--CGCGTACACACCGCCCGTCA-CTCTCCTCAAGTACCTCCACATCAAACAATCAT-ATT--ACAGAT--TTAAACA--AATACA----AGAGGAGA-CAAGTCGTAACAAGG--
      Score=775.6
    
    Sequence similarity: 83.60%
    Alignment score: 775.60
    
    Alignment 4 (NC_012920.1_homosapiens_mitochondrion):
    AGAC-A---TACCC--CA-A----------T-CTT--CCCCAT-CC----ATCC--C---ATCTCCCT-----CTACC-----CA-TTT-T-TAGCTAAT---TGCCACTC--ATCGGC----CACTGAGTTACCCAAT-TTCCCT-------CTCCCATCCATCAA--CTCATCCAAAGCTAACAT---TT---TCC---CC-----A--A------A--ACAGGCACCCT-A-ACA---ACAGACATG----G--CCATGGTGGACGTGAAT--CCCT-TCGGGAAAGGTCTGG-CCG-CGC-T-----GGA-C---C-T---G--CG-AGGCAAGGTGCGAGTGCTGG--AATACGACCCCTC-AGCGC-AGCGTTTC-CTGATTGCCTACGCCAAT-GGCG--ACGTT-GCACTCTT-TGAATCTTTCA-CCTCCCTCACT--CCAAATAA-CGCTATA--TTGC---CT--GGCCTAT--T--CCTGACC--GAGTCTGCCCTG-CCGAATTTTGGCGGTCCCGTCGTGACAGCCGCACACACCGTCTCCCTG-TCGCCAGATGGC-CG-TTA-CCTC--GTCGCCAAT-GCAA-CATC-----AAGCAGCGAG-GTTGCT--G-T------CGTGCCCG---TGAAC-GCC----ACTGCG-CCG-G-GTTCG-------CT
       | |   |||||  || |          | | |  |||||| ||    | ||  |   | | |||      | |||     || ||| | |||||  |   |  | |||  |   ||    ||||||   |   ||| ||   |       ||  ||  |||| |  | |||  ||| | || ||   ||   |||   ||     |  |      |  | ||  |   | | |||   | || |||     |  |||  || || ||   |  |||| |    |||  ||    ||  ||  |     ||| |   | |   |  || | ||||  ||| |  |||    || |||    ||  ||| | |||    | |  |   ||  | |||   || |  ||    |||      |||    || | |||   |   |  |  ||||| ||  | |  ||     ||  |  ||||  |  ||   ||  | || ||    | |  |||||   | ||   | |    ||||      |||||     | | |   ||     | || ||| || |  ||   |||| | || |  |     ||  |  ||| |||  |  | |      | | |||    | ||  | |    ||| |  ||  | ||| |       ||
    ---CCATCCTACCCAGCACACACACACCGCTGC-TAACCCCATACCCCGAA-CCAACCAAA-C-CCC-AAAGAC-ACCCCCCACAGTTTATGTAGCT--TACCT--C-CTCAAA---GCAATACACTGA---A---AATGTT---TAGACGGGCT--CA--CATC-ACCC-CAT--AAA-C-AA-ATAGGTTTGGTCCTAGCCTTTCTATTAGCTCTTAGTA-AG--A---TTACACATGCA-AG-CAT-CCCCGTTCCA--GT-GA-GT---TCACCCTCT----AAA--TC---ACC-ACG-ATCAAAAGGAACAAGCATCAAGCACGCA-GCAA--TGC-A--GCT--CAAA-ACG----CT-TAGC-CTAGC----CAC--A---CC--C-CCA--CGG-GAAAC---AGCA-----GTGA----TT-AACCT---T---TAGC--AATAAACG--A-AAGTT--TAACTAAG--CTATACTAACC---CCAGG-GT-TG----GTC--AATTT---C-GT---G-C----CAGC------CACCG-----C-GGT---CA-----CACGATTAACC-CAAGT---CAATAG-AAGC--CGGCGTAA--A--GAGTGTT--TTAGATCACCCCC-T-CCC-CAAT-AA-AG-CTAAAACT-C-ACC-TGAGTT-GTAAAAAACT
      Score=514.8
    
    Sequence similarity: 62.40%
    Alignment score: 514.80


## Generate with species prompt


```python
from evo2.utils import make_phylotag_from_gbif

species = 'Phascolarctos cinereus' # Koala bear

species_tag_prompt = make_phylotag_from_gbif(species)

print(f"Species tag prompt: {species_tag_prompt}") # Check if the GBIF API returned a valid species tag!

# Generate species tag
koala_sequence = model.generate(
    [species_tag_prompt],
    n_tokens=500,
    temperature=1.0,
)

print(f"Koala sequence:")
print(koala_sequence.sequences[0])
```

    Species tag prompt: |D__ANIMALIA;P__CHORDATA;C__MAMMALIA;O__DIPROTODONTIA;F__PHASCOLARCTIDAE;G__PHASCOLARCTOS;S__PHASCOLARCTOS CINEREUS|
    Initializing inference params with max_seqlen=616


    Prompt: "|D__ANIMALIA;P__CHORDATA;C__MAMMALIA;O__DIPROTODONTIA;F__PHASCOLARCTIDAE;G__PHASCOLARCTOS;S__PHASCOLARCTOS CINEREUS|",	Output: "GAAGAATACTTGCTTTTATAAGACAATTTGATGTACTAATCATATATTGGATACAGGGCCGTGCACACGAGCATCATACTATGAAGCATCTATGTACACTATCAAAATTGCAGTTGGTTCAAAATAGAGACTATTTTAGTAAAGATATCAATTTTAAGGGTCCTTTTAAAACGGCACTTATCTTTTTCATCCACGAAATGAGAAGGTAAAATGTGAAACTGAAATGAGAAGACAAATATAAATGCATATTAAAAACTCTAGTTTATTATGTAATTCACTTTTGTTACGAAAACTATATTGTTGCTGAGTATTATATTGTTAATTGCATAAGTTTGATGACTTTGCATTTTACTTCAATTCATCAATATCTCCTGTCATACATATTTCAAGAATAAGGTACAACTTGGCGTCTTGCGGGTATATCATTTCTTGTATCTTCTAGAATACATTTTAGTAATTCTTTCTATTGTATTGCTTGTATTCATTTACCCTTCTCGTTC",	Score: -1.296371579170227
    Koala sequence:
    GAAGAATACTTGCTTTTATAAGACAATTTGATGTACTAATCATATATTGGATACAGGGCCGTGCACACGAGCATCATACTATGAAGCATCTATGTACACTATCAAAATTGCAGTTGGTTCAAAATAGAGACTATTTTAGTAAAGATATCAATTTTAAGGGTCCTTTTAAAACGGCACTTATCTTTTTCATCCACGAAATGAGAAGGTAAAATGTGAAACTGAAATGAGAAGACAAATATAAATGCATATTAAAAACTCTAGTTTATTATGTAATTCACTTTTGTTACGAAAACTATATTGTTGCTGAGTATTATATTGTTAATTGCATAAGTTTGATGACTTTGCATTTTACTTCAATTCATCAATATCTCCTGTCATACATATTTCAAGAATAAGGTACAACTTGGCGTCTTGCGGGTATATCATTTCTTGTATCTTCTAGAATACATTTTAGTAATTCTTTCTATTGTATTGCTTGTATTCATTTACCCTTCTCGTTC

