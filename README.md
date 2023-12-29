# ProtFound-PD

The official code repository of "Hunting for peptide binders of specific targets with data-centric generative language models".

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Instructions for Use](#instructions-for-use)
- - [Data Preparation](#data-preparation)
- - [Model Pretraining](#model-pretraining)
- - [Finetuning on Your Data](#finetuning-on-your-data)
- - [Peptide Generation](#peptide-generation)
-  - [Visualization of the reduced distribution space](#visualization-of-the-reduced-distribution-space)
- [Pretrained and Finetuned Models Availability](#pretrained-and-finetuned-models-availability)
- [Acknowledgments](#acknowledgments)
- [License](#license)

# Overview

The increasing frequency of emerging viral infections calls for more    efficient and low-cost drug design methods. Peptide binders have  emerged as a strong contender to curb the pandemic due to their    efficacy, safety, and specificity. Here, we propose a customizable    low-cost pipeline incorporating model auditing strategy and    data-centric methodology for controllable peptide generation. A    generative protein language model, pretrained on approximately 140    million protein sequences, is directionally fine-tuned to generate    peptides with desired properties and binding specificity. The    subsequent multi-level structure screening reduces the synthetic    distribution space of peptide candidates regularly to identify    authentic high-quality samples, i.e. potential peptide binders, at    *in silico* stage. Paired with molecular dynamics simulations, the number of candidates that need to be verified in wet-lab experiments    is quickly reduced from more than 2.2 million to 16. These potential    binders are characterized by enhanced yeast display to determine    expression levels and binding affinity to the target. The results    show that only a dozen candidates need to be characterized to obtain    the peptide binder with ideal binding strength and binding    specificity. Overall, this work achieves efficient and low-cost    peptide design based on a generative language model, increasing the    speed of *de novo* protein design to an unprecedented level. The    proposed pipeline is customizable, that is, suitable for rapid design    of multiple protein families with only minor modifications.

![Our pipeline](/imgs/2023-12-29/2857mNZwVFdXrpre.png)
# Requirements

[mindspore](https://www.mindspore.cn/en) >=1.5.

**!! NOTE: Pretraining and fine-tuning of our generative protein language model, and peptide sequence generation are conducted on **Ascend-910 (32GB)** with MindSpore. If you use a GPU, please follow the instructions on [MindSpore's official website](https://www.mindspore.cn/en) to install the GPU-adapted version.**



#  Instructions for Use





## Data Preparation

Dataset for pretraining can be obtained from [BFD](https://bfd.mmseqs.com/), [Pfam](http://pfam.xfam.org/) and [UniProt](https://www.uniprot.org/). In this work, the miniprotein scaffold libraries used for the first round fine-tuning can be obtained from [previous work](https://www.nature.com/articles/s41586-022-04654-9). In addition, the peptide sequences carrying targeting information used for the second round fine-tuning can be obtained through the related suites of [Rosetta](https://www.rosettacommons.org/).

After data filtering and processing is completed, convert it to a txt format in which a line means a sequence.

Structure your data directory as follows:

```plaintext
your_data_dir/
  ├─01.txt
  ...
  └─09.txt
```

Convert the dataset to MindRecord format using:

```shell
python prepare_data.py --data_url your_data_dir --save_dir your_save_dir
```

## Model Pretraining

Initiate pretraining with:

```shell
python train.py --train_url your_data_dir
```

**Note:** This code is tested on Pengcheng CloudBrain II (Ascend-910). Modifications may be needed for other environments.



## Finetuning on Your Data

To finetune the pretrained model on your dataset:

```shell
python train.py --train_url your_data_dir --load_checkpoint_path pretrained_ckpt_path --finetune
```

## Peptide Generation

Generate peptides by specifying your template and the checkpoint path:

```shell
python generate_peptide.py --head_txt your_template_peptide --load_checkpoint_path pretrained_ckpt_path
```


## Visualization of the reduced distribution space

To see whether structure-based calculations shrink the synthetic distribution space in the desired direction, we extract protein descriptors of candidates in different screening stages and use factor analysis for dimensionality reduction visualization. Combining multiple protein descriptors allows us to gain a more comprehensive and detailed understanding of the underlying patterns of protein sequences. Here, we employ four length-independent descriptors ($k$-mer spectra, Normalized Moreau-Broto Autocorrelation descriptor, Quasi-sequence-order descriptor, Composition, Transition, and Distribution descriptor) to ensure that sequences of different lengths have equal-length descriptor vectors.  All the descriptors are finally concatenated together for analysis.

  - get_descriptor.py                # Script to extract descriptors from protein sequences
  -  vis.py                           # Script for visualizing descriptors
  -  example.sh                       # Example shell script to demonstrate descriptor extraction and visualization


# Pretrained and Finetuned Models Availability

Pretrained model can be download [here](https://figshare.com/ndownloader/files/43847313).
Finetuned model can be download [here](https://figshare.com/ndownloader/files/43847910).

# Acknowledgments

This repository builds upon the PanGu-Alpha codebase. For comprehensive details, [see here](https://openi.pcl.ac.cn/PCL-Platform.Intelligence/PanGu-Alpha).

# License
This project is covered under the **Apache 2.0 License**.
