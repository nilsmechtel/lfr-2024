# LFR-2024: Unsupervised Dataset Distillation

This repository contains the implementation of unsupervised dataset distillation for the "Learning Feature Representations" course project. The project explores the potential of distilling large datasets into compact synthetic datasets optimized for downstream tasks.

## Environment Setup

To set up the environment, use the following commands:

```bash
conda env create -f environment.yml
conda activate lfr
```

## Running Dataset Distillation

To view all available arguments and options, use the help flag:

```bash
python dataset_distillation.py --help
```

To run dataset distillation with the default configuration:

```bash
python dataset_distillation.py
```
