# Formulating Few-shot Fine-tuning Towards Language Model Pre-training: A Pilot Study on Named Entity Recognition

### What is this?
This is the repo for the paper `Formulating Few-shot Fine-tuning Towards Language Model Pre-training: A Pilot Study on Named Entity Recognition`.
We designed a new task for NER few-shot fine-tuning. Here are the performances of our method (FFF-NER) when compared with
Sequence Labeling (LC) and Prototype Method (P) on 10 datasets for 5-shot Named Entity Recognition. The performances below are 
averaged over 10 splits, and the variance is expected to be pretty high for most of the datasets. For the dataset and splits,
refer to the Dataset section below.

| Dataset        | LC   | P    | FFF-NER (Paper) | FFF-NER (rerun w/ pytorch/) |
|----------------|------|------|-----------------|-----------------------------|
| CoNLL          | 53.5 | 58.4 | 67.90 (3.95)    | 67.46 (3.70)                |
| Onto           | 57.7 | 53.3 | 66.40 (1.62)    | 65.65 (2.19)                |
| WikiGold       | 47.0 | 51.1 | 61.42 (8.82)    | 64.05 (4.36)                |
| WNUT17         | 25.7 | 29.5 | 30.48 (3.69)    | 30.12 (3.54)                |
| MIT Movie      | 51.3 | 38.0 | 60.32 (1.12)    | 60.20 (1.59)                |
| MIT Restaurant | 48.7 | 44.1 | 51.99 (2.33)    | 52.03 (3.48)                |
| SNIPS          | 79.2 | 75.0 | 83.43 (1.44)    | 83.81 (0.84)                |
| ATIS           | 90.8 | 84.2 | 92.75 (0.99)    | 93.22 (0.90)                |
| Multiwoz       | 12.3 | 21.9 | 47.56 (8.46)    | 50.12 (5.36)                |
| I2B2           | 36.0 | 32.0 | 49.37 (5.08)    | 48.12 (7.99)                |
| Average        | 50.2 | 48.8 | 61.15           | 61.48                       |


### Code
Please check `pytorch/` for the pytorch version of the code.  The tensorflow version in `tensorflow/` is under construction.

### Dataset
`dataset/` contains some sample data for verifications. 
Check https://few-shot-ner-benchmark.github.io/ for the full datasets.