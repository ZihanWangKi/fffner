# Formulating Few-shot Fine-tuning Towards Language Model Pre-training: A Pilot Study on Named Entity Recognition

### What is this?
This is the repo for the paper `Formulating Few-shot Fine-tuning Towards Language Model Pre-training: A Pilot Study on Named Entity Recognition`.
We designed a new task for NER few-shot fine-tuning. Here are the performances of our method (FFF-NER) when compared with
Sequence Labeling (LC) and Prototype Method (P) on 10 datasets for 5-shot Named Entity Recognition.

| Dataset        | LC   | P    | FFF-NER |
|----------------|------|------|---------|
| CoNLL          | 53.5 | 58.4 | 67.90   |
| Onto           | 57.7 | 53.3 | 66.40   |
| WikiGold       | 47.0 | 51.1 | 61.42   |
| WNUT17         | 25.7 | 29.5 | 30.48   |
| MIT Movie      | 51.3 | 38.0 | 60.32   |
| MIT Restaurant | 48.7 | 44.1 | 51.99   |
| SNIPS          | 79.2 | 75.0 | 83.43   |
| ATIS           | 90.8 | 84.2 | 92.75   |
| Multiwoz       | 12.3 | 21.9 | 47.56   |
| I2B2           | 36.0 | 32.0 | 49.37   |
| Average        | 50.2 | 48.8 | 61.15   |


### Code
Please check `pytorch/` for the pytorch version of the code.  The tensorflow version in `tensorflow/` is under construction.

### Dataset
`dataset/` contains some sample data for verifications. 
Check https://few-shot-ner-benchmark.github.io/ for the full datasets.