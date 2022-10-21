# FFF-NER python ver.

### Setup
Install pytorch, transformers, pytorch_lightning. 
The code was written for an older version of these three packages, but I tested with the newest version 
pytorch = 1.12, transformers = 4.23.1, pytorch_lightning = 1.7.7 and the code is functioning.

### Dataset
We provide a minimum reproduction dataset in `../dataset/`, containing one split of two datasets we used, conll2003 and restaurant and their corresponding test sets.
The formats should be easy to understand and to add more datasets for other purposes.

### Running
```bash
TOKENIZERS_PARALLELISM=false python model.py \
  --gpu 1 \
  --lr 2e-5 \
  --pretrained_lm bert-base-uncased \
  --dataset_name conll2003 \
  --train_path_suffix few_shot_5_0 \
  --val_path_suffix few_shot_5_0 \
  --test_path_suffix test \
  --negative_multiplier 3.0 \
  --max_epochs 30
```
