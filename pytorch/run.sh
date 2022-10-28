for dataset in atis conll2003 i2b2 mitmovie multiwoz onto restaurant snips wikigold WNUT17; do
  for f in {0..9}; do
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false python model.py \
      --gpu 1 \
      --lr 2e-5 \
      --pretrained_lm bert-base-uncased \
      --dataset_name ${dataset} \
      --train_path_suffix few_shot_5_${f} \
      --val_path_suffix few_shot_5_${f} \
      --test_path_suffix test \
      --negative_multiplier 3.0 \
      --max_epochs 30
  done
done