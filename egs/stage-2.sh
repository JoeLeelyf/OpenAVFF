#!/bin/bash

contrast_loss_weight=0.05
mae_loss_weight=1.0
norm_pix_loss=True

pretrain_path=../../path/to/stage-1.pth

lr=1e-4
epoch=25
lrscheduler_start=10
lrscheduler_decay=0.5
lrscheduler_step=5
dataset_mean=-5.081
dataset_std=4.4849
target_length=1024
noise=True
batch_size=48
lr_adapt=False

n_print_steps=100

tr_data=../../data/trainset_real.csv
te_data=../../data/valset_real.csv

save_dir=./exp/stage-2
mkdir -p $save_dir
mkdir -p ${save_dir}/models

CUDA_CACHE_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -W ignore ../src/run_pretrain.py \
--data-train ${tr_data} --data-val ${te_data} --save-dir $save_dir \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size \
--lrscheduler_start ${lrscheduler_start} --lrscheduler_decay ${lrscheduler_decay} --lrscheduler_step ${lrscheduler_step} \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} --noise ${noise} \
--lr_adapt ${lr_adapt} \
--norm_pix_loss ${norm_pix_loss} \
--pretrain_path ${pretrain_path} \
--mae_loss_weight ${mae_loss_weight} --contrast_loss_weight ${contrast_loss_weight}