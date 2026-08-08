#!/bin/bash
DEVICE=$1
DATASET=$2
EPOCH=$3
DATA_RATE=$4
MASKING=$5
ALPHA=$6

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py \
            --gpu           $DEVICE \
            --eval_epoch    20 \
            --only_test     0 \
            --data_choice   $DATASET \
            --data_split    "norm" \
            --data_rate     $DATA_RATE \
            --epoch         $EPOCH \
            --epoch_per_CYCLES 50 \
            --CYCLES        5 \
            --lr            0.001  \
            --scheduler     "fixed"\
            --optim         "adam"\
            --rho           0.1\
            --alpha_        $ALPHA\
            --strategy      "frobenius"\
            --tau3          0.01\
            --mask          $MASKING\
            --early_stop_threshold 1e-7\
            --batch_size    3500 \
            --csls          \
            --csls_k        3 \
            --random_seed   42 \
            --exp_id        "seed_42" \
            --workers       12 \
            --dist          0 \
            --accumulation_steps 1 \
            --ratio         1.0     \
            --num_layers   3
