#!/usr/bin/bash

num_nodes=1
num_gpu_per_node=8

bsz=32
max_steps=500
data_path="data/seed/seed.jsonl"
output_dir="outputs/forward/Mf"
# max_steps=765  # 21,301 curated instances (score=5) + 3,200 seed data for M1 training
# data_path="data/curated/m1.jsonl"
# output_dir="/dev/shm/tzhu/Humback/models/m1_with_diff_sys_prompt"
#max_steps=2400
#data_path=data/curated/m1_v2.jsonl
#output_dir="/dev/shm/tzhu/Humback/models/m1_strict_score_matching_2400steps"

mkdir -p $output_dir
bsz_per_dev=$(echo "${bsz} / ${num_nodes} / ${num_gpu_per_node}" | bc)

torchrun \
    --nnodes ${num_nodes} \
    --nproc_per_node ${num_gpu_per_node} \
    -m src.core.train \
        --deepspeed conf/ds_zero2default.json \
        --model_name_or_path meta-llama/Llama-2-7b-hf \
        --data_path ${data_path} \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --adam_beta1 0.9 \
        --adam_beta2 0.95 \
        --learning_rate "1e-5" \
        --final_lr "9e-6" \
        --weight_decay 0.1 \
        --max_grad_norm 1.0 \
        --evaluation_strategy "no" \
        --logging_strategy steps \
        --logging_steps 1 \
        --output_dir ${output_dir} \
        --overwrite_output_dir \
        --ddp_timeout 30000 \
        --logging_first_step True \
        --bf16 False \
	--fp16 False \
        --tf32 False \
        --ddp_find_unused_parameters False \
        --gradient_checkpointing \
        --report_to none \
        --log_level info \
        --lazy_preprocess True \
        --max_steps ${max_steps} \
        --save_strategy steps \
        --save_steps 500 \
#export CUDA_VISIBLE_DEVICES=0
#python -m ipdb src/core/train_flash_attn.py \
