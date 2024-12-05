export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
data_path="data/seed/seed.jsonl"
output_dir="outputs_new"
num_nodes=1
num_gpu_per_node=8
torchrun \
    --nnodes ${num_nodes} \
    --nproc_per_node ${num_gpu_per_node} \
    --master_port 29505 \
    -m src.core.train_infer \
        --deepspeed conf/ds_zero1default.json \
        --model_name_or_path meta-llama/Llama-2-7b-hf \
        --data_path ${data_path} \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --adam_beta1 0.9 \
        --adam_beta2 0.95 \
        --learning_rate "1e-5" \
        --weight_decay 0.1 \
        --max_grad_norm 1.0 \
        --logging_strategy steps \
        --logging_steps 1 \
        --overwrite_output_dir \
        --ddp_timeout 30000 \
        --logging_first_step True \
        --bf16 True \
        --tf32 False \
        --ddp_find_unused_parameters False \
        --gradient_checkpointing \
        --report_to none \
        --log_level info \
        --lazy_preprocess True \
        --max_steps 50 \
        --evaluation_strategy 'no' \
        --save_strategy="no" \
        --output_dir ${output_dir} \
        #--save_strategy steps \
        #--save_steps 50 \
        python -m ipdb src/core/train_infer.py \