set -e
# Configuration
N=2
max_N=4
frequency=50
num_nodes=1
num_gpu_per_node=8
SINGLE_GPU=0
MULTI_GPU="0,1,2,3,4,5,6,7"
#MULTI_GPU=8,9,10,11,12,13,14,15
model_name_or_path="meta-llama/Llama-2-7b-hf"
base_output_dir="outputs_a100"
data_path="data/seed/seed.jsonl"

# Utility function to execute training with torchrun
run_torchrun() {
    local model_path=$1
    local data_path=$2
    local output_dir=$3
    local batch_size=$4
    local max_steps=$5
    local extra_args=$6

    mkdir -p $output_dir

    torchrun \
        --nnodes ${num_nodes} \
        --nproc_per_node ${num_gpu_per_node} \
        --master_port 29533 \
        -m src.core.train \
        --deepspeed conf/ds_zero1default.json \
        --model_name_or_path ${model_path} \
        --data_path ${data_path} \
        --per_device_train_batch_size ${batch_size} \
        --per_device_eval_batch_size ${batch_size} \
        --adam_beta1 0.9 \
        --adam_beta2 0.95 \
        --learning_rate "1e-5" \
        --weight_decay 0.1 \
        --max_grad_norm 1.0 \
        --logging_strategy steps \
        --logging_steps 1 \
        --output_dir ${output_dir} \
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
        --max_steps ${max_steps} \
        --save_strategy steps \
        --save_steps ${frequency} \
        --evaluation_strategy 'no' \
        $extra_args
}

# Utility function to run prediction with python script
run_predict() {
    local model_path=$1
    local data_filepath=$2
    local save_filepath=$3
    local prompt_column_name=$4
    local temperature=$5
    local top_p=$6
    local extra_args=$7

    python -m src.core.predict_vllm \
        --model_path=${model_path} \
        --data_filepath=${data_filepath} \
        --save_filepath=${save_filepath} \
        --prompt_column_name=${prompt_column_name} \
        --dtype float16 \
        --temperature ${temperature} \
        --top_p ${top_p} \
        --tensor_parallel_size=1 \
        $extra_args
}
##


# 俩初始阶段依赖frequency  其余三个分别由previous,current,next组成
# 0. Finetune Mr0
if [ $N -eq 1 ]; then
    output_dir="${base_output_dir}/backward/Mr0"
    run_torchrun "${model_name_or_path}" "${data_path}" "$output_dir" 4 $frequency "--reverse"
fi

while [ $N -le $max_N ]; do
    echo "Running iteration $N"
    previous_max_steps=$(((N - 1) * frequency)) 
    current_max_steps=$((N * frequency)) 
    next_max_steps=$(((N + 1) * frequency)) 

    # 1. Predict with Mr
    model_path="${base_output_dir}/backward/Mr$((N - 1))/checkpoint-${current_max_steps}"
    save_path="${base_output_dir}/predict_iteration${N}/R0_I.jsonl"
    #if [ $N -gt 2 ]; then
    run_predict "$model_path" "$data_path" "$save_path" "response" 0.7 0.9 "--reverse"
    #fi

    # 2. Finetune Mf
    export CUDA_VISIBLE_DEVICES=${MULTI_GPU}
    data_path="${base_output_dir}/predict_iteration${N}/R0_I.jsonl"
    output_dir="${base_output_dir}/forward/Mf${N}"
    model_path="${base_output_dir}/forward/Mf$((N - 1))/checkpoint-${previous_max_steps}"
    if [ $N -eq 1 ]; then
        run_torchrun "${model_name_or_path}" "$data_path" "$output_dir" 4 $frequency
    elif [ $N -gt 1 ]; then
        run_torchrun "$model_path" "$data_path" "$output_dir" 4 $current_max_steps
    fi

    # 3. Predict with Mf
    model_path="${base_output_dir}/forward/Mf${N}/checkpoint-${current_max_steps}"
    save_path="${base_output_dir}/predict_iteration${N}/I0_R.jsonl"
    run_predict "$model_path" "$data_path" "$save_path" "instruction" 0.7 0.9

    # 4. Finetune Mr
    export CUDA_VISIBLE_DEVICES=${MULTI_GPU}
    data_path="${base_output_dir}/predict_iteration${N}/I0_R.jsonl"
    output_dir="${base_output_dir}/backward/Mr${N}"
    model_path="${base_output_dir}/backward/Mr$((N-1))/checkpoint-${current_max_steps}"
    run_torchrun "$model_path" "$data_path" "$output_dir" 4 $next_max_steps "--reverse"

    N=$((N + 1))
done

# 5. Unlabelled prediction
export CUDA_VISIBLE_DEVICES=${MULTI_GPU}
model_path="${base_output_dir}/backward/Mr${max_N}/checkpoint-${next_max_steps}"
data_path2="data/unlabelled/falcon-refinedweb-sampled.jsonl"
save_path="${base_output_dir}/predict_iteration${max_N}/unlabelled.jsonl"
run_predict "$model_path" "$data_path2" "$save_path" "content" 0.7 0.9 "--reverse"

# Merge datasets
python merge.py --file1 "${data_path}" --file2 "${base_output_dir}/predict_iteration${max_N}/unlabelled.jsonl" --output "${base_output_dir}/predict_iteration${max_N}/seed_unlabelled.jsonl"

# 6. Finetune final model
export CUDA_VISIBLE_DEVICES=${MULTI_GPU}
data_path="${base_output_dir}/predict_iteration${max_N}/seed_unlabelled.jsonl"
output_dir="${base_output_dir}/m${max_N}/m${max_N}_seed_unlabelled"
#run_torchrun "$model_name_or_path" "$data_path" "$output_dir" 12 400 #放在下面了 因为batch_size改变

mkdir -p $output_dir

torchrun \
        --nnodes 1 \
        --nproc_per_node 8 \
        --master_port 29502 \
        -m src.core.train \
        --deepspeed conf/ds_zero1default.json \
        --model_name_or_path ${model_name_or_path} \
        --data_path ${data_path} \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --adam_beta1 0.9 \
        --adam_beta2 0.95 \
        --learning_rate "2e-5" \
        --weight_decay 0.1 \
        --max_grad_norm 1.0 \
        --logging_strategy steps \
        --logging_steps 1 \
        --output_dir ${output_dir} \
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
        --max_steps 600 \
        --save_strategy steps \
        --save_steps 600 \
        --warmup_ratio 0.05 \
        --evaluation_strategy 'no'


conda activate lm_eval
lm_eval --model hf \
    --model_args pretrained=${output_dir}/checkpoint-600 \
    --tasks squadv2,ifeval \
    --device cuda \
    --output_path eval_result
lm_eval --model hf     --model_args pretrained=outputs_a100/m4/m4_seed_unlabelled/checkpoint-600  --tasks openllm  --device cuda --output_path eval_result