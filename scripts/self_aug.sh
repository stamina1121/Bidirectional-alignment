#!/usr/bin/bash

export CUDA_VISIBLE_DEVICES="3"
#export CUDA_VISIBLE_DEVICES="8,9,10,11,12,13,14,15"


#model_path="Spico/Humback-M0"
#model_path="Spico/Humback-Myx"
#model_path="huggingface/Myx"
#model_path="meta-llama/Llama-2-7b-hf"
model_path="outputs/backward/Mr_bsz32_steps500_evalsteps100/checkpoint-200"
#model_path="outputs/backward/Mr/checkpoint-300"
data_filepath="data/unlabelled/falcon-refinedweb-sampled.jsonl"
save_filepath="outputs/m1/unlabelled_gen_instruction.jsonl"


prompt_column_name="content"

#!/bin/bash

# 设置模型路径和其他通用参数
select=reverse # 从命令行传入参数 "reverse" 或 "forward"

# 检查 select 参数并执行对应的命令
if [ "$select" = "reverse" ]; then
    model_path="outputs/backward/Mr2_bsz32_steps500_a100/checkpoint-100"
    model_path="outputs/backward/Mr1_bsz32_steps500_a100/checkpoint-200"
    echo "Running reverse model..."

    python -m ipdb src/core/predict_vllm.py \
        --model_path=${model_path} \
        --data_filepath="data/seed/seed.jsonl" \
        --save_filepath="outputs2/predict_iteration1/R0_I.jsonl" \
        --prompt_column_name="response" \
        --dtype float16 \
        --temperature 0.7 \
        --top_p 0.9 \
        --reverse
        #--tensor_parallel_size=8 \
elif [ "$select" = "forward" ]; then
    model_path="outputs/forward/Mf_bsz32_steps500_clone/checkpoint-250"
    echo "Running forward model..."

   # python -m ipdb src/core/predict_vllm2.py \
fi   
    
