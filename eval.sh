#sleep $((3*60*60))

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

#model="/data/local/user/fanyi/github/LLaMA-Factory/saves/llama-2-7b-hf-seed-back/full/sft/checkpoint-560"
#model="huggingface/Myx"
#model="meta-llama/Llama-2-7b-hf"
#model="outputs/backward/Mf_bsz32_steps500_evalsteps100/checkpoint-500"
# model="outputs/m1/m1_seed_first/checkpoint-7500"
# model="outputs/forward/Mf2_bsz32_steps500_a100/checkpoint-100"
# model="outputs/forward/Mf1_bsz32_steps500_a100/checkpoint-200"
# model="outputs/forward/Mf1_bsz32_steps500_a100/checkpoint-100"
# model="outputs/m1/m1_seed_unlabelled_400000/checkpoint-19500"
# model="outputs/m1/m1_seed_unlabelled_400000_github/checkpoint-6500"
# model="outputs/m1/m1_seed_unlabelled_400000_github/checkpoint-13000"
# model="outputs/m1/m1_seed_unlabelled_400000/checkpoint-13000"
# model="outputs/m1/m1_seed_unlabelled_400000/checkpoint-19500"

#python src/core/llm_eval.py --model outputs_v100/forward/Mf/checkpoint-200 --tasks openllm   
# 一个一个模型评估 否则很容易混淆
model=
export CUDA_VISIBLE_DEVICES=0                
lm_eval --model hf \
    --model_args pretrained=${model} \
    --tasks openllm \
    --device cuda \

lm_eval --model hf \
    --model_args pretrained=${model} \
    --tasks ifeval,squadv2 \
    --device cuda \
    --output_path eval_result
    



#python
