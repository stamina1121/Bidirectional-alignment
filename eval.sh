#sleep $((3*60*60))

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export CUDA_VISIBLE_DEVICES=9
#model="/data/local/user/fanyi/github/LLaMA-Factory/saves/llama-2-7b-hf-seed-back/full/sft/checkpoint-560"
#model="huggingface/Myx"
#model="meta-llama/Llama-2-7b-hf"
#model="outputs/backward/Mf_bsz32_steps500_evalsteps100/checkpoint-500"
model="outputs/m1/m1_seed_first/checkpoint-7500"
model="outputs/forward/Mf2_bsz32_steps500_a100/checkpoint-100"
model="outputs/forward/Mf1_bsz32_steps500_a100/checkpoint-200"
model="outputs/forward/Mf1_bsz32_steps500_a100/checkpoint-100"
model="outputs/m1/m1_seed_unlabelled_400000/checkpoint-19500"
model="outputs/m1/m1_seed_unlabelled_400000_github/checkpoint-6500"
model="outputs/m1/m1_seed_unlabelled_400000_github/checkpoint-13000"
model="outputs/m1/m1_seed_unlabelled_400000/checkpoint-13000"
model="outputs/m1/m1_seed_unlabelled_400000/checkpoint-19500"

lm_eval --model hf \
    --model_args pretrained=outputs2/m1_github/m1_github_seed_unlabelled_20000/checkpoint-1200 \
    --tasks squadv2 \
    --device cuda


#lm_eval --model hf \
#    --model_args pretrained=outputs/forward/Mf3/checkpoint-100 \
#    --tasks arc_challenge \
#    --num_fewshot 25 \
#    --device cuda:15 \
    
model="outputs2/m1/m1_seed_unlabelled_20000/checkpoint-3000"


#python src/core/llm_eval.py \
#	--model ${model} \
#	--num_fewshot 0 \
#	--tasks truthfulqa


#python src/core/llm_eval.py \
#	--model ${model} \
#	--tasks openllm
echo $model
