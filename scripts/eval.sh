model="outputs/backward/Mr"
#model="huggingface/M0"
#model="meta-llama/Llama-2-7b-hf"
#model="/data/local/user/fanyi/models/Llama-2-7b-hf"
export CUDA_VISIBLE_DEVICES=1
python -m ipdb src/core/llm_eval.py \
	--model ${model} \
	--tasks arc_challenge,winogrande 
	
