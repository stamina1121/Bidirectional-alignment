import math
import transformers
from src.core.trainer import ScheduledTrainer
from src.data import make_data_module,make_supervised_data_module_train_eval
from src.utils.config import DataArguments, ModelArguments, TrainingArguments
from src.utils.io import safe_save_model_for_hf_trainer
import argparse
from src.data import InferenceDataset
from src.utils.io import dump_jsonlines, load_jsonlines
from tqdm import tqdm
from src.data import InferenceDataset
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from vllm import LLM, SamplingParams
from accelerate import Accelerator
import deepspeed

def infer(model, tokenizer, input_data, prompt_column_name, N):
    
    print(dir(model))
  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if prompt_column_name == 'instruction':
        reverse = False
    elif prompt_column_name == 'response':
        reverse = True
    elif prompt_column_name == 'content':
        reverse = True

    # llm = LLM(
    #     model,
    #     tensor_parallel_size=1,
    #     gpu_memory_utilization=0.8,  # 降低显存占用比例
    #     dtype='float16',
        
    # )

    # sampling_params = SamplingParams(
    #     temperature=0.7,
    #     top_p=0.9,
    #     max_tokens=1024,
    # )
    generation_kwargs = {
        "temperature": 0.7,
        "top_p": 0.9,
        "max_new_tokens": 1024,
        "do_sample": True
    }
    data = InferenceDataset(
        input_data,
        content_name=prompt_column_name,
        reverse=reverse,
    )
    prompts = data.get_all()  
    # results = llm.generate(prompts[:16800], use_tqdm=True, sampling_params=sampling_params)
    results = []
   
    # accelerator = Accelerator(deepspeed_config="conf/ds1/ds_zero1default.json")
    # model = accelerator.prepare(model)

    if reverse and N % 2 == 0:
        prompts = prompts[1600:]
    elif reverse and N % 2 == 1:
        prompts = prompts[:1600]
    else:
        prompts = prompts

    for prompt in tqdm(prompts, desc="Generating responses"):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_kwargs)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True, use_cache=True)
        results.append(response[len(prompt):].strip())
    # batch_size=8
    # for i in tqdm(range(0, len(prompts), batch_size), desc="Generating responses"):
    # # 获取当前批次的 prompt 列表
    #     batch_prompts = prompts[i:i + batch_size]
        
    #     # 将批次中的所有提示传递给 tokenizer，返回张量，并将它们移到目标设备
    #     inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        
    #     # 使用 model.generate 进行推理，关闭梯度计算
    #     with torch.no_grad():
    #         outputs = model.generate(**inputs, **generation_kwargs)
        
    #     # 解码输出，并去除特殊符号
    #     batch_responses = tokenizer.batch_decode(outputs, skip_special_tokens=True, use_cache=True)
        
    #     # 对每个输出去掉原始提示部分，确保格式不变
    #     for prompt, response in zip(batch_prompts, batch_responses):
    #         results.append(response[len(prompt):].strip())

    dump_jsonl = []
    
    if reverse:
         for raw, result in zip(input_data, results):
            dump_jsonl.append(
                {
                "instruction": result,
                "response": raw[prompt_column_name], ###实际上response是results，只是为了方便后续的微调。
                }
            )
    else:
        for raw, result in zip(input_data, results):
            dump_jsonl.append(
                {
                "instruction": raw[prompt_column_name],
                "response": result,
                }
            )
    return dump_jsonl
            
def train(model, tokenizer, data_module, training_args, optimizer_state, frequency):
    optimizer = AdamW(model.parameters(), lr=1e-5)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        # 初始化学习率
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=training_args.max_steps, last_epoch=training_args.max_steps-frequency) 
        #后续trainer的训练步数由这个决定
        training_args.max_stesp = frequency
    else:
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=training_args.max_steps)

    
    # if optimizer_state is not None:
    #     # 假设 state_dict 中有优化器步数，通常为 'step'
    #     # last_optimizer_step = optimizer_state.get('state', {}).get(0, {}).get('step', 0)
    #     # if last_optimizer_step > 0:
    #     if True:
    #         for _ in range(10):
    # lr_scheduler.step()
    trainer = ScheduledTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        optimizers=(optimizer, lr_scheduler),
        **data_module,
    )
    trainer.train()

    optimizer_state = optimizer.state_dict()

    model.config.use_cache = True
    return model, optimizer_state

def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 配置模型
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False

    # 加载模型和 tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    # 加载数据
    if training_args.evaluation_strategy != "no":
        print("Loading training and evaluation data...")
        data_module = make_supervised_data_module_train_eval(tokenizer=tokenizer, data_args=data_args)
    else:
        print("Loading training data...")
        data_module = make_data_module(tokenizer=tokenizer, data_args=data_args)

    raw_data = load_jsonlines(data_args.data_path)

    # 初始化训练循环参数
    N = 1
    max_N = 4
    Mf = None  # 初始化 Mf
    frequency = training_args.max_steps

    # Step 1: 初始微调 Mr
    print("=== Step 1: Initial fine-tuning of Mr ===")
    Mr, Mr_optimizer= train(model, tokenizer, data_module, training_args, None, frequency)

    while N <= max_N:
        print(f"=== Iteration {N} ===")
        print(f"=== Save_steps {frequency} ===")
        
        current_max_steps=(N * frequency)
        next_max_steps=((N + 1) * frequency)
        

        # Step 2: 使用 Mr 推理生成数据  reverse!!!
        Mr_data = infer(Mr, tokenizer, raw_data, "response", N)
        
        # Step 3: 构造微调数据集并微调 Mf
        training_args.max_steps = current_max_steps
        print(training_args.max_steps)
        data_module_finetuned = make_data_module(tokenizer=tokenizer, data_args=data_args, custom_data=Mr_data)

        if N == 1:
            Mf, Mf_optimizer= train(model, tokenizer, data_module_finetuned, training_args, None, frequency)
        else:
            Mf, Mf_optimizer= train(Mf, tokenizer, data_module_finetuned, training_args, Mf_optimizer, frequency)

        # Step 4: 使用 Mf 推理生成数据
        Mf_data = infer(Mf, tokenizer, raw_data, 'instruction', N)

        # Step 5: 构造微调数据集并微调 Mr
        training_args.max_steps = next_max_steps
        print(training_args.max_steps)
        data_module_finetuned = make_data_module(tokenizer=tokenizer, data_args=data_args, custom_data=Mf_data)
        Mr, Mr_optimizer = train(Mr, tokenizer, data_module_finetuned, training_args, Mr_optimizer,frequency)

        # 增加迭代计数
        N += 1

    print("Training and inference loop completed.")
##save
if __name__ == "__main__":
    main()
    
        






