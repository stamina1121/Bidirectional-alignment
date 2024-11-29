import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from src.data import InferenceDataset
from src.utils.io import dump_jsonlines, load_jsonlines
import torch

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=args.dtype).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    generation_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": True
    }

    raw_data = load_jsonlines(args.data_filepath)
    data = InferenceDataset(
        raw_data,
        content_name=args.prompt_column_name,
        reverse=args.reverse,
    )
    prompts = data.get_all()  #len(prompts = 50200)

    # 00:25 / 100 prompts on one GPU
    results = []
    #x = 500
    for prompt in tqdm(prompts, desc="Generating responses"):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, **generation_kwargs)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append(response[len(prompt):].strip())

    # 07:24 / 100 prompts on one GPU
    # results = []
    # for prompt in tqdm(prompts):
    #     result = llm.generate(prompt, use_tqdm=False, sampling_params=sampling_params)
    #     results.append(result)

    dump_jsonl = []
    
    if args.reverse:
         for raw, result in zip(raw_data, results):
            dump_jsonl.append(
                {
                "instruction": result,
                "response": raw[args.prompt_column_name], ###实际上response是results，只是为了方便后续的微调。
                }
            )
    else:
        for raw, result in zip(raw_data, results):
            dump_jsonl.append(
                {
                "instruction": raw[args.prompt_column_name],
                "response": result,
                }
            )
    dump_jsonlines(dump_jsonl, args.save_filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--data_filepath", type=str)
    parser.add_argument("--save_filepath", type=str)
    parser.add_argument("--prompt_column_name", type=str, default="instruction")
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    args = parser.parse_args()

    main(args)
