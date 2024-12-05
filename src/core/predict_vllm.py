import argparse

from vllm import LLM, SamplingParams

from src.data import InferenceDataset
from src.utils.io import dump_jsonlines, load_jsonlines


def main(args):
    llm = LLM(
        args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=0.8,  # 降低显存占用比例
        dtype=args.dtype,
        
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )

    raw_data = load_jsonlines(args.data_filepath)
    data = InferenceDataset(
        raw_data,
        content_name=args.prompt_column_name,
        reverse=args.reverse,
    )
    prompts = data.get_all()  #len(prompts = 50200)
    #tokenizer = llm.tokenizer  # 使用 LLM 的分词器
    #prompt_token_lengths = [len(tokenizer(prompt)["input_ids"]) for prompt in prompts]
    #total_token_length = sum(prompt_token_lengths)

    # 00:25 / 100 prompts on one GPU
    print("warning:prompts numver 20k!!!!")
    results = llm.generate(prompts[:16800], use_tqdm=True, sampling_params=sampling_params)

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
                "full_prompt": result.prompt, ###通过prompt判断输入输出
                "instruction": result.outputs[0].text,
                "response": raw[args.prompt_column_name], ###实际上response是results，只是为了方便后续的微调。
                }
            )
    else:
        for raw, result in zip(raw_data, results):
            dump_jsonl.append(
                {
                "full_prompt": result.prompt,
                "instruction": raw[args.prompt_column_name],
                "response": result.outputs[0].text,
                }
            )
    dump_jsonlines(dump_jsonl, args.save_filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--data_filepath", type=str)
    parser.add_argument("--save_filepath", type=str)
    parser.add_argument("--prompt_column_name", type=str, default="instruction")
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    args = parser.parse_args()

    main(args)
