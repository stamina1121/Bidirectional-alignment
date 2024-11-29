import lm_eval
from lm_eval import utils as lm_eval_utils
from lm_eval.api.registry import ALL_TASKS
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table
import argparse
import transformers
def evaluate(args):
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=False, use_auth_token=args.hf_token)
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model, use_auth_token=args.hf_token)
    model.to(args.device)
    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.lm_eval_batch_size)

    task_names = args.tasks.split(',')
    results = lm_eval.simple_evaluate(hflm, tasks=task_names, batch_size=args.lm_eval_batch_size, num_fewshot=args.num_fewshot)
    if "groups" in results:
        print(make_table(results, "groups"))
    else:
        print(make_table(results))
    print(make_table(results))

if __name__ == "__main__":
    #Adding necessary input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',type=str)
    parser.add_argument('--lm_eval_batch_size',default = 1, type=int)   
    parser.add_argument('--num_fewshot',default = None, type=int)   
    parser.add_argument('--tasks',default=None,type=str)
    parser.add_argument('--hf_token', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    evaluate(args)

