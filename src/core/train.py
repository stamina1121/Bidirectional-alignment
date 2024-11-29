"""
The code is borrowed from https://github.com/lm-sys/FastChat/blob/main/fastchat/train/train.py

Here is the original license:
    https://github.com/lm-sys/FastChat/blob/main/LICENSE
"""

import math
import pathlib

import transformers

from src.core.trainer import ScheduledTrainer
from src.data import make_supervised_data_module,make_supervised_data_module_train_eval
from src.utils.config import DataArguments, ModelArguments, TrainingArguments
from src.utils.io import safe_save_model_for_hf_trainer
from transformers import EarlyStoppingCallback


def train():
    #import ipdb;ipdb.set_trace()
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False

    # Load model and tokenizer
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

    # Load data
    if training_args.evaluation_strategy != 'no':
        print('loading training data and eval data')
        data_module = make_supervised_data_module_train_eval(tokenizer=tokenizer, data_args=data_args)
    else:
        print('just training data')
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    # Start trainner
    trainer = ScheduledTrainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module)

    #if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
    if 'checkpoint' in model_args.model_name_or_path:
        #trainer.train(resume_from_checkpoint=model_args.model_name_or_path)
        trainer.train(resume_from_checkpoint=model_args.model_name_or_path)
    else:
        trainer.train()
    model.config.use_cache = True
    trainer.save_model()
    trainer.save_state()
    #safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
