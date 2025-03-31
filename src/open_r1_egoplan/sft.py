# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging
import os
import sys

import datasets
from dataclasses import dataclass, field
from typing import Optional
import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, set_seed, AutoProcessor
from transformers.trainer_utils import get_last_checkpoint
import trl
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from datasets import Dataset, DatasetDict
from qwen_vl_utils import process_vision_info
logger = logging.getLogger(__name__)


@dataclass
class SFTConfig(trl.SFTConfig):
    """
    args for callbacks, benchmarks etc
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The benchmarks to run after training."}
    )
    callbacks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The callbacks to run during training."}
    )
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use for benchmarking."},
    )
    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The Hub model branch to push the model to."},
    )
    jsonl_path: Optional[str] = field(
        default=None,
        metadata={"help": "json file path"},
    )
    data_root_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Root directory to datasets"},
    )
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})



processor = None


def convert_example(example):
    messages = example['prompt']
    messages.append({
        "role": "assistant",
        "content": [{"type": "text", "text": example['response']}],
    })
    example["messages"] = messages
    return example


def collate_fn(examples):
    for example in examples:
        for message in example["prompt"]:
            if isinstance(message["content"], list):
                new_content = []
                for ele in message["content"]:
                    new_ele = {k: v for k, v in ele.items() if v is not None}
                    if 'video' in new_ele:
                        new_ele['video'] = os.path.join(training_args.data_root_dir, new_ele['video'])
                    if 'image' in new_ele:
                        new_ele['image'] = os.path.join(training_args.data_root_dir, new_ele['image'])
                    new_content.append(new_ele)
                message["content"] = new_content
        convert_example(example)

    texts = [
        processor.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
        for example in examples
    ]
    image_inputs = []
    video_inputs = []
    for example in examples:
        imgs, vids = process_vision_info(example["messages"])
        image_inputs.append(imgs)
        video_inputs.append(vids)
    batch = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
    )

    labels = batch["input_ids"].clone()
    # labels[labels == processor.tokenizer.pad_token_id] = -100
    # image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    # labels[labels == image_token_id] = -100

    im_start_token_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
    for i in range(labels.size(0)):
        last_index = (labels[i] == im_start_token_id).nonzero(as_tuple=True)[0][-1].item()
        labels[i, :last_index+3] = -100

    batch["labels"] = labels
    return batch


def create_dataset_from_jsonl_simple(jsonl_path):
    base_dataset = Dataset.from_json(jsonl_path)
    return DatasetDict({
        "train": base_dataset
    })

def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Data parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ################
    # Load datasets
    ################

    if training_args.jsonl_path:
        # # load dataset from jsonl
        dataset = create_dataset_from_jsonl_simple(training_args.jsonl_path)
    else:
        # Load the dataset
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)


    ################
    # Load tokenizer
    ################
    global processor
    if "vl" in model_args.model_name_or_path.lower():
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
        )
        logger.info("Using AutoProcessor for vision-language model.")
    else:
        processor = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
        )
        logger.info("Using AutoTokenizer for text-only model.")
    if hasattr(processor, "pad_token") and processor.pad_token is None:
        processor.pad_token = processor.eos_token
    elif hasattr(processor.tokenizer, "pad_token") and processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    ###################
    # Model init kwargs
    ###################
    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    # training_args.model_init_kwargs = model_kwargs
    from transformers import Qwen2VLForConditionalGeneration
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path, **model_kwargs
    )
    ############################
    # Initialize the SFT Trainer
    ############################
    training_args.dataset_kwargs = {
        "skip_prepare_dataset": True,
    }
    training_args.remove_unused_columns = False
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=processor.tokenizer,
        data_collator=collate_fn,
        peft_config=get_peft_config(model_args)
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["R1-V"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)
    #############
    # push to hub
    #############

    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)
        processor.push_to_hub(training_args.hub_model_id)




if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)