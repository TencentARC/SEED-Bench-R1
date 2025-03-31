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

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from datasets import load_dataset
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
from open_r1_egoplan.trainer import Qwen2VLGRPOTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
import random
from functools import partial

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy",],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    jsonl_path: Optional[str] = field(
        default=None,
        metadata={"help": "json file path"},
    )
    data_root_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Root directory to datasets"},
    )
    freeze_visual_encoder: Optional[int] = field(
        default=0,
        metadata={"help": "Whether to freeze visual encoder"},
    )

def accuracy_reward(completions, solution, prompts, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    print(f"------------- Sample {kwargs['sample_id'][0]} -------------")
    print(prompts[0][0]['content'][-1]['text']+'\n')
    print(solution[0]+'\n')
    print(contents[:2]) # print online completion
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    for content, sol, prompt, sample_id in zip(contents, solution, prompts, kwargs['sample_id']):
        reward = 0.0
        # Try symbolic verification first
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        # If symbolic verification failed, try string matching
        if reward == 0.0:
            try:
                # Extract answer from solution if it has think/answer tags
                sol_match = re.search(r"<answer>(.*?)</answer>", sol, re.DOTALL)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

                # Extract answer from content if it has think/answer tags
                content_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
                # student_answer = content_match.group(1).strip() if content_match else content.strip()
                if content_match:
                    student_answer = content_match.group(1).strip()
                    # HACK, if First letter is correct reward 1
                    # Compare the extracted answers
                    if student_answer[0] == ground_truth[0]:
                        reward = 1.0
                else:
                    reward = 0.0
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail

        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward for Sample {sample_id}: {reward} -------------\n")
                f.write(f"Question: {prompt[0]['content'][-1]['text']}\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

from datasets import Dataset, DatasetDict
import json

def create_dataset_from_jsonl_simple(jsonl_path):
    base_dataset = Dataset.from_json(jsonl_path)
    return DatasetDict({
        "train": base_dataset
    })


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    
    if script_args.jsonl_path:
        # # load dataset from jsonl
        dataset = create_dataset_from_jsonl_simple(script_args.jsonl_path)
    else:
        # Load the dataset
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    QUESTION_TEMPLATE = "{Question} Output the thinking process in <think> </think> and final answer in <answer> </answer> tags, i.e., <think> reasoning process here </think><answer> answer here </answer>. "

    def make_conversation_egoplan(example, data_root_dir):
        if 'golden_choice_idx' not in example:
            negative_answers = random.sample(example["negative_answers"], 3)
            options = negative_answers + [example["answer"]]
        else:
            options = [example['choice_a'], example['choice_b'], example['choice_c'], example['choice_d']]

        random.shuffle(options)
        answer_index = options.index(example["answer"])
        problem = f"{example['question']}\n" + "\n".join([f"{chr(65 + i)}. {option}" for i, option in enumerate(options)]) + "\n"
        solution = f"<answer>{chr(65 + answer_index)}.</answer>"
        

        content = []
        if len(example['task_progress_metadata']) > 0:
            video_path = os.path.join(data_root_dir, 'videos', example['video_source'], example['video_basename'])
            content.append({"type": "video", "video": video_path})

        image_path = os.path.join(data_root_dir, 'images', example['video_source'], example['current_observation_basename'])
        content.extend([
            {"type": "image", "image": image_path},
            {"type": "text", "text": QUESTION_TEMPLATE.format(Question=problem)},
        ])

        feature = {
            "prompt": [
                {
                    "role": "user",
                    "content": content,
                },
            ],
            'problem': problem,
            'solution': solution,
        }

        # if len(example['task_progress_metadata']) > 0:
        #     video_path = os.path.join(data_root_dir, 'videos', example['video_source'], example['video_basename'])
        #     feature['video'] = video_path
        # image_path = os.path.join(data_root_dir, 'images', example['video_source'], example['current_observation_basename'])
        # feature['image'] = image_path

        return feature

    dataset = dataset.map(partial(make_conversation_egoplan, data_root_dir=script_args.data_root_dir))
    # if "image" in dataset[script_args.dataset_train_split].features:
    #     dataset = dataset.map(make_conversation_image)  # Utilize multiprocessing for faster mapping
    # elif "video" in dataset[script_args.dataset_train_split].features:
    #     dataset = dataset.map(
    #         make_conversation_video,
    #     )
    # else:
    #     dataset = dataset.map(make_conversation)
    #     dataset = dataset.remove_columns("messages")
    
    # import pdb; pdb.set_trace()

    trainer_cls = Qwen2VLGRPOTrainer

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    training_args.freeze_visual_encoder = script_args.freeze_visual_encoder
    main(script_args, training_args, model_args)
