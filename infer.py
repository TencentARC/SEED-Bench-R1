from qwen_vl_utils import process_vision_info
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import os
import argparse
import json
from tqdm import tqdm
from math_verify import parse, verify
import re
import numpy as np

QUESTION_TEMPLATE = "{Question} Output the thinking process in <think> </think> and final answer in <answer> </answer> tags, i.e., <think> reasoning process here </think><answer> answer here </answer>. "

def make_conversation_egoplan(example, data_root_dir):
    options = [
        example['choice_a'],
        example['choice_b'],
        example['choice_c'],
        example['choice_d'],
    ]

    answer_index = example['golden_choice_idx']

    problem = f"{example['question']}\n" + "\n".join([f"{chr(65 + i)}. {option}" for i, option in enumerate(options)]) + "\n"
    solution = f"<answer>{answer_index}.</answer>"

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
        "sample_id": example["sample_id"],
        "prompt": [
            {
                "role": "user",
                "content": content,
            },
        ],
        'problem': problem,
        'solution': solution,
    }
    return feature

            

def make_conversation_longvideobench(example, data_root_dir):
    options = example['candidates']
    answer_index = chr(65 + example['correct_choice'])

    problem = f"{example['question']}\n" + "\n".join([f"{chr(65 + i)}. {option}" for i, option in enumerate(options)]) + "\n"
    solution = f"<answer>{answer_index}.</answer>"
    video_path = os.path.join(data_root_dir, "videos", example["video_path"])
    content = [
        {"type": "video", "video": video_path},
        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=problem)},
    ]

    feature = {
        "sample_id": example["sample_id"],
        "prompt": [
            {
                "role": "user",
                "content": content,
            },
        ],
        'problem': problem,
        'solution': solution,
    }
    return feature


def accuracy_reward(content, sol):
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
    return reward





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/group/40101/milkcychen/Open-R1-Video/ckpt/Qwen2-VL-7B-EgoPlan-GRPO/egoplan-it-8k-remove-formatreward-matchletterreward-f16export/checkpoint-850")
    parser.add_argument("--test_file_path", default="/group/40121/public_datasets/SEED-Bench-R1/annotations/validation_L1.jsonl")
    parser.add_argument("--data_root_dir", default="/group/40121/public_datasets/SEED-Bench-R1")
    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument("--do_sampling", type=int, default=1)
    # parser.add_argument("--")
    args = parser.parse_args()

    # default: Load the model on the available device(s)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    processor = AutoProcessor.from_pretrained(args.model_path)

    test_file_basename = os.path.basename(args.test_file_path)
    output_eval_results_path = os.path.join(args.model_path, f"eval_results_for_{test_file_basename}")
    
    rewards = []
    evaluated_example_ids = set()
    if os.path.exists(output_eval_results_path):
        with open(output_eval_results_path) as fi:
            for line in tqdm(fi):
                example = json.loads(line)
                evaluated_example_ids.add(example['sample_id'])
                rewards.append(example['reward'])

    test_examples = []
    if args.test_file_path.endswith('.jsonl'):
        with open(args.test_file_path) as f:
            examples = []
            for line in f:
                example = json.loads(line)
                examples.append(example)
    else:
        with open(args.test_file_path) as f:
            examples = json.load(f)


    for example in examples:
        if 'sample_id' not in example:
            example['sample_id'] = example['id']
        if example['sample_id'] not in evaluated_example_ids:
            test_examples.append(example)
        else:
            print(f"skip {example['sample_id']}")


    t = tqdm(total=len(test_examples))
    
    with open(output_eval_results_path, 'a') as fo:
        for i in range(0, len(test_examples), args.test_batch_size):
            batch = test_examples[i:i+args.test_batch_size]
            if 'EgoPlan' in args.data_root_dir:
                features = [make_conversation_egoplan(example, args.data_root_dir) for example in batch]
            elif 'LongVideoBench' in args.data_root_dir:
                features = [make_conversation_longvideobench(example, args.data_root_dir) for example in batch]
            else:
                raise NotImplementedError
            prompts = [feature['prompt'] for feature in features]
            prompt_texts = [
                processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                for msg in prompts
            ]
            image_inputs, video_inputs, video_kwargs = process_vision_info(prompts, return_video_kwargs=True)
            prompt_inputs = processor(
                text=prompt_texts,
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
            )
            prompt_inputs = prompt_inputs.to("cuda")

            # Inference
            generated_ids = model.generate(**prompt_inputs, 
                max_new_tokens=256,
                do_sample=args.do_sampling,
                temperature=1
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(prompt_inputs.input_ids, generated_ids)
            ]
            output_texts = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            for prompt, output_text, feature in zip(prompts, output_texts, features):
                reward = accuracy_reward(output_text, feature['solution'])
                rewards.append(reward)
                print(f"------------- Sample {feature['sample_id']} -------------\n")
                print(f"Question: {prompt[0]['content'][-1]['text']}\n")
                print(f"Content: {output_text}\n")
                print(f"Solution: {feature['solution']}\n")
                print(f"Reward: {reward}\n")

                feature['response'] = output_text
                feature['reward'] = reward
                fo.write(json.dumps(feature)+"\n")
                fo.flush()

                t.update(1)
                t.set_postfix(reward_mean=np.mean(rewards))

