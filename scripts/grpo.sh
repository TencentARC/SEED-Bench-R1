export WANDB_PROJECT=Qwen2-VL-7B-GRPO
export WANDB_NAME=training_6k-remove-formatreward-matchletterreward-f16
export DEBUG_MODE=true
export PROJECT_ROOT=/group/40101/milkcychen/SEED-Bench-R1
export LOG_PATH=${PROJECT_ROOT}/output_ckpt/$WANDB_PROJECT/$WANDB_NAME/completions_log.txt

mkdir -p ${PROJECT_ROOT}/output_ckpt/$WANDB_PROJECT/$WANDB_NAME

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12352" \
    src/open_r1_egoplan/grpo.py \
    --deepspeed scripts/zero3_offload.json \
    --output_dir ${PROJECT_ROOT}/output_ckpt/$WANDB_PROJECT/$WANDB_NAME \
    --model_name_or_path ${PROJECT_ROOT}/pretrained_ckpt/Qwen2-VL-7B-Instruct \
    --dataset_name xxx \
    --jsonl_path ${PROJECT_ROOT}/data/annotations/training_6k.jsonl \
    --data_root_dir ${PROJECT_ROOT}/data \
    --max_prompt_length 8192 \
    --learning_rate 1e-6 \
    --beta 0.1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $WANDB_NAME \
    --save_steps 10 \
    --save_only_model true

