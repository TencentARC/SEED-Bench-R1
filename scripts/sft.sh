export PROJECT_ROOT="/group/40101/milkcychen/SEED-Bench-R1"

accelerate launch --config_file ${PROJECT_ROOT}/scripts/zero3.yaml \
src/open_r1_egoplan/sft.py \
--config ${PROJECT_ROOT}/scripts/qwen2vl_sft_config.yaml

