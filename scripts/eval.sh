python -u infer.py \
--model_path "checkpoint_path" \
--test_file_path "/group/40101/milkcychen/SEED-Bench-R1/data/annotations/validation_L1.jsonl" \
--data_root_dir "/group/40101/milkcychen/SEED-Bench-R1/data" \
--test_batch_size 1

python -u infer.py \
--model_path "checkpoint_path" \
--test_file_path "/group/40101/milkcychen/SEED-Bench-R1/data/annotations/validation_L2.jsonl" \
--data_root_dir "/group/40101/milkcychen/SEED-Bench-R1/data" \
--test_batch_size 1


python -u infer.py \
--model_path "checkpoint_path" \
--test_file_path "/group/40101/milkcychen/SEED-Bench-R1/data/annotations/validation_L3.jsonl" \
--data_root_dir "/group/40101/milkcychen/SEED-Bench-R1/data" \
--test_batch_size 1