1. 4M parameter LLAMA model (BF16)

GPU RAM usage = 9GB

```
python run_clm.py \
    --model_name_or_path Maykeye/TinyLLama-v0 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --do_train \
    --do_eval \
    --num_train_epochs 128 \
    --block_size 1024 \
    --output_dir /tmp/test-clm
```

2. 125M params opt model (FP16)

GPU RAM = 7GB

```
rm -rf /tmp/test-clm && 
python run_clm.py \
    --model_name_or_path facebook/opt-125m \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --num_train_epochs 128 \
    --block_size 1024 \
    --output_dir /tmp/test-clm
```

After enabling the Gradient Checkpointing

GPU RAM = 8GB for bs=8

```
rm -rf /tmp/test-clm && 
python run_clm.py \
    --model_name_or_path facebook/opt-125m \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --gradient_checkpointing \
    --num_train_epochs 32 \
    --block_size 1024 \
    --output_dir /tmp/test-clm
```