Train a 4M parameter LLAMA model

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