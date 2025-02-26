# 4M parameter LLAMA model (BF16)

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

# 125M params opt model (FP16)

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

## Enable the Gradient Checkpointing

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

## torch.compile()

[Instruction](https://huggingface.co/docs/transformers/en/perf_train_gpu_one#using-torchcompile)

Note: the "inductor" backend is not supported because my 1080Ti GPU is too old.

TODO: profile the performance

When torch.compile() is enabled: 7.3GB bs=4

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
    --gradient_checkpointing \
    --torch_compile \
    --torch_compile_backend=cudagraphs \
    --num_train_epochs 32 \
    --block_size 1024 \
    --output_dir /tmp/test-clm
```

```
When torch.compile() is disabled: 5.3GB bs=4
rm -rf /tmp/test-clm && 
python run_clm.py \
    --model_name_or_path facebook/opt-125m \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --gradient_checkpointing \
    --num_train_epochs 32 \
    --block_size 1024 \
    --output_dir /tmp/test-clm
```