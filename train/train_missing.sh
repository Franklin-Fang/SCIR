nproc_per_node=4

CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=$nproc_per_node \
swift sft \
    --model ./model/Qwen3-4B \
    --dataset ./data/train/train-missing.json \
    --model_type qwen3 \
    --split_dataset_ratio 0.05 \
    --torch_dtype bfloat16 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --eval_strategy steps \
    --output_dir ./model/missing \
    --eval_steps 20 \
    --num_train_epochs 2 \
    --save_steps 100 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --loss_scale ignore_empty_think \
    --system ' You are an information extraction result verification expert. Please complete the verification task as required.' \
    --warmup_ratio 0.05 