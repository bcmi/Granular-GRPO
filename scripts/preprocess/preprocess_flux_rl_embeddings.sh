GPU_NUM=8 # 2,4,8
MODEL_PATH="/mnt/shared-storage-user/mllm/bujiazi/model_ckpts/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"
OUTPUT_DIR="data/rl_embeddings"

torchrun --nproc_per_node=$GPU_NUM --master_port 19002 \
    fastvideo/data_preprocess/preprocess_flux_embedding.py \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --prompt_dir "./prompts.txt"