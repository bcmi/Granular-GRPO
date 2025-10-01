GPU_NUM=8 # 2,4,8
MODEL_PATH="./ckpt/flux"
OUTPUT_DIR="data/rl_embeddings"

torchrun --nproc_per_node=$GPU_NUM --master_port 19002 \
    fastvideo/data_preprocess/preprocess_flux_embedding.py \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --prompt_dir "./prompts.txt"