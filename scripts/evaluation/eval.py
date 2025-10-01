import os
import re
import torch
import torch.distributed as dist
from pathlib import Path
from diffusers import FluxPipeline
from diffusers import FluxTransformer2DModel
from torch.utils.data import Dataset, DistributedSampler

class PromptDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r') as f:
            self.prompts = [line.strip() for line in f if line.strip()]
        
    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]

def sanitize_filename(text, max_length=200):
    sanitized = re.sub(r'[\\/:*?"<>|]', '_', text)
    return sanitized[:max_length].rstrip() or "untitled"

def distributed_setup():
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size

def main():
    rank, local_rank, world_size = distributed_setup()
    
    model_path = "CKPT_PATH"
    flux_path = "./ckpt/flux"

    transformer = FluxTransformer2DModel.from_pretrained(model_path, use_safetensors=True, torch_dtype=torch.float16).to("cuda")
    pipe = FluxPipeline.from_pretrained(flux_path, transformer=None,  torch_dtype=torch.float16).to("cuda")
    pipe.transformer = transformer

    dataset = PromptDataset("scripts/evaluation/prompt_test.txt")
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    output_dir = Path("IMAGE_SAVE_FOLDER")
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx in sampler:
        prompt = dataset[idx]
        try:
            generator = torch.Generator(device=f"cuda:{local_rank}")
            generator.manual_seed(42 + idx + rank*1000)
            
            image = pipe(
                prompt,
                guidance_scale=3.5,
                height=1024,
                width=1024,
                num_inference_steps=50,
                max_sequence_length=512,
                generator=generator,
            ).images[0]

            filename = sanitize_filename(prompt)
            save_path = output_dir / f"{filename}.png"
            image.save(save_path)

            print(f"[Rank {rank}] Generated: {save_path.name}")

        except Exception as e:
            print(f"[Rank {rank}] Error processing '{prompt[:20]}...': {str(e)}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()