import torch
from diffusers import FluxPipeline
from diffusers import FluxTransformer2DModel

model_path = "/mnt/shared-storage-user/zhouyujie/DanceGRPO/save_exp/spo_v35/ckpt/checkpoint-300-0"
flux_path = "/mnt/shared-storage-user/mllm/bujiazi/model_ckpts/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"
device = "cuda:0"

transformer = FluxTransformer2DModel.from_pretrained(model_path, use_safetensors=True, torch_dtype=torch.float16).to(device)
pipe = FluxPipeline.from_pretrained(flux_path, transformer=None,  torch_dtype=torch.float16).to(device)
pipe.transformer = transformer

prompt = "A golden Labrador retriever is leaping excitedly on the green grass, chasing a soap bubble that glows with a rainbow in the sun, National Geographic photography style"

image = pipe(
    prompt,
    guidance_scale=3.5,
    height=1024,
    width=1024,
    num_inference_steps=50,
    max_sequence_length=512,
).images[0]

save_path = "g2rpo.png"
image.save(save_path)