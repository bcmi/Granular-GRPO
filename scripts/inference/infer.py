import torch
from diffusers import FluxPipeline
from diffusers import FluxTransformer2DModel
from safetensors.torch import load_file

device = "cuda:0"

model_path = "ckpt/g2rpo/diffusion_pytorch_model.safetensors"
flux_path = "ckpt/flux"

pipe = FluxPipeline.from_pretrained(flux_path, use_safetensors=True,  torch_dtype=torch.float16)
model_state_dict = load_file(model_path)
pipe.transformer.load_state_dict(model_state_dict, strict=True)
pipe = pipe.to(device)

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