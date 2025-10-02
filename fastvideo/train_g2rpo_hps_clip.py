# Copyright (c) [2025] [FastVideo Team]
# Copyright (c) [2025] [ByteDance Ltd. and/or its affiliates.]
# SPDX-License-Identifier: [Apache License 2.0] 
#
# This file has been modified by [ByteDance Ltd. and/or its affiliates.] in 2025.
#
# Original file was released under [Apache License 2.0], with the full license text
# available at [https://github.com/hao-ai-lab/FastVideo/blob/main/LICENSE].
#
# This modified file is released under the same license.

import argparse
import math
import os
from pathlib import Path
from fastvideo.utils.parallel_states import (
    initialize_sequence_parallel_state,
    destroy_sequence_parallel_group,
    get_sequence_parallel_state,
    nccl_info,
)
from fastvideo.utils.communications_flux import sp_parallel_dataloader_wrapper
import time
from torch.utils.data import DataLoader
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from torch.utils.data.distributed import DistributedSampler
from fastvideo.utils.dataset_utils import LengthGroupedSampler
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from fastvideo.utils.fsdp_util import get_dit_fsdp_kwargs, apply_fsdp_checkpointing
from fastvideo.utils.load import load_transformer
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from fastvideo.dataset.latent_flux_rl_datasets import LatentDataset, latent_collate_function
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from fastvideo.utils.checkpoint import (
    save_checkpoint,
    save_lora_checkpoint,
)
from fastvideo.utils.logging_ import main_print
import cv2
from diffusers.image_processor import VaeImageProcessor

check_min_version("0.31.0")
import time
from collections import deque
import numpy as np
from einops import rearrange
import torch.distributed as dist
from torch.nn import functional as F
from typing import List
from PIL import Image
from diffusers import FluxTransformer2DModel, AutoencoderKL
from diffusers.utils.torch_utils import randn_tensor

def sd3_time_shift(shift, t):
    return (shift * t) / (1 + (shift - 1) * t)
    

def flow_grpo_step(
    model_output: torch.Tensor,
    latents: torch.Tensor,
    eta: float,
    sigmas: torch.Tensor,
    index: int,
    prev_sample: torch.Tensor,
    generator=None,
):

    device = model_output.device
    sigma = sigmas[index].to(device)
    sigma_prev = sigmas[index + 1].to(device)
    sigma_max = sigmas[1].item()
    dt = sigma_prev - sigma

    pred_original_sample = latents - sigma * model_output
 
    std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma))) * eta
    
    prev_sample_mean = latents*(1+std_dev_t**2/(2*sigma)*dt)+model_output*(1+std_dev_t**2*(1-sigma)/(2*sigma))*dt
    
    if prev_sample is None:
        variance_noise = randn_tensor(
            model_output.shape, 
            generator=generator, 
            device=device, 
            dtype=model_output.dtype
        )
        prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1*dt) * variance_noise
    
    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * ((std_dev_t * torch.sqrt(-1*dt))**2))
        - torch.log(std_dev_t * torch.sqrt(-1*dt))
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )

    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    return prev_sample, pred_original_sample, log_prob


def assert_eq(x, y, msg=None):
    assert x == y, f"{msg or 'Assertion failed'}: {x} != {y}"


def prepare_latent_image_ids(batch_size, height, width, device, dtype):
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )

    return latent_image_ids.to(device=device, dtype=dtype)

def pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

    return latents

def unpack_latents(latents, height, width, vae_scale_factor):
    batch_size, num_patches, channels = latents.shape

    # VAE applies 8x compression on images but we must also account for packing which requires
    # latent height and width to be divisible by 2.
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))

    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)

    latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

    return latents


def run_anchor_sample_step(
        args,
        z,
        progress_bar,
        sigma_schedule,
        transformer,
        encoder_hidden_states, 
        pooled_prompt_embeds, 
        text_ids,
        image_ids, 
        grpo_sample,
        eta,
    ):
    all_latents = [z]

    for i in progress_bar:
        B = encoder_hidden_states.shape[0]
        sigma = sigma_schedule[i]
        timestep_value = int(sigma * 1000)
        timesteps = torch.full([encoder_hidden_states.shape[0]], timestep_value, device=z.device, dtype=torch.long)
        transformer.eval()

        with torch.autocast("cuda", torch.bfloat16):
            pred= transformer(
                hidden_states=z,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timesteps/1000,
                guidance=torch.tensor(
                    [3.5],
                    device=z.device,
                    dtype=torch.bfloat16
                ),
                txt_ids=text_ids.repeat(encoder_hidden_states.shape[1],1),
                pooled_projections=pooled_prompt_embeds,
                img_ids=image_ids,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]

        z, pred_original, _ = flow_grpo_step(pred, z.to(torch.float32), 0, sigmas=sigma_schedule, index=i, prev_sample=None)
        all_latents.append(z.to(torch.bfloat16))

    all_latents = torch.stack(all_latents, dim=1)

    return pred_original, all_latents

def run_sde_sample_step(
        args,
        z,
        eta_step,
        sigma_schedule,
        transformer,
        encoder_hidden_states, 
        pooled_prompt_embeds, 
        text_ids,
        image_ids, 
        grpo_sample,
        eta,
    ):
    all_latents = [z]
    all_log_probs = []

    i=eta_step
    sigma = sigma_schedule[i]
    timestep_value = int(sigma * 1000)
    timesteps = torch.full([encoder_hidden_states.shape[0]], timestep_value, device=z.device, dtype=torch.long)
    transformer.eval()

    with torch.autocast("cuda", torch.bfloat16):
        pred= transformer(
            hidden_states=z,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timesteps/1000,
            guidance=torch.tensor(
                [3.5],
                device=z.device,
                dtype=torch.bfloat16
            ),
            txt_ids=text_ids.repeat(encoder_hidden_states.shape[1],1),
            pooled_projections=pooled_prompt_embeds,
            img_ids=image_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

    z, pred_original, log_prob = flow_grpo_step(pred, z.to(torch.float32), eta, sigmas=sigma_schedule, index=i, prev_sample=None)
    z.to(torch.bfloat16)
    all_latents.append(z)
    all_log_probs.append(log_prob)

    all_latents = torch.stack(all_latents, dim=1)
    all_log_probs = torch.stack(all_log_probs, dim=1)

    return all_latents, all_log_probs

def run_ode_sample_step(
        args,
        z,
        progress_bar,
        sigma_schedule,
        transformer,
        encoder_hidden_states, 
        pooled_prompt_embeds, 
        text_ids,
        image_ids, 
        grpo_sample,
        eta,
    ):

    for i in progress_bar:

        sigma = sigma_schedule[i]
        timestep_value = int(sigma * 1000)
        timesteps = torch.full([encoder_hidden_states.shape[0]], timestep_value, device=z.device, dtype=torch.long)
        transformer.eval()

        with torch.autocast("cuda", torch.bfloat16):
            pred= transformer(
                hidden_states=z,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timesteps/1000,
                guidance=torch.tensor(
                    [3.5],
                    device=z.device,
                    dtype=torch.bfloat16
                ),
                txt_ids=text_ids.repeat(encoder_hidden_states.shape[1],1), 
                pooled_projections=pooled_prompt_embeds,
                img_ids=image_ids,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]

        z, pred_original, _ = flow_grpo_step(pred, z.to(torch.float32), 0, sigmas=sigma_schedule, index=i, prev_sample=None)

    return pred_original


def get_pred(
            args,
            latents,
            encoder_hidden_states, 
            pooled_prompt_embeds, 
            text_ids,
            image_ids,
            transformer,
            timesteps,
):
    transformer.train()
    with torch.autocast("cuda", torch.bfloat16):
        pred= transformer(
            hidden_states=latents,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timesteps/1000,
            guidance=torch.tensor(
                [3.5],
                device=latents.device,
                dtype=torch.bfloat16
            ),
            txt_ids=text_ids.repeat(encoder_hidden_states.shape[1],1),
            pooled_projections=pooled_prompt_embeds,
            img_ids=image_ids.squeeze(0),
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]
    
    return pred

def sample_reference_model(
    args,
    device, 
    transformer,
    vae,
    encoder_hidden_states, 
    pooled_prompt_embeds, 
    text_ids,
    reward_model,
    tokenizer,
    caption,
    preprocess_val,
    reward_model_2,
    preprocess_val_2,
):
    w, h, t = args.w, args.h, args.t
    granular_list = args.granular_list

    sample_steps = args.sampling_steps
    sigma_schedule = torch.linspace(1, 0, args.sampling_steps + 1)
    sigma_schedule = sd3_time_shift(args.shift, sigma_schedule)

    assert_eq(
        len(sigma_schedule),
        sample_steps + 1,
        "sigma_schedule must have length sample_steps + 1",
    )

    B = encoder_hidden_states.shape[0] 
    SPATIAL_DOWNSAMPLE = 8 
    IN_CHANNELS = 16
    latent_w, latent_h = w // SPATIAL_DOWNSAMPLE, h // SPATIAL_DOWNSAMPLE
    batch_size = 1
    batch_indices = torch.chunk(torch.arange(B), B // batch_size)
    granular_nums = len(granular_list)

    all_input_latents = []
    all_output_latents = []
    all_log_probs = []
    all_rewards = [[] for _ in range(granular_nums)]
    all_rewards_2 = [[] for _ in range(granular_nums)]

    all_image_ids = []
    eval_rewards = []
    eval_rewards_2 = []

    if args.init_same_noise:
        input_latents = torch.randn(
            (1, IN_CHANNELS, latent_h, latent_w),
            device=device,
            dtype=torch.bfloat16,
        )
    
    for index, batch_idx in enumerate(batch_indices):
        batch_encoder_hidden_states = encoder_hidden_states[batch_idx]
        batch_pooled_prompt_embeds = pooled_prompt_embeds[batch_idx]
        batch_text_ids = text_ids[batch_idx]
        batch_caption = [caption[i] for i in batch_idx]

        image_ids = prepare_latent_image_ids(len(batch_idx), latent_h // 2, latent_w // 2, device, torch.bfloat16)
        grpo_sample = True

        step_rewards = [[] for _ in range(granular_nums)]
        step_rewards_2 = [[] for _ in range(granular_nums)]
        step_input_latents = []
        step_output_latents = []
        step_log_probs = []
        step_image_ids = []

        vae.enable_tiling()
        image_processor = VaeImageProcessor(16)
        rank = int(os.environ["RANK"])

        with torch.no_grad():
            if index % args.num_generations == 0:
                progress_bar = tqdm(range(0, sample_steps), desc="Anchor Progress")
                input_latents_new = pack_latents(input_latents, len(batch_idx), IN_CHANNELS, latent_h, latent_w)
                
                eval_latents, anchor_latents = run_anchor_sample_step(
                    args,
                    input_latents_new,
                    progress_bar,
                    sigma_schedule,
                    transformer,
                    batch_encoder_hidden_states,
                    batch_pooled_prompt_embeds,
                    batch_text_ids, 
                    image_ids,
                    grpo_sample, 
                    eta=0,
                )

                with torch.inference_mode():
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        eval_latents = unpack_latents(eval_latents, h, w, 8)
                        eval_latents = (eval_latents / 0.3611) + 0.1159
                        eval_image = vae.decode(eval_latents, return_dict=False)[0]
                        decoded_eval_image = image_processor.postprocess(eval_image)

                ## eval reward
                with torch.no_grad():
                    image_pil = decoded_eval_image[0]
                    image = preprocess_val(image_pil).unsqueeze(0).to(device=device, non_blocking=True)
                    image_2 = preprocess_val_2(image_pil).unsqueeze(0).to(device=device, non_blocking=True)

                    text = tokenizer([batch_caption[0]]).to(device=device, non_blocking=True)

                    with torch.amp.autocast('cuda'):
                        outputs = reward_model(image, text)
                        image_features, text_features = outputs["image_features"], outputs["text_features"]
                        logits_per_image = image_features @ text_features.T
                        hps_score = torch.diagonal(logits_per_image)

                        ## clip score
                        clip_image_features = reward_model_2.encode_image(image_2)
                        clip_text_features = reward_model_2.encode_text(text)
                        clip_image_features = F.normalize(clip_image_features, dim=-1)
                        clip_text_features = F.normalize(clip_text_features, dim=-1)
                        clip_score = (clip_image_features @ clip_text_features.T)[0]

                    eval_rewards.append(hps_score)
                    eval_rewards_2.append(clip_score)

            for eta_step in args.eta_step_list:
                input_sde_sample = anchor_latents[:, eta_step]

                batch_latents, batch_log_probs = run_sde_sample_step(
                    args,
                    input_sde_sample,
                    eta_step,
                    sigma_schedule,
                    transformer,
                    batch_encoder_hidden_states,
                    batch_pooled_prompt_embeds, 
                    batch_text_ids, 
                    image_ids, 
                    grpo_sample, 
                    eta=args.eta,
                )

                input_ode_latents = batch_latents[:, 1]

                for j, g in enumerate(granular_list):
                    prefix = sigma_schedule[:eta_step+2]
                    suffix = sigma_schedule[eta_step+2::g]
                    sigma_schedule_j = torch.cat((prefix, suffix))
                    sample_steps_j = len(sigma_schedule_j)
                    progress_bar_ode_j = tqdm(range(eta_step+1, sample_steps_j-1), desc=f"Sampling Progress {j+1}")
                    
                    latents_j = run_ode_sample_step(
                        args,
                        input_ode_latents,
                        progress_bar_ode_j,
                        sigma_schedule_j, 
                        transformer,
                        batch_encoder_hidden_states,
                        batch_pooled_prompt_embeds,
                        batch_text_ids, 
                        image_ids, 
                        grpo_sample,
                        eta=0,
                    )

                    dense_rewards_j = []
                    dense_rewards_j_2 = []

                    with torch.inference_mode():
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            latents_j = unpack_latents(latents_j, h, w, 8)
                            latents_j = (latents_j / 0.3611) + 0.1159
                            image_j = vae.decode(latents_j, return_dict=False)[0]
                            decoded_image_j = image_processor.postprocess(image_j)

                    with torch.no_grad():
                        image_pil_j = decoded_image_j[0]
                        image_j = preprocess_val(image_pil_j).unsqueeze(0).to(device=device, non_blocking=True)
                        image_j_2 = preprocess_val_2(image_pil_j).unsqueeze(0).to(device=device, non_blocking=True)

                        text = tokenizer([batch_caption[0]]).to(device=device, non_blocking=True)

                        with torch.amp.autocast('cuda'):
                            outputs = reward_model(image_j, text)
                            image_features, text_features = outputs["image_features"], outputs["text_features"]
                            logits_per_image = image_features @ text_features.T
                            hps_score = torch.diagonal(logits_per_image)

                            ## clip score
                            clip_image_features = reward_model_2.encode_image(image_j_2)
                            clip_text_features = reward_model_2.encode_text(text)
                            clip_image_features = F.normalize(clip_image_features, dim=-1)
                            clip_text_features = F.normalize(clip_text_features, dim=-1)
                            clip_score = (clip_image_features @ clip_text_features.T)[0]

                        dense_rewards_j.append(hps_score)
                        dense_rewards_j_2.append(clip_score)

                    step_rewards[j].append(torch.cat(dense_rewards_j, dim=0))
                    step_rewards_2[j].append(torch.cat(dense_rewards_j_2, dim=0))

                step_input_latents.append(batch_latents[:, 0])
                step_output_latents.append(batch_latents[:, 1])
                step_log_probs.append(batch_log_probs[:, 0])

        for j in range(granular_nums):
            ## hps
            step_rewards[j] = torch.stack(step_rewards[j], dim=1)
            all_rewards[j].append(step_rewards[j])
            ## clip
            step_rewards_2[j] = torch.stack(step_rewards_2[j], dim=1)
            all_rewards_2[j].append(step_rewards_2[j])

        all_input_latents.append(torch.stack(step_input_latents, dim=1))
        all_output_latents.append(torch.stack(step_output_latents, dim=1))
        all_log_probs.append(torch.stack(step_log_probs, dim=1))
        all_image_ids.append(image_ids)

    all_input_latents = torch.cat(all_input_latents, dim=0)
    all_output_latents = torch.cat(all_output_latents, dim=0)
    all_log_probs = torch.cat(all_log_probs, dim=0)
    all_image_ids = torch.stack(all_image_ids, dim=0)

    for j in range(granular_nums):
        all_rewards[j] = torch.cat(all_rewards[j], dim=0).to(torch.float32)
        all_rewards_2[j] = torch.cat(all_rewards_2[j], dim=0).to(torch.float32)

    eval_rewards = torch.cat(eval_rewards, dim=0).to(torch.float32)
    eval_rewards_2 = torch.cat(eval_rewards_2, dim=0).to(torch.float32)
    all_eval_rewards = [eval_rewards, eval_rewards_2]

    return all_rewards, all_rewards_2, all_input_latents, all_output_latents, all_log_probs, sigma_schedule, all_image_ids, all_eval_rewards
    

def gather_tensor(tensor):
    if not dist.is_initialized():
        return tensor
    world_size = dist.get_world_size()
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0)

def train_one_step(
    args,
    device,
    transformer,
    vae,
    reward_model,
    tokenizer,
    optimizer,
    lr_scheduler,
    loader,
    noise_scheduler,
    max_grad_norm,
    preprocess_val,
    reward_model_2,
    preprocess_val_2,
):
    total_loss = 0.0
    optimizer.zero_grad()


    (
        encoder_hidden_states, 
        pooled_prompt_embeds, 
        text_ids, 
        caption,
    ) = next(loader)

    def repeat_tensor(tensor):
        if tensor is None:
            return None
        return torch.repeat_interleave(tensor, args.num_generations, dim=0)

    encoder_hidden_states = repeat_tensor(encoder_hidden_states)
    pooled_prompt_embeds = repeat_tensor(pooled_prompt_embeds) 
    text_ids = repeat_tensor(text_ids) 

    if isinstance(caption, str):
        caption = [caption] * args.num_generations
    elif isinstance(caption, list):
        caption = [item for item in caption for _ in range(args.num_generations)]
    else:
        raise ValueError(f"Unsupported caption type: {type(caption)}")

    (
        all_rewards,
        all_rewards_2,
        all_input_latents, 
        all_output_latents, 
        all_log_probs, 
        sigma_schedule, 
        all_image_ids, 
        all_eval_rewards,
     ) = sample_reference_model(
            args,
            device, 
            transformer,
            vae,
            encoder_hidden_states,
            pooled_prompt_embeds, 
            text_ids,
            reward_model,  
            tokenizer,
            caption,
            preprocess_val,
            reward_model_2,
            preprocess_val_2,
        )

    batch_size = all_input_latents.shape[0]
    device = all_input_latents.device
    train_sigma_schedule = sigma_schedule.clone()[args.eta_step_list]
    timestep_value = [int(sigma * 1000) for sigma in train_sigma_schedule][:args.sampling_steps]

    timestep_values = [timestep_value[:] for _ in range(batch_size)]
    timesteps = torch.tensor(timestep_values, device=all_input_latents.device, dtype=torch.long)

    samples = {
        "timesteps": timesteps,
        "latents": all_input_latents,
        "next_latents": all_output_latents,
        "log_probs": all_log_probs,
        "all_rewards": all_rewards, ## hps
        "all_rewards_2": all_rewards_2, ## clip
        "image_ids": all_image_ids,
        "text_ids": text_ids,
        "encoder_hidden_states": encoder_hidden_states,
        "pooled_prompt_embeds": pooled_prompt_embeds,
    }

    gathered_reward_hps = gather_tensor(all_eval_rewards[0])
    gathered_reward_clip = gather_tensor(all_eval_rewards[1])

    # eval 
    if dist.get_rank()==0:
        print("gathered_hps_reward", gathered_reward_hps)
        reward_path = os.path.join(args.output_dir, "hps_reward.txt")
        with open(reward_path, 'a') as f: 
            f.write(f"{gathered_reward_hps.mean().item()}\n")

        print("gathered_clip_reward", gathered_reward_clip)
        reward_path = os.path.join(args.output_dir, "clip_reward.txt")
        with open(reward_path, 'a') as f: 
            f.write(f"{gathered_reward_clip.mean().item()}\n")

    n = len(samples["pooled_prompt_embeds"]) // (args.num_generations)

    ## hps
    hps_advantages = torch.zeros_like(samples["all_rewards"][0])

    for rewards in samples["all_rewards"]:
        group_advantages = torch.zeros_like(rewards)
        for i in range(n):
            start_idx = i * args.num_generations
            end_idx = (i + 1) * args.num_generations

            group_rewards = rewards[start_idx:end_idx]
            group_mean = group_rewards.mean(dim=0)
            group_std = group_rewards.std(dim=0) + 1e-8

            group_advantages[start_idx:end_idx] = (group_rewards - group_mean) / group_std

        hps_advantages += group_advantages

    ## clip
    clip_advantages = torch.zeros_like(samples["all_rewards_2"][0])

    for rewards in samples["all_rewards_2"]:
        group_advantages = torch.zeros_like(rewards)
        for i in range(n):
            start_idx = i * args.num_generations
            end_idx = (i + 1) * args.num_generations

            group_rewards = rewards[start_idx:end_idx]
            group_mean = group_rewards.mean(dim=0)
            group_std = group_rewards.std(dim=0) + 1e-8

            group_advantages[start_idx:end_idx] = (group_rewards - group_mean) / group_std

        clip_advantages += group_advantages

    samples["advantages"] = hps_advantages + clip_advantages

    train_timesteps = int(len(samples["timesteps"][0]))
    clip_range = args.clip_range
    adv_clip_max = args.adv_clip_max

    for t_idx in range(train_timesteps):

        lat_0   = samples["latents"][0, t_idx].unsqueeze(0)         
        t_0     = samples["timesteps"][0, t_idx].unsqueeze(0)       
        enc_0   = samples["encoder_hidden_states"][0].unsqueeze(0)  
        pooled  = samples["pooled_prompt_embeds"][0].unsqueeze(0)  
        text_0  = samples["text_ids"][0].unsqueeze(0)              
        image_0 = samples["image_ids"][0].unsqueeze(0)           

        pred = get_pred(args, lat_0, enc_0, pooled, text_0, image_0, transformer, t_0) 
        pred_batch = pred.repeat(args.num_generations, 1, 1)

        z, pred_original, new_log_probs = flow_grpo_step(
            pred_batch,
            samples["latents"][:, t_idx].float(), 
            args.eta,
            sigma_schedule,
            args.eta_step_list[t_idx],
            prev_sample=samples["next_latents"][:, t_idx].float(),
        )

        advantages = torch.clamp(
            samples["advantages"][:, t_idx], -adv_clip_max, adv_clip_max
        )

        ratio = torch.exp(new_log_probs - samples["log_probs"][:, t_idx])

        unclipped_loss = -advantages * ratio
        clipped_loss = -advantages * torch.clamp(
            ratio,
            1.0 - clip_range,
            1.0 + clip_range,
        )
        loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

        loss.backward()
        avg_loss = loss.detach().clone()
        dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
        total_loss += avg_loss.item()

        grad_norm = transformer.clip_grad_norm_(max_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        if dist.get_rank() % 8 == 0:
            print("ratio", ratio)
            print("advantage", advantages)
            print("final loss", loss.item())
        dist.barrier()

    return total_loss, grad_norm.item()


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True

    # ## debug with single GPU
    # local_rank = 0
    # rank = 0
    # world_size = 1
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12345'
    # os.environ["LOCAL_RANK"] = "0"
    # os.environ["RANK"] = "0"
    # dist.init_process_group(backend='nccl',rank=rank, world_size = world_size)

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl")

    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    initialize_sequence_parallel_state(args.sp_size)

    if args.seed is not None:
        set_seed(args.seed + rank)

    # Handle the repository creation
    if rank <= 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required
    from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
    from typing import Union
    import huggingface_hub
    from hpsv2.utils import root_path, hps_version_map

    model, preprocess_train, preprocess_val = create_model_and_transforms(
        'ViT-H-14',
        args.hps_clip_path,
        precision='amp',
        device=device,
        jit=False,
        force_quick_gelu=False,
        force_custom_text=False,
        force_patch_dropout=False,
        force_image_size=None,
        pretrained_image=False,
        image_mean=None,
        image_std=None,
        light_augmentation=True,
        aug_cfg={},
        output_dict=True,
        with_score_predictor=False,
        with_region_predictor=False
    )

    checkpoint = torch.load(args.hps_path, map_location=f'cuda:{device}')
    model.load_state_dict(checkpoint['state_dict'])
    processor = get_tokenizer('ViT-H-14')
    reward_model = model.to(device)
    reward_model.eval()

    ## clip score
    from open_clip import create_model_from_pretrained
    clip_model, clip_preprocess_val = create_model_from_pretrained(
        f'local-dir:{args.clip_score_path}') 

    reward_model_2 = clip_model.to(device)
    reward_model_2.eval()
    preprocess_val_2 = clip_preprocess_val
    
    transformer = FluxTransformer2DModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="transformer",
            torch_dtype = torch.float32
    )
    
    fsdp_kwargs, no_split_modules = get_dit_fsdp_kwargs(
        transformer,
        args.fsdp_sharding_startegy,
        False,
        args.use_cpu_offload,
        args.master_weight_type,
    )
    
    transformer = FSDP(transformer, **fsdp_kwargs,)

    if args.gradient_checkpointing:
        apply_fsdp_checkpointing(
            transformer, no_split_modules, args.selective_checkpointing
        )
    
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype = torch.bfloat16,
    ).to(device)

    main_print(
        f"--> Initializing FSDP with sharding strategy: {args.fsdp_sharding_startegy}"
    )
    # Load the reference model
    main_print(f"--> model loaded")

    # Set model as trainable.
    transformer.train()

    noise_scheduler = None

    params_to_optimize = transformer.parameters()
    params_to_optimize = list(filter(lambda p: p.requires_grad, params_to_optimize))

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8,
    )

    init_steps = 0
    main_print(f"optimizer: {optimizer}")

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=1000000,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
        last_epoch=init_steps - 1,
    )

    train_dataset = LatentDataset(args.data_json_path, args.num_latent_t, args.cfg)
    sampler = DistributedSampler(
            train_dataset, rank=rank, num_replicas=world_size, shuffle=True, seed=args.sampler_seed
        )

    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        collate_fn=latent_collate_function,
        pin_memory=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )

    # Train!
    total_batch_size = (
        args.train_batch_size 
        * world_size 
        / args.sp_size
        * args.train_sp_batch_size
    )
    main_print("***** Running training *****")
    main_print(f"  Num examples = {len(train_dataset)}")
    main_print(f"  Dataloader size = {len(train_dataloader)}")
    main_print(f"  Resume training from step {init_steps}")
    main_print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    main_print(
        f"  Total train batch size (w. data & sequence parallel, accumulation) = {total_batch_size}"
    )
    main_print(f"  Total optimization steps per epoch = {args.max_train_steps}")
    main_print(
        f"  Total training parameters per FSDP shard = {sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 1e9} B"
    )
    # print dtype
    main_print(f"  Master weight dtype: {transformer.parameters().__next__().dtype}")

    progress_bar = tqdm(
        range(0, 100000),
        initial=init_steps,
        desc="Steps",
        disable=local_rank > 0,
    )

    loader = sp_parallel_dataloader_wrapper(
        train_dataloader,
        device,
        args.train_batch_size,
        args.sp_size,
        args.train_sp_batch_size,
    )

    step_times = deque(maxlen=100)

    for epoch in range(1):
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)

        ## 1 - 301
        for step in range(init_steps+1, args.max_train_steps+1):
            start_time = time.time()

            # save ckpt
            if step > 150 and step % args.checkpointing_steps == 0:
                ckpt_path = os.path.join(args.output_dir, "ckpt")
                save_checkpoint(transformer, rank, ckpt_path,
                                step, epoch)

                dist.barrier()

            loss, grad_norm = train_one_step(
                args,
                device, 
                transformer,
                vae,
                reward_model,
                processor,
                optimizer,
                lr_scheduler,
                loader,
                noise_scheduler,
                args.max_grad_norm,
                preprocess_val,
                reward_model_2,
                preprocess_val_2,
            )

            step_time = time.time() - start_time
            step_times.append(step_time)
            avg_step_time = sum(step_times) / len(step_times)
    
            progress_bar.set_postfix(
                {
                    "loss": f"{loss:.4f}",
                    "step_time": f"{step_time:.2f}s",
                    "grad_norm": grad_norm,
                }
            )
            progress_bar.update(1)

    if get_sequence_parallel_state():
        destroy_sequence_parallel_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset & dataloader
    parser.add_argument("--data_json_path", type=str, default="data/rl_embeddings/videos2caption.json")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--num_latent_t",
        type=int,
        default=1,
        help="number of latent frames",
    )

    parser.add_argument("--pretrained_model_name_or_path", 
        type=str, 
        default="/mnt/shared-storage-user/mllm/bujiazi/model_ckpts/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"
    )

    ## reward model path
    parser.add_argument(
        "--hps_path",
        type=str,
        default="/mnt/shared-storage-user/mllm/zhouyujie/HPSv2/HPS_v2.1_compressed.pt",
        help="path to load hps reward model",
    )

    parser.add_argument(
        "--hps_clip_path",
        type=str,
        default='/mnt/shared-storage-user/mllm/zhouyujie/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin',
        help="path to load hps clip model",
    )

    parser.add_argument(
        "--clip_score_path",
        type=str,
        default="/mnt/shared-storage-user/mllm/zhouyujie/DFN5B-CLIP-ViT-H-14-384",
        help="path to load clip score reward model"
    )

    # diffusion setting
    parser.add_argument("--cfg", type=float, default=0.0)

    # validation & logs
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="exp_flux/test",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    # optimizer & scheduler & Training
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=300,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=True,
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument("--selective_checkpointing", type=float, default=1.0)
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--use_cpu_offload",
        action="store_true",
        help="Whether to use CPU offload for param & gradient & optimizer states.",
    )

    parser.add_argument("--sp_size", type=int, default=1, help="For sequence parallel")
    parser.add_argument(
        "--train_sp_batch_size",
        type=int,
        default=1,
        help="Batch size for sequence parallel training",
    )

    parser.add_argument("--fsdp_sharding_startegy", default="full")

    # lr_scheduler
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of cycles in the learning rate scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0001, help="Weight decay to apply."
    )
    parser.add_argument(
        "--master_weight_type",
        type=str,
        default="fp32",
        help="Weight type to use - fp32 or bf16.",
    )

    #GRPO training
    parser.add_argument(
        "--h",
        type=int,
        default=720,   
        help="video height",
    )
    parser.add_argument(
        "--w",
        type=int,
        default=720,   
        help="video width",
    )
    parser.add_argument(
        "--t",
        type=int,
        default=1,   
        help="video length",
    )
    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=16,   
        help="sampling steps",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.7,   
        help="noise eta",
    )
    parser.add_argument(
        "--sampler_seed",
        type=int,
        default=1223627,   
        help="seed of sampler",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=12,   
        help="num_generations per prompt",
    )
    parser.add_argument(
        "--init_same_noise",
        action="store_true",
        default=True,
        help="whether use the same noise within each prompt",
    )
    parser.add_argument(
        "--shift",
        type = float,
        default=3.0,
        help="shift for timestep scheduler",
    )
    parser.add_argument(
        "--clip_range",
        type = float,
        default=1e-4,
        help="clip range for grpo",
    )
    parser.add_argument(
        "--adv_clip_max",
        type = float,
        default=5.0,
        help="clipping advantage",
    )
    parser.add_argument(
        "--eta_step_list",
        nargs='+', 
        type=int,  
        help="A list of integers for eta steps.",
        default=[1]
    )
    parser.add_argument(
        "--granular_list",
        nargs='+', 
        type=int,  
        help="A list of integers for different chosen granularities.",
        default=[1,2]
    )

    args = parser.parse_args()
    main(args)