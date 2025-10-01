from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
import torch
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
from torch.nn import functional as F
from open_clip import create_model_from_pretrained, get_tokenizer
from transformers import AutoProcessor, AutoModel

def initialize_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dict = {}

    process_path = "ckpt/CLIP-ViT-H-14-laion2B-s32B-b79K"
    # download from https://huggingface.co/yuvalkirstain/PickScore_v1
    model_path = "ckpt/PickScore_v1"

    processor = AutoProcessor.from_pretrained(process_path)
    reward_model = AutoModel.from_pretrained(model_path)
    reward_model.to(device).eval()

    model_dict['model'] = reward_model
    model_dict['preprocess_val'] = processor

    return model_dict, device

def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        if filename.endswith(".png"):
            img_path = os.path.join(folder, filename)
            image = Image.open(img_path).convert("RGB")
            images.append(image)
            filenames.append(filename)
    return images, filenames

def main():
    model_dict, device = initialize_model()
    model = model_dict['model']
    preprocess_val = model_dict['preprocess_val']

    tokenizer = get_tokenizer('ViT-H-14')
    reward_model = model.to(device)
    reward_model.eval()

    img_folder = "IMAGE_SAVE_FOLDER"
    images, filenames = load_images_from_folder(img_folder)

    eval_rewards = []
    with torch.no_grad():
        for image_pil, filename in tqdm(zip(images, filenames), total=400):
                
            image_inputs = preprocess_val(
                images=[image_pil],
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(device)

            prompt = os.path.splitext(filename)[0]  # 剔除文件扩展名
            
            text_inputs = preprocess_val(
                text=prompt,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(device)

            # Get embeddings
            image_embs = reward_model.get_image_features(**image_inputs)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

            text_embs = reward_model.get_text_features(**text_inputs)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

            # Calculate scores
            score = reward_model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
            eval_rewards.append(score.item())

    avg_reward = sum(eval_rewards) / len(eval_rewards) if eval_rewards else 0
    print(f"Average pickscore score: {avg_reward:.4f}")

if __name__ == "__main__":
    main()