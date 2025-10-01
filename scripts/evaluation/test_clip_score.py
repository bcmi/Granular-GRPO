from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
import torch
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
from torch.nn import functional as F
from open_clip import create_model_from_pretrained, get_tokenizer

def initialize_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dict = {}

    processor = get_tokenizer('ViT-H-14')
    reward_model, preprocess_dgn5b = create_model_from_pretrained(
        'local-dir:ckpt/clip_score') 
    reward_model.to(device).eval()
    model_dict['model'] = reward_model
    model_dict['preprocess_val'] = preprocess_dgn5b

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

            image = preprocess_val(image_pil).unsqueeze(0).to(device=device, non_blocking=True)
            prompt = os.path.splitext(filename)[0]
            text = tokenizer([prompt]).to(device=device, non_blocking=True)

            ## get score
            clip_image_features = reward_model.encode_image(image)
            clip_text_features = reward_model.encode_text(text)
            clip_image_features = F.normalize(clip_image_features, dim=-1)
            clip_text_features = F.normalize(clip_text_features, dim=-1)
            clip_score = (clip_image_features @ clip_text_features.T)[0]
            clip_score = clip_score.item()
            eval_rewards.append(clip_score)

    avg_reward = sum(eval_rewards) / len(eval_rewards) if eval_rewards else 0
    print(f"Average CLIP score: {avg_reward:.4f}")

if __name__ == "__main__":
    main()