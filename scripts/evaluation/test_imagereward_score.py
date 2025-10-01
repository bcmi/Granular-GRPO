import torch
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
from torch.nn import functional as F
from open_clip import create_model_from_pretrained, get_tokenizer
import ImageReward as RM


def initialize_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dict = {}
    ## download from https://huggingface.co/zai-org/ImageReward
    model_path = "ckpt/ImageReward/ImageReward.pt"
    config_path = "ckpt/ImageReward/med_config.json"
    model = RM.load(model_path, device=device, med_config=config_path)

    return model, device
    
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
    model, device = initialize_model()

    reward_model = model.to(device)
    reward_model.eval()

    img_folder = "IMAGE_SAVE_FOLDER"
    images, filenames = load_images_from_folder(img_folder)

    eval_rewards = []
    with torch.no_grad():
        for image_pil, filename in tqdm(zip(images, filenames), total=400):
            prompt = os.path.splitext(filename)[0]
            ## get score
            rewards = reward_model.score(prompt, image_pil)

            eval_rewards.append(rewards)

    avg_reward = sum(eval_rewards) / len(eval_rewards) if eval_rewards else 0
    print(f"Average image reward score: {avg_reward:.4f}")

if __name__ == "__main__":
    main()