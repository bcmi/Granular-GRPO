from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
import torch
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm

def initialize_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dict = {}
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        'ViT-H-14',
        'ckpt/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin',
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
    model_dict['model'] = model
    model_dict['preprocess_val'] = preprocess_val
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

    cp = "ckpt/hps/HPS_v2.1_compressed.pt"
    checkpoint = torch.load(cp, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    tokenizer = get_tokenizer('ViT-H-14')
    reward_model = model.to(device)
    reward_model.eval()

    img_folder = "IMAGE_SAVE_FOLDER"
    images, filenames = load_images_from_folder(img_folder)

    eval_rewards = []
    with torch.no_grad():
        for image_pil, filename in tqdm(zip(images, filenames), total=400):

            image = preprocess_val(image_pil).unsqueeze(0).to(device=device, non_blocking=True)
            prompt = os.path.splitext(filename)[0]  # 剔除文件扩展名
            text = tokenizer([prompt]).to(device=device, non_blocking=True)
            outputs = reward_model(image, text)
            image_features, text_features = outputs["image_features"], outputs["text_features"]
            logits_per_image = image_features @ text_features.T
            hps_score = torch.diagonal(logits_per_image).item()  # 转换为 Python 数值
            eval_rewards.append(hps_score)

    avg_reward = sum(eval_rewards) / len(eval_rewards) if eval_rewards else 0
    print(f"Average HPS score: {avg_reward:.4f}")

if __name__ == "__main__":
    main()