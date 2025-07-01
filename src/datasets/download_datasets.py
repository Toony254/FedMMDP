import os
import pickle
from datasets import load_dataset
import clip
import torch
from tqdm import tqdm
seed, buffer_size = 42, 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

dataset = load_dataset("gmongaras/Imagenet21K_Recaption", split="train", streaming=True)

tokenizer = clip.tokenize
clip_model, preprocess = clip.load("ViT-B/32", device = device)

def tokenize_function(examples):
    tokens = tokenizer(examples["recaption_short"], truncate=True)
    return {"cap_tokens": tokens[0]}
def img_preprocess(examples):
    processed_image = preprocess(examples["image"].convert("RGB"))
    return {"processed_img": processed_image}

for i in tqdm(range(10, 30), desc="Processing shards"):
    subset = dataset.shard(num_shards=7770, index=i)
    subset = subset.remove_columns("recaption")
    
    imgcap_dataset = subset.shuffle(seed=seed, buffer_size=buffer_size)
    imgcap_dataset = imgcap_dataset.map(tokenize_function, batched=False)
    imgcap_dataset = imgcap_dataset.map(img_preprocess, batched=False)
    imgcap_dataset = imgcap_dataset.remove_columns("image")
    
    # Define the directory to save the dataset
    save_dir = "data/"
    os.makedirs(save_dir, exist_ok=True)

    # Save the dataset as a list of samples
    dataset_path = os.path.join(save_dir, f"imgcap_dataset{i}.pkl")
    with open(dataset_path, "wb") as f:
        pickle.dump(list(imgcap_dataset), f)
    print(f"Dataset saved to {dataset_path}")