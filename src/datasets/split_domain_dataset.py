import os
os.environ["CUDA_ViSIBLE_DEVICES"] = "2"

from datasets import load_dataset, Dataset, concatenate_datasets
from imagenet_label import ImageNetAnalyzer
from nltk.corpus import wordnet as wn
import networkx as nx
import clip
from PIL import Image
from tqdm import tqdm
import io
import sys
import logging
import shutil
sys.stdout.reconfigure(line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

query_labels = []
with open("src/datasets/imagenet_synsets.txt", "r") as f:
    for line in f:
        if line.startswith("n"):
            wnid = line.split()[0]
            query_labels.append(wnid)

synsets = [wn.synset_from_pos_and_offset('n', int(wnid[1:])) for wnid in query_labels]

analyzer = ImageNetAnalyzer(synsets, query_wnids=query_labels)
hierarchy_graph = analyzer.build_hierarchy_graph(max_depth=20)
subtrees = analyzer.find_deep_subtrees(min_depth=5, num_subtrees=10)
domain_subtrees = []
tokenizer = clip.tokenize
_, preprocess = clip.load("RN50", device="cuda")

for idx, (root, size, level) in enumerate(subtrees):
    descendants = nx.descendants(analyzer.hierarchy_graph, root)
    subtree_nodes = [root] + list(descendants)
    for i, nodes in enumerate(subtree_nodes):
        nodes = nodes.split(".")[1]
        subtree_nodes[i] = nodes
    domain_subtrees.append(subtree_nodes)
logger.info("domain_subtrees: %s", domain_subtrees)

def process_data(example, index):
    try:
        example["class_id"] = index
        example["cap_tokens"] = tokenizer(example["recaption_short"], truncate=True)
        
        # Add error handling to skip problematic images
        try:
            img = Image.open(io.BytesIO(example["image"]))
            # Try to convert to RGB mode, which may skip some problematic images
            img = img.convert('RGB')
            example["processed_img"] = preprocess(img).cpu().numpy().tolist()
        except (ValueError, IOError) as e:
            logger.warning(f"Skipping problematic image {example['id']}: {str(e)}")
            return None  # Return None indicates this sample should be skipped
            
        return example
    except Exception as e:
        logger.error(f"Error processing sample {example['id']}: {str(e)}")
        return None

BATCH_SIZE = 1000  # Number of samples to process in each batch
TEMP_DIR = "temp_batches"  # Directory for temporary batch storage

def process_and_save_batches():
    logger.info("Loading dataset...")
    dataset = load_dataset("dataset/data", split="train", streaming=True, cache_dir="./cache")
    
    # Create temporary directories for each domain
    for domain_idx in range(len(domain_subtrees)):
        os.makedirs(f"{TEMP_DIR}/domain_{domain_idx}", exist_ok=True)
    
    batch_counters = {i: 0 for i in range(len(domain_subtrees))}
    skipped_count = 0  # Record number of skipped samples
    
    # Process data in batches
    for example in tqdm(dataset, desc="Processing examples", miniters=10000):
        for domain_idx, wnid_list in enumerate(domain_subtrees):
            if any(example["id"].startswith(sid) for sid in wnid_list):
                index = wnid_list.index(example["id"].split("_")[0])
                processed_example = process_data(example, index)
                
                # Skip this sample if processing failed
                if processed_example is None:
                    skipped_count += 1
                    continue
                
                # Add processed data to temporary list
                if not hasattr(process_and_save_batches, f'batch_{domain_idx}'):
                    setattr(process_and_save_batches, f'batch_{domain_idx}', [])
                
                getattr(process_and_save_batches, f'batch_{domain_idx}').append(processed_example)
                
                # Save when batch reaches specified size
                if len(getattr(process_and_save_batches, f'batch_{domain_idx}')) >= BATCH_SIZE:
                    batch_data = getattr(process_and_save_batches, f'batch_{domain_idx}')
                    batch_dataset = Dataset.from_list(batch_data)
                    batch_dataset = batch_dataset.remove_columns(["image", "recaption"])
                    
                    # Save batch
                    batch_path = f"{TEMP_DIR}/domain_{domain_idx}/batch_{batch_counters[domain_idx]}"
                    batch_dataset.save_to_disk(batch_path)
                    
                    # Clean up memory
                    setattr(process_and_save_batches, f'batch_{domain_idx}', [])
                    batch_counters[domain_idx] += 1
                break
    
    logger.info(f"Processing completed, skipped {skipped_count} problematic samples")
    
    # Save remaining incomplete batch data
    for domain_idx in range(len(domain_subtrees)):
        if hasattr(process_and_save_batches, f'batch_{domain_idx}'):
            remaining_data = getattr(process_and_save_batches, f'batch_{domain_idx}')
            if remaining_data:
                batch_dataset = Dataset.from_list(remaining_data)
                batch_dataset = batch_dataset.remove_columns(["image", "recaption"])
                batch_path = f"{TEMP_DIR}/domain_{domain_idx}/batch_{batch_counters[domain_idx]}"
                batch_dataset.save_to_disk(batch_path)

def combine_and_split_datasets():
    logger.info("Combining and splitting datasets...")
    for domain_idx in range(len(domain_subtrees)):
        # Read all batches for this domain
        batch_paths = [f"{TEMP_DIR}/domain_{domain_idx}/batch_{i}" 
                      for i in range(len(os.listdir(f"{TEMP_DIR}/domain_{domain_idx}")))]
        
        if not batch_paths:
            continue
            
        # Load and combine all batches
        datasets = [Dataset.load_from_disk(path) for path in batch_paths]
        combined_dataset = concatenate_datasets(datasets)

        combined_dataset = combined_dataset.map(lambda x: {'class_id': x['class_id'] + domain_idx * 10})
        
        # Split into training and test sets
        train_test_split = combined_dataset.train_test_split(test_size=0.1)
        train_set = train_test_split["train"]
        test_set = train_test_split["test"]
        
        # Save final datasets
        os.makedirs(f"data/domain_dataset_{domain_idx}", exist_ok=True)
        os.makedirs(f"data/domain_dataset_{domain_idx}_test", exist_ok=True)
        
        train_set.save_to_disk(f"data/domain_dataset_{domain_idx}")
        test_set.save_to_disk(f"data/domain_dataset_{domain_idx}_test")
        logger.info(f"domain_dataset_{domain_idx} saved successfully!")

def main():
    # Create temporary directory
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # Process data and save batches
    process_and_save_batches()
    
    # Combine batches and create final datasets
    combine_and_split_datasets()

if __name__ == "__main__":
    main()