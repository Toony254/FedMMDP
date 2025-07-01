from datasets import load_from_disk
import os

# Create output directory
output_dir = "/home/bd/data/zs/FedMMDP/data/processed_datasets"
os.makedirs(output_dir, exist_ok=True)

for domain in range(5, 10):
    # Load dataset
    dataset = load_from_disk("/home/bd/data/zs/FedMMDP/data/" + f"domain_dataset_{domain}")
    
    # Process dataset
    dataset = dataset.map(lambda x: {'class_id': x['class_id'] + (domain % 5) * 10})
    
    # Save processed dataset
    output_path = os.path.join(output_dir, f"domain_dataset_{domain}")
    dataset.save_to_disk(output_path)
    print(f"Processed and saved dataset for domain {domain} to {output_path}")

    dataset = load_from_disk("/home/bd/data/zs/FedMMDP/data/" + f"domain_dataset_{domain}_test")
    
    # Process dataset
    dataset = dataset.map(lambda x: {'class_id': x['class_id'] + (domain % 5) * 10})
    
    # Save processed dataset
    output_path = os.path.join(output_dir, f"domain_dataset_{domain}_test")
    dataset.save_to_disk(output_path)
    print(f"Processed and saved dataset for domain {domain} to {output_path}")

print("All datasets have been processed and saved successfully!")