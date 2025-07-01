import os
os.environ['CUDA_VISIBLE_DEVICES']='3'
import pickle
from collections import Counter
from datasets import Dataset, concatenate_datasets
num_datasets = 5
for i in range(1, num_datasets + 1):
    # Reload the dataset from local files
    dataset_list = []
    for file in os.listdir("data/full_dataset/dataset{}".format(i)):
        if file.endswith(".pkl"):
            dataset_path = os.path.join("data/full_dataset/dataset{}".format(i), file)
            print(f"Loading dataset from {dataset_path}")
            with open(dataset_path, "rb") as f:
                imgcap_dataset = pickle.load(f)
                dataset_list.append(imgcap_dataset)

    print("Dataset reloaded from local files.")

    for j in range(len(dataset_list)):
        dataset_list[j] = Dataset.from_list(dataset_list[j])
    full_dataset = concatenate_datasets(dataset_list)

    # Filter the dataset to keep only classes with at least 100 samples
    class_counts = Counter(sample["class"] for sample in full_dataset)

    popular_classes = {cls: count for cls, count in class_counts.items() if count >= 100}
    dataset = full_dataset.filter(lambda x: x["class"] in popular_classes)
    print(f"Filtered dataset size: {len(dataset)}")

    print("Class:{}, Count:{}".format(popular_classes.keys(), popular_classes.values()))

    print("Class_size:{}".format(len(popular_classes)))

    class_to_id = {cls: idx for idx, cls in enumerate(set(dataset["class"]))}
    
    def convert_class_to_id(example):
        example["class_id"] = class_to_id[example["class"]]
        return example

    dataset = dataset.map(convert_class_to_id)

    dataset_split = dataset.train_test_split(test_size=0.1)
    dataset_train = dataset_split['train']
    dataset_test = dataset_split['test']
    if not os.path.exists("data/filtered_dataset/dataset{}_train".format(i)):
        os.makedirs("data/filtered_dataset/dataset{}_train".format(i))
    if not os.path.exists("data/filtered_dataset/dataset{}_test".format(i)):
        os.makedirs("data/filtered_dataset/dataset{}_test".format(i))
    dataset_train.save_to_disk("data/filtered_dataset/dataset{}_train".format(i))
    dataset_test.save_to_disk("data/filtered_dataset/dataset{}_test".format(i))
    print(f"Dataset {i} saved to disk.")