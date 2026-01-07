import torch

def collate_fn(batch):
    processed_imgs = []
    cap_tokens = []
    other_data = {}
    
    for item in batch:
        if "processed_img" in item:
            if isinstance(item["processed_img"], torch.Tensor):
                processed_imgs.append(item["processed_img"])
            else:
                processed_imgs.append(torch.tensor(item["processed_img"], dtype=torch.float))

        if "cap_tokens" in item:
            if isinstance(item["cap_tokens"], torch.Tensor):
                cap_tokens.append(item["cap_tokens"])
            else:
                cap_tokens.append(torch.tensor(item["cap_tokens"], dtype=torch.long))
                
        for k, v in item.items():
            if k not in ["processed_img", "cap_tokens"]:
                if k not in other_data:
                    other_data[k] = []
                other_data[k].append(v)

    result = {}
    if processed_imgs:
        result["processed_img"] = torch.stack(processed_imgs)
    if cap_tokens:
        result["cap_tokens"] = torch.stack(cap_tokens)

    for k, v in other_data.items():
        result[k] = v
    
    return result