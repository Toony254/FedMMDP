from __future__ import annotations

import json
import math
import pickle
from pathlib import Path

from datasets import load_from_disk


CLIP_ROOT = Path("/home/bd/data/zs/FedMMDP/preprocessed_iapr")
SIGLIP_ROOT = Path("/home/bd/data/zs/FedMMDP/preprocessed_iapr_siglip")
CLIP_DOMAIN_ROOT = CLIP_ROOT / "domain_datasets"
SIGLIP_DOMAIN_ROOT = SIGLIP_ROOT / "domain_datasets"
NUM_DOMAINS = 5


def approx_unit_norm(vec):
    norm = math.sqrt(sum(float(x) * float(x) for x in vec))
    return norm


def main() -> None:
    problems = []

    with open(CLIP_ROOT / "label2id.json", "r", encoding="utf-8") as f:
        clip_label2id = json.load(f)
    with open(SIGLIP_ROOT / "label2id.json", "r", encoding="utf-8") as f:
        siglip_label2id = json.load(f)
    if clip_label2id != siglip_label2id:
        problems.append("label2id.json differs")

    for domain_idx in range(NUM_DOMAINS):
        clip_mapping_path = CLIP_DOMAIN_ROOT / f"domain_dataset_{domain_idx}" / "class_mapping.pkl"
        siglip_mapping_path = SIGLIP_DOMAIN_ROOT / f"domain_dataset_{domain_idx}" / "class_mapping.pkl"
        with open(clip_mapping_path, "rb") as f:
            clip_mapping = pickle.load(f)
        with open(siglip_mapping_path, "rb") as f:
            siglip_mapping = pickle.load(f)
        if clip_mapping != siglip_mapping:
            problems.append(f"domain {domain_idx}: class_mapping differs")

        for split_name in ("train", "test"):
            clip_ds = load_from_disk(str(CLIP_DOMAIN_ROOT / f"domain_dataset_{domain_idx}" / split_name))
            siglip_ds = load_from_disk(str(SIGLIP_DOMAIN_ROOT / f"domain_dataset_{domain_idx}" / split_name))

            if clip_ds.num_rows != siglip_ds.num_rows:
                problems.append(f"domain {domain_idx} {split_name}: row count differs")
                continue
            if clip_ds.column_names != siglip_ds.column_names:
                problems.append(f"domain {domain_idx} {split_name}: columns differ {clip_ds.column_names} vs {siglip_ds.column_names}")

            clip_first = clip_ds[0]
            siglip_first = siglip_ds[0]
            if len(clip_first["processed_img"]) != 1024:
                problems.append(f"domain {domain_idx} {split_name}: clip processed_img dim is {len(clip_first['processed_img'])}, expected 1024")
            if len(siglip_first["processed_img"]) != 768:
                problems.append(f"domain {domain_idx} {split_name}: siglip processed_img dim is {len(siglip_first['processed_img'])}, expected 768")
            if len(siglip_first["cap_tokens"]) != 768:
                problems.append(f"domain {domain_idx} {split_name}: siglip cap_tokens dim is {len(siglip_first['cap_tokens'])}, expected 768")

            for idx in range(clip_ds.num_rows):
                clip_row = clip_ds[idx]
                siglip_row = siglip_ds[idx]
                for field in ("id", "class_id", "domain_id"):
                    if clip_row[field] != siglip_row[field]:
                        problems.append(
                            f"domain {domain_idx} {split_name} row {idx}: field {field} differs "
                            f"{clip_row[field]} vs {siglip_row[field]}"
                        )
                        break

            clip_norm = approx_unit_norm(clip_first["processed_img"])
            siglip_img_norm = approx_unit_norm(siglip_first["processed_img"])
            siglip_txt_norm = approx_unit_norm(siglip_first["cap_tokens"])
            print(
                f"domain {domain_idx} {split_name}: rows={clip_ds.num_rows}, "
                f"columns={clip_ds.column_names}, clip_dim=1024, siglip_dim=768, "
                f"norms=(clip_img={clip_norm:.6f}, siglip_img={siglip_img_norm:.6f}, siglip_txt={siglip_txt_norm:.6f})"
            )

    if problems:
        print("FOUND DIFFERENCES:")
        for item in problems:
            print(f"- {item}")
        raise SystemExit(1)

    print("All structural checks passed.")
    print("Expected representation difference only: CLIP features are 1024-dim, SigLIP features are 768-dim.")


if __name__ == "__main__":
    main()
