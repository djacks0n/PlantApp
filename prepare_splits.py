

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from dataset import CLASS_TO_IDX, CLASS_NAMES, SPLIT_DIR

CSV_PATH = "lookalikes_16_classes_dataset.csv"
IMAGE_DIR = Path("plant_images")
SEED = 67  # keep the same so the proportions are the same

def main():
    df = pd.read_csv(CSV_PATH)

    species_col = "scientific_name"
    print(f"Using species column: {species_col}")

    # Map species names to class indices
    df["label"] = df[species_col].map(CLASS_TO_IDX)
    df["label"] = df["label"].astype(int)

    print("\nImages per class:")
    counts = df["label"].value_counts().sort_index()
    for idx, n in counts.items():
        print(f"  {idx:2d}  {CLASS_NAMES[idx]:<30} {n}")

    # Stratified split
    
    keep = df[["image_path", "label", species_col, "Invasive"]]
    
    train_df, temp_df = train_test_split(
        keep, test_size=0.30, train_size=0.70, stratify=keep["label"], random_state=SEED
    )
    
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, train_size=0.50, stratify=temp_df["label"], random_state=SEED
    )

    SPLIT_DIR.mkdir(exist_ok=True)
    train_df.to_csv(SPLIT_DIR / "train.csv", index=False)
    val_df.to_csv(SPLIT_DIR / "val.csv", index=False)
    test_df.to_csv(SPLIT_DIR / "test.csv", index=False)

    print(f"\n Saved splits: train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")


if __name__ == "__main__":
    main()