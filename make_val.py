from argparse import ArgumentParser
import json
import numpy as np
from sklearn.model_selection import StratifiedKFold
import os

def create_stratified_folds(
    load_anno_path: str,
    save_dir: str,
    n_splits: int,
    seed: int = 42):
    """
    Splits COCO annotations into stratified folds for cross-validation.

    The stratification is based on the number of annotations per image.
    For each fold, it creates a train and a validation annotation file.

    Args:
        load_anno_path (str): Path to the COCO annotation file to split.
        save_dir (str): Directory where the output fold files will be saved.
        n_splits (int): The number of folds to create.
        seed (int): Random seed for reproducibility.
    """
    os.makedirs(save_dir, exist_ok=True)

    with open(load_anno_path) as f:
        coco_data = json.load(f)
    
    images = coco_data['images']
    annotations = coco_data['annotations']
    categories = coco_data['categories']
    
    # Create a map of image_id to its annotation count for stratification
    image_id_to_ann_count = {img['id']: 0 for img in images}
    for ann in annotations:
        if ann['image_id'] in image_id_to_ann_count:
            image_id_to_ann_count[ann['image_id']] += 1

    image_indices = np.arange(len(images))
    image_ann_counts = np.array([image_id_to_ann_count[img['id']] for img in images])

    # Perform stratified k-fold split based on annotation counts
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    print(f"Creating {n_splits} stratified folds...")

    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(image_indices, image_ann_counts)):
        train_images = [images[i] for i in train_indices]
        val_images = [images[i] for i in val_indices]

        train_ids = {img['id'] for img in train_images}
        val_ids = {img['id'] for img in val_images}

        train_annotations = [ann for ann in annotations if ann['image_id'] in train_ids]
        val_annotations = [ann for ann in annotations if ann['image_id'] in val_ids]

        print(f"Fold {fold_idx}:")
        print(f"  Train images: {len(train_images)}, annotations: {len(train_annotations)}")
        print(f"  Val images: {len(val_images)}, annotations: {len(val_annotations)}")

        # Create COCO structure for the training fold
        train_coco = {
            'info': coco_data.get('info', {}),
            'licenses': coco_data.get('licenses', []),
            'images': train_images,
            'annotations': train_annotations,
            'categories': categories
        }

        # Create COCO structure for the validation fold
        val_coco = {
            'info': coco_data.get('info', {}),
            'licenses': coco_data.get('licenses', []),
            'images': val_images,
            'annotations': val_annotations,
            'categories': categories
        }

        train_save_path = os.path.join(save_dir, f"train_fold_{fold_idx}.json")
        val_save_path = os.path.join(save_dir, f"val_fold_{fold_idx}.json")

        with open(train_save_path, 'w', encoding='utf-8') as f:
            json.dump(train_coco, f, ensure_ascii=False, indent=2)
        
        with open(val_save_path, 'w', encoding='utf-8') as f:
            json.dump(val_coco, f, ensure_ascii=False, indent=2)
        
        print(f"  Saved: {train_save_path}")
        print(f"  Saved: {val_save_path}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Create stratified folds for COCO annotations.")
    parser.add_argument(
        "--load-anno-path", 
        default='./dataset/annotations/train.json', 
        type=str,
        help="Path to the master COCO annotation file."
    )
    parser.add_argument(
        "--save-dir", 
        default='./dataset/annotations/folds', 
        type=str,
        help="Directory to save the folded annotation files."
    )
    parser.add_argument(
        "--n-splits", 
        default=5, 
        type=int,
        help="Number of folds to create."
    )
    parser.add_argument(
        "--seed", 
        default=42, 
        type=int,
        help="Random seed for the split."
    )
    args = parser.parse_args()
    
    create_stratified_folds(
        load_anno_path=args.load_anno_path,
        save_dir=args.save_dir,
        n_splits=args.n_splits,
        seed=args.seed
    )
