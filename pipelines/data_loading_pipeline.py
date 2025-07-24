import os
import torchio as tio
from torch.utils.data import DataLoader
from typing import List, Dict, Any

def get_subjects_list(split_dir: str) -> List[Dict[str, Any]]:
    subjects = []
    for fname in os.listdir(split_dir):
        if fname.endswith('_image.nii.gz'):
            subject_id = fname.replace('_image.nii.gz', '')
            image_path = os.path.join(split_dir, f"{subject_id}_image.nii.gz")
            label_path = os.path.join(split_dir, f"{subject_id}_label.nii.gz")
            if os.path.exists(label_path):
                subjects.append({
                    'subject_id': subject_id,
                    'image': image_path,
                    'label': label_path
                })
    return subjects

def get_dataloader(data_dir: str, split: str, batch_size: int, num_workers: int = 4, shuffle: bool = False) -> DataLoader:
    split_dir = os.path.join(data_dir, split)
    subjects_list = get_subjects_list(split_dir)
    subjects = [
        tio.Subject(
            image=tio.ScalarImage(s['image']),
            label=tio.LabelMap(s['label']),
            subject_id=s['subject_id']
        ) for s in subjects_list
    ]
    dataset = tio.SubjectsDataset(subjects)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader
