import os
import torchio as tio
from typing import List, Dict, Any

def get_prediction_loader(input_dir: str, batch_size: int = 1, num_workers: int = 2):
    subjects = []
    for fname in os.listdir(input_dir):
        if fname.endswith('_image.nii.gz'):
            subject_id = fname.replace('_image.nii.gz', '')
            image_path = os.path.join(input_dir, f"{subject_id}_image.nii.gz")
            subjects.append({
                'subject_id': subject_id,
                'image': image_path
            })
    tio_subjects = [
        tio.Subject(
            image=tio.ScalarImage(s['image']),
            subject_id=s['subject_id']
        ) for s in subjects
    ]
    dataset = tio.SubjectsDataset(tio_subjects)
    loader = tio.SubjectsLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return loader, subjects 