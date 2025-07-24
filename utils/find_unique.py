import nibabel as nib
import numpy as np

# === Set the path to your .nii or .nii.gz file ===
# nii_path = "data_preprocessed/train/liver_8_label.nii.gz"
nii_path = "data_preprocessed/train/9A_label.nii.gz"
# === Load the NIfTI file ===
nii_img = nib.load(nii_path)
nii_data = nii_img.get_fdata()

# === Get unique values ===
unique_vals = np.unique(nii_data)

# === Display ===
print(f"Unique values in {nii_path}:")
print(unique_vals)
