import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_nifti(file_path):
    """
    Analyze a NIfTI file and print its information.
    
    Args:
        file_path (str): Path to the NIfTI file (.nii or .nii.gz)
    """
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        return

    try:
        # Load the NIfTI file
        nifti_img = nib.load(file_path)
        print("NIfTI file loaded successfully.\n")

        # Get the image data as a NumPy array
        data = nifti_img.get_fdata()
        
        # Get the header
        header = nifti_img.header

        # --- Basic Information ---
        print("=== NIfTI File Information ===")
        print(f"File Path: {file_path}")
        
        # Shape of the image
        print(f"Shape: {data.shape}")
        
        # Number of dimensions
        print(f"Number of Dimensions: {data.ndim}")
        
        # Data type
        print(f"Data Type: {data.dtype}")
        
        # Voxel dimensions (in mm)
        voxel_sizes = header.get_zooms()
        print(f"Voxel Sizes (mm): {voxel_sizes}")
        
        # Number of voxels
        total_voxels = np.prod(data.shape)
        print(f"Total Number of Voxels: {total_voxels}")
        
        # Unique values in the data
        unique_values = np.unique(data)
        print(f"Number of Unique Values: {len(unique_values)}")
        print(f"Unique Values (first 10): {unique_values[:10]}")
        
        # Min and max values
        print(f"Minimum Value: {np.min(data)}")
        print(f"Maximum Value: {np.max(data)}")
        
        # Mean and standard deviation
        print(f"Mean Value: {np.mean(data):.4f}")
        print(f"Standard Deviation: {np.std(data):.4f}")
        
        # --- Header Information ---
        print("\n=== Header Information ===")
        print(f"Header Keys: {list(header.keys())}")
        print(f"Affine Matrix:\n{header.get_best_affine()}")
        print(f"Qform Code: {header['qform_code']}")
        print(f"Sform Code: {header['sform_code']}")
        print(f"Data Scaling Slope: {header['scl_slope']}")
        print(f"Data Scaling Intercept: {header['scl_inter']}")

        # --- Visualization ---
        # Display a middle slice (if 3D or 4D)
        if data.ndim >= 3:
            # Select middle slice along the last dimension (e.g., axial slice for 3D)
            slice_idx = data.shape[-1] // 2
            if data.ndim == 3:
                slice_data = data[:, :, slice_idx]
            elif data.ndim == 4:
                slice_data = data[:, :, slice_idx, 0]  # Take first timepoint if 4D

            plt.figure(figsize=(8, 8))
            plt.imshow(slice_data, cmap='gray')
            plt.title(f"Middle Slice (Index {slice_idx})")
            plt.colorbar(label='Intensity')
            plt.axis('off')
            plt.show()
        else:
            print("Visualization skipped: Data is not 3D or 4D.")

    except Exception as e:
        print(f"Error processing NIfTI file: {str(e)}")

def main():
    # Example usage
    # Replace with your NIfTI file path
    # nifti_file_path = "JIPMER-DATA/test/liver_20_mask.nii"  # Update this path
    nifti_file_path = "data_preprocessed/test/1A_image.nii.gz"  # Update this path
    # nifti_file_path = "/home/icmr/Documents/MultiPhaseSegmentation/SegFormer/JIPMER_DATASET/niigz liver/Venous Phase/LS2V.nii"  # Update this path
    analyze_nifti(nifti_file_path)

if __name__ == "__main__":
    main()

