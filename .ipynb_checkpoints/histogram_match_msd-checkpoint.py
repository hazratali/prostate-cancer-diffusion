import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
from glob import glob

def linear_match_intensities(source_data, reference_data):
    """
    Matches the Mean and Std Dev of the source tissue to the reference tissue.
    Includes robust clipping to prevent outliers.
    """
    # 1. Identify tissue 
    # We use a low threshold to find non-zero pixels
    src_mask = source_data > 1e-3
    ref_mask = reference_data > 1e-3
    
    # Safety: If image is empty, return as is
    if np.sum(src_mask) == 0 or np.sum(ref_mask) == 0:
        return source_data

    # Calculate statistics on TISSUE ONLY
    src_mean = np.mean(source_data[src_mask])
    src_std = np.std(source_data[src_mask])
    
    ref_mean = np.mean(reference_data[ref_mask])
    ref_std = np.std(reference_data[ref_mask])

    # Apply linear transform (Z-Score matching)
    # (X - mu_s) / sigma_s * sigma_r + mu_r
    matched_data = (source_data - src_mean) / (src_std + 1e-8) * ref_std + ref_mean

    # Clean up
    # Clip values to 0 (no negative MRI values)
    matched_data = np.maximum(matched_data, 0)

    robust_max = np.percentile(reference_data, 99.9)
    matched_data = np.minimum(matched_data, robust_max)
    
    matched_data[~src_mask] = 0

    return matched_data

def main():
    # Paths
    BASE_DIR = "/space/local/cug/nnUNet_raw/Dataset006_LundClone"
    REAL_DIR = os.path.join(BASE_DIR, "imagesTr")
    SYNTH_DIR = os.path.join(BASE_DIR, "synth_images")
    OUT_DIR = os.path.join(BASE_DIR, "synth_images_matched")
    
    os.makedirs(OUT_DIR, exist_ok=True)
    
    synth_files = glob(os.path.join(SYNTH_DIR, "*.nii.gz"))
    print(f"Found {len(synth_files)} synthetic images to normalize.")
    
    for sf in tqdm(synth_files, desc="Linear Normalization"):
        base_name = os.path.basename(sf).replace("_synth_0000.nii.gz", "_0000.nii.gz")
        real_ref = os.path.join(REAL_DIR, base_name)
        
        if os.path.exists(real_ref):
            # Load
            src_nii = nib.load(sf)
            ref_nii = nib.load(real_ref)
            
            src_data = src_nii.get_fdata().astype(np.float32)
            ref_data = ref_nii.get_fdata().astype(np.float32)

            # Match
            matched_data = linear_match_intensities(src_data, ref_data)

            # Save
            out_f = os.path.join(OUT_DIR, os.path.basename(sf))
            new_nii = nib.Nifti1Image(matched_data, src_nii.affine, src_nii.header)
            nib.save(new_nii, out_f)
        else:
            print(f"Warning: Reference real image not found for {base_name}")

if __name__ == "__main__":
    main()