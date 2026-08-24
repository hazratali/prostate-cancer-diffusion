#!/usr/bin/env python3
# filter_best_samples.py
#
# Selects the best synthetic samples based on SSIM score against the real image.


import argparse, os, glob, shutil
import numpy as np
import nibabel as nib
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

def norm_minmax(x):
    """Normalize to [0, 1] for SSIM calculation"""
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-8: return x
    return (x - mn) / (mx - mn)

def find_real_image_path(stem, real_img_dir):
    # find matching real file
    imgs_modality = glob.glob(os.path.join(real_img_dir, f"{stem}_0000.nii.gz"))
    imgs_plain = glob.glob(os.path.join(real_img_dir, f"{stem}.nii.gz"))
    all_imgs = sorted(list(set(imgs_modality + imgs_plain)))
    if all_imgs: return all_imgs[0]
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth_dir", required=True, help="Folder with Histogram Matched synthetic images")
    ap.add_argument("--real_dir", required=True, help="Folder with Real images (imagesTr)")
    ap.add_argument("--labels_dir", required=True, help="Folder with Real Labels (labelsTr)")
    ap.add_argument("--output_dir", required=True, help="New folder to save the FINAL filtered dataset")
    ap.add_argument("--keep_ratio", type=float, default=0.5, help="Ratio to keep (0.5 = Top 50%)")
    args = ap.parse_args()

    # Setup output folders
    final_img_dir = os.path.join(args.output_dir, "imagesTr")
    final_lab_dir = os.path.join(args.output_dir, "labelsTr")
    rejected_dir = os.path.join(args.output_dir, "rejected")
    
    os.makedirs(final_img_dir, exist_ok=True)
    os.makedirs(final_lab_dir, exist_ok=True)
    os.makedirs(rejected_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.synth_dir, "*.nii.gz")))
    scores = []

    print(f"Scoring {len(files)} synthetic volumes...")

    for synth_path in tqdm(files):
        fname = os.path.basename(synth_path)
        stem = fname.split("_synth")[0] # e.g. case_007
        
        real_path = find_real_image_path(stem, args.real_dir)
        if not real_path: continue

        # Load volumes
        try:
            s_vol = nib.load(synth_path).get_fdata().astype(np.float32)
            r_vol = nib.load(real_path).get_fdata().astype(np.float32)
        except:
            print(f"Error loading {fname}, skipping.")
            continue
        
        # Calculate SSIM on the middle slice (Z-axis is usually last)
        
        mid = s_vol.shape[2] // 2
        
        s_slice = norm_minmax(s_vol[..., mid])
        r_slice = norm_minmax(r_vol[..., mid])
        
        score = ssim(s_slice, r_slice, data_range=1.0)
        scores.append({
            "path": synth_path,
            "real_path": real_path, 
            "stem": stem,
            "score": score
        })

    # Sort by Score (High = Better)
    scores.sort(key=lambda x: x["score"], reverse=True)

    # Split
    cutoff = int(len(scores) * args.keep_ratio)
    keep = scores[:cutoff]
    reject = scores[cutoff:]

    print(f"\n--- Filtering Results ---")
    print(f"Total Scored: {len(scores)}")
    print(f"Keeping Top {int(args.keep_ratio*100)}%: {len(keep)} files")
    print(f"Best SSIM: {keep[0]['score']:.4f}")
    print(f"Lowest Kept SSIM: {keep[-1]['score']:.4f}")

    # --- Copy Files ---
    print("Copying 'Good' files to final dataset...")
    for item in tqdm(keep):
        # Copy Image
        dest_name = f"{item['stem']}_synth.nii.gz" # Clean name for nnU-Net
        shutil.copy(item['path'], os.path.join(final_img_dir, dest_name))
        
        # Copy Label
   
        lab_name = f"{item['stem']}.nii.gz"
        src_lab = os.path.join(args.labels_dir, lab_name)
        if os.path.exists(src_lab):
            shutil.copy(src_lab, os.path.join(final_lab_dir, lab_name))
        else:
            print(f"Warning: Label not found for {item['stem']}")

    print("Moving 'Bad' files to rejected folder...")
    for item in reject:
        fname = os.path.basename(item['path'])
        shutil.copy(item['path'], os.path.join(rejected_dir, fname))

    print(f"\nDone! Filtered dataset is ready at: {args.output_dir}")

if __name__ == "__main__":
    main()