import argparse, os, pathlib, torch, nibabel as nib, numpy as np
from tqdm import tqdm
from glob import glob
from skimage.transform import resize

# IMPORT TRAINING SCRIPT
try:
    from train_weak_cond import UNet2D, Diffusion, norm_slice
except ImportError:
    print("Error: train_weak_cond.py not found in the current directory.")
    exit()

def generate_synthetic_volume(diff_model, vol_lab, vol_img, k, device):
    """
    vol_lab: [S, H, W] - The Mask
    vol_img: [S, H, W] - The Real Image (used for low-res hint)
    """
    S, H, W = vol_lab.shape
    diff_model.eval()
    synth_vol = np.zeros_like(vol_lab, dtype=np.float32)
    half = k // 2

    for z in tqdm(range(S), desc="Slices", leave=False):
        # 1. Prepare Mask Slab (k slices)
        zs = [np.clip(z + d, 0, S - 1) for d in range(-half, half + 1)]
        slab_mask = (vol_lab[zs, ...] > 0).astype(np.float32)
        
        # 2. Prepare Low-Res Condition
        real_slice = vol_img[z, ...]
        real_slice_norm = norm_slice(real_slice) 
        
        # Resize often returns float64 
        low_res_shape = (H // 8, W // 8)
        low_res_small = resize(real_slice_norm, low_res_shape, anti_aliasing=True)
        low_res_upsampled = resize(low_res_small, (H, W), order=0, anti_aliasing=False, preserve_range=True)
        
        # 3. Concatenate (Mask + LowRes)
        cond = np.concatenate([slab_mask, low_res_upsampled[None, ...]], axis=0)
        
        # FIX IS HERE: .float() forces conversion from Double to Float32
        cond_tensor = torch.from_numpy(cond).float().unsqueeze(0).to(device)

        # 4. Generate
        with torch.no_grad():
            pred = diff_model.sample(cond_tensor, steps=50)
        synth_vol[z, ...] = pred[0, 0].cpu().numpy()
        
    return synth_vol

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mask_dir", required=True)
    ap.add_argument("--image_dir", required=True, help="Path to real images for low-res conditioning")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.ckpt, map_location=device)
    train_args = checkpoint['args']
    
    # Init Model
    net = UNet2D(in_ch=train_args['k'] + 2, base=train_args['base'], out_ch=1).to(device)
    diff = Diffusion(net, train_args['timesteps']).to(device)
    
    if 'ema' in checkpoint:
        net.load_state_dict(checkpoint['ema'])
        print("Loaded EMA weights.")
    else:
        net.load_state_dict(checkpoint['net'])

    mask_files = sorted(glob(os.path.join(args.mask_dir, "*.nii.gz")))
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Generating with Low-Res Conditioning from: {args.image_dir}")

    for mf in tqdm(mask_files, desc="Processing Volumes"):
        name = os.path.basename(mf)
        # Find corresponding real image
        img_name = name.replace(".nii.gz", "_0000.nii.gz")
        img_path = os.path.join(args.image_dir, img_name)
        
        if not os.path.exists(img_path):
            print(f"Skipping {name}, real image not found at {img_path}")
            continue

        # Load Data
        m_nii = nib.load(mf)
        i_nii = nib.load(img_path)
        
        m_data = m_nii.get_fdata().astype(np.float32)
        i_data = i_nii.get_fdata().astype(np.float32)
        
        # Orient to (Slices, H, W)
        m_data = np.transpose(m_data, (2, 1, 0))
        i_data = np.transpose(i_data, (2, 1, 0))
        
        # Generate
        synth_data = generate_synthetic_volume(diff, m_data, i_data, train_args['k'], device)
        
        # Orient back
        synth_data = np.transpose(synth_data, (2, 1, 0))
        
        out_name = name.replace(".nii.gz", "_synth_0000.nii.gz")
        nib.save(nib.Nifti1Image(synth_data, m_nii.affine, m_nii.header), 
                 os.path.join(args.out_dir, out_name))

if __name__ == "__main__":
    main()