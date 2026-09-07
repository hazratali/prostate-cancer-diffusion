#!/usr/bin/env python3

import argparse, os, glob, pathlib, random
import numpy as np
import nibabel as nib
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.transform import resize # <-- New dependency

# ------------------ Model Architecture ------------------


def conv_bn_gn(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1, bias=False), nn.GroupNorm(8, o), nn.SiLU(),
        nn.Conv2d(o, o, 3, padding=1, bias=False), nn.GroupNorm(8, o), nn.SiLU()
    )
class Up(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.up = nn.ConvTranspose2d(i, o, 2, 2)
        self.conv = conv_bn_gn(i, o)
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], 1))
class UNet2D(nn.Module):
    def __init__(self, in_ch, base=24, out_ch=1, grad_ckpt=False):
        super().__init__()
        self.grad_ckpt = grad_ckpt        
        self.e1 = conv_bn_gn(in_ch, base)
        self.e2 = nn.Sequential(nn.MaxPool2d(2), conv_bn_gn(base, base * 2))
        self.bott = nn.Sequential(nn.MaxPool2d(2), conv_bn_gn(base * 2, base * 4))
        self.u2 = Up(base * 4, base * 2)
        self.u1 = Up(base * 2, base)
        self.head = nn.Conv2d(base, out_ch, 1)
    def forward(self, x):
        s1 = self.e1(x); s2 = self.e2(s1); b = self.bott(s2)
        u2 = self.u2(b, s2); u1 = self.u1(u2, s1); return self.head(u1)

# ------------------ Diffusion Logic ------------------
# (Must exactly match the training script)

class Diffusion(nn.Module):
    def __init__(self, net, timesteps=1000):
        super().__init__()
        self.model = net
        self.T = timesteps
        b = torch.linspace(1e-4, 0.02, timesteps)
        a = 1 - b
        ac = torch.cumprod(a, 0)
        self.register_buffer("betas", b)
        self.register_buffer("alphas_cum", ac)
    
    @torch.no_grad()
    def sample(self, cond, steps=50):
        b, _, h, w = cond.shape # cond is [B, k+1, H, W]
        device = cond.device
        x = torch.randn(b, 1, h, w, device=device)
        ts = torch.linspace(self.T - 1, 0, steps, dtype=torch.long, device=device)
        for ti in ts:
            t = torch.full((b,), int(ti), device=device, dtype=torch.long)
            eps = self.model(torch.cat([cond, x], dim=1)) # No t_map
            beta = self.betas[t].view(-1,1,1,1)
            alpha_c = self.alphas_cum[t].view(-1,1,1,1)
            mean = (x - (1 - alpha_c).sqrt() * eps) / alpha_c.sqrt()
            x = mean + (beta.sqrt() * torch.randn_like(x) if ti > 0 else 0)
            x = x.clamp(0, 1)
        return x

# ------------------ Helper Functions ------------------

def norm_slice(x):
    """Normalize a 2D slice to [0, 1] using 1-99th percentile."""
    p1, p99 = np.percentile(x, (1, 99))
    if p99 <= p1: mn, mx = float(x.min()), float(x.max())
    else: mn, mx = p1, p99
    return np.clip((x - mn) / (mx - mn + 1e-6), 0, 1)

def find_real_image_path(stem, real_img_dir):
    """Finds the matching real image file for a given stem."""
    # Find all .nii.gz files, handling _0000 modality or plain files
    imgs_modality = glob.glob(os.path.join(real_img_dir, f"{stem}_0000.nii.gz"))
    imgs_plain = glob.glob(os.path.join(real_img_dir, f"{stem}.nii.gz"))
    all_imgs = sorted(list(set(imgs_modality + imgs_plain)))
    if all_imgs:
        return all_imgs[0]
    return None

# ------------------ 3D Generation Logic ------------------

@torch.no_grad()
def generate_3d_volume(diff_model, lab_vol_raw, real_vol_raw, k, device, steps):
    """
    Generates a 3D synthetic image from a 3D label and real volume.
    """
    D, H, W = lab_vol_raw.shape
    half = k // 2
    
    lab_vol = (lab_vol_raw > 0).astype(np.float32)
    synth_vol = np.zeros((D, H, W), dtype=np.float32)

    for z in tqdm(range(D), leave=False, desc="Slices"):
        # 1. Create the k-slice mask slab [k, H, W]
        zs = [np.clip(z + d, 0, D - 1) for d in range(-half, half + 1)]
        slab_mask = lab_vol[zs, ...]
        
        # 2. Create the low-res image [1, H, W]
        real_slice_norm = norm_slice(real_vol_raw[z, ...])
        low_res_shape = (H // 8, W // 8)
        low_res_img = resize(real_slice_norm, low_res_shape, anti_aliasing=True)
        low_res_img = resize(low_res_img, (H, W), order=0, anti_aliasing=False, preserve_range=True)[None, ...]
        
        # 3. Create the final [k+1, H, W] condition
        cond_np = np.concatenate([slab_mask, low_res_img], axis=0).astype(np.float32)
        
        # 4. Convert to tensor [1, k+1, H, W]
        cond = torch.from_numpy(cond_np).float().to(device).unsqueeze(0)
        
        # 5. Sample from diffusion model
        img_slice_tensor = diff_model.sample(cond, steps=steps) # [1, 1, H, W]
        
        # 6. Convert back to numpy and place in canvas
        img_slice_np = img_slice_tensor.squeeze().cpu().numpy() # [H, W]
        synth_vol[z, ...] = img_slice_np
        
    return synth_vol

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to 'best.pt' model checkpoint.")
    ap.add_argument("--labels_dir", required=True, help="Directory of 3D NIfTI label files (e.g., labelsTr).")
    ap.add_argument("--real_dir", required=True, help="Directory of REAL 3D NIfTI images (e.g., imagesTr).") # <-- NEW
    ap.add_argument("--output_dir", required=True, help="Directory to save synthetic 3D images.")
    ap.add_argument("--num_vols", type=int, default=None, help="Generate only this many total volumes (randomly selected).")
    ap.add_argument("--steps", type=int, default=50, help="Number of diffusion sampling steps.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")
    
    ckpt = torch.load(args.model, map_location=device, weights_only=True)
    if 'args' not in ckpt:
        raise ValueError("Checkpoint 'best.pt' does not contain 'args'.")
        
    model_args = ckpt['args']
    k = model_args['k']
    base = model_args['base']
    timesteps = model_args['timesteps']
    print(f"Loaded model params: k={k}, base={base}, timesteps={timesteps}")

    net = UNet2D(in_ch=k + 2, base=base, out_ch=1).to(device)
    diff = Diffusion(net, timesteps=timesteps).to(device)
    
    ema_weights = ckpt['ema']
    clean_weights = {}
    for k_key, v in ema_weights.items():
        clean_key = k_key[10:] if k_key.startswith("_orig_mod.") else k_key
        clean_weights[clean_key] = v
    net.load_state_dict(clean_weights, strict=True)
    net.eval()
    
    try:
        net = torch.compile(net, mode="reduce-overhead")
        print("Model compiled for inference.")
    except Exception as e:
        print(f"Could not compile model for inference: {e}. Running in eager mode.")
        
    label_files = sorted(glob.glob(os.path.join(args.labels_dir, "**", "*.nii.gz"), recursive=True))
    print(f"Found {len(label_files)} label files.")
    if not label_files:
        raise FileNotFoundError(f"No .nii.gz files found in {args.labels_dir}")
    
    if args.num_vols:
        random.shuffle(label_files)
        label_files = label_files[:args.num_vols]
        print(f"Randomly selected {len(label_files)} files to generate.")
    
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    for lab_path in tqdm(label_files, desc="Generating Volumes"):
        stem = os.path.basename(lab_path).replace(".nii.gz", "")
        
        real_path = find_real_image_path(stem, args.real_dir)
        if not real_path:
            print(f"Warning: No real image found for stem {stem}. Skipping.")
            continue
            
        lab_nii = nib.load(lab_path)
        real_nii = nib.load(real_path)
        
        lab_vol_nifti = lab_nii.get_fdata().astype(np.float32)
        real_vol_nifti = real_nii.get_fdata().astype(np.float32)
        
        lab_vol_raw = np.transpose(lab_vol_nifti, (2, 1, 0))
        real_vol_raw = np.transpose(real_vol_nifti, (2, 1, 0))
        
        synth_vol = generate_3d_volume(diff, lab_vol_raw, real_vol_raw, k, device, args.steps)
        
        synth_vol_nifti_order = np.transpose(synth_vol, (2, 1, 0))
        
        out_name = f"{stem}_synth_weakcond.nii.gz"
        out_nii = nib.Nifti1Image(
            synth_vol_nifti_order, 
            affine=lab_nii.affine, 
            header=lab_nii.header
        )
        out_nii.to_filename(os.path.join(args.output_dir, out_name))

    print(f"\nGeneration complete. {len(label_files)} volumes saved to {args.output_dir}")

if __name__ == "__main__":
    main()