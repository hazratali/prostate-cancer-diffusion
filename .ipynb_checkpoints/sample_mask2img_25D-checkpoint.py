#!/usr/bin/env python3
import argparse, os, math, json, random, shutil
from glob import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib


def ensure_dir(d): os.makedirs(d, exist_ok=True)

def list_pairs(labels_dir, images_dir=None):
    labs = sorted(glob(os.path.join(labels_dir, "case_*.nii.gz")))
    if images_dir is None: return [(l, None) for l in labs]
    pairs = []
    for l in labs:
        stem = os.path.basename(l).replace(".nii.gz","")
        img = os.path.join(images_dir, f"{stem}_0000.nii.gz")
        if os.path.isfile(img): pairs.append((l, img))
    return pairs

def load_vol(path):  # returns np array (z,y,x), affine, header
    nii = nib.load(path); arr = np.asarray(nii.dataobj)
    if arr.ndim == 4: arr = arr.squeeze()
    return arr, nii.affine, nii.header

def save_nii(arr, affine, header, out_path):
    nib.save(nib.Nifti1Image(arr.astype(np.float32), affine, header), out_path)

def vol_to_patches(vol, k=3):
    z, y, x = vol.shape
    pad = k//2
    volp = np.pad(vol, ((pad,pad),(0,0),(0,0)), mode='edge')
    patches = []
    for i in range(z):
        patches.append(volp[i:i+k])
    return patches  # list of (k,Y,X)

def reassemble_slices(slices):  # list of (Y,X) -> (Z,Y,X)
    return np.stack(slices, axis=0)

# -------------------- building blocks --------------------
class ConvBNAct(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.SiLU(),
            nn.Conv2d(c_out, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.SiLU(),
        )
    def forward(self, x): return self.net(x)

class Down(nn.Module):
    def __init__(self): super().__init__(); self.pool = nn.MaxPool2d(2)
    def forward(self, x): return self.pool(x)

class Up(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.up = nn.ConvTranspose2d(c_in, c_in//2, 2, stride=2)
        self.conv = ConvBNAct(c_in, c_out)
    def forward(self, x, skip):
        x = self.up(x)
        dh = skip.shape[-2] - x.shape[-2]
        dw = skip.shape[-1] - x.shape[-1]
        if dh != 0 or dw != 0:
            x = F.pad(x, (0, max(0, dw), 0, max(0, dh)))
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

# -------------------- 3 UNet variants--------------------
class UNet25DTiny(nn.Module):
    # no downsampling, just encoder 
    def __init__(self, in_ch=4, base_ch=32, out_ch=1):
        super().__init__()
        self.enc1 = ConvBNAct(in_ch, base_ch)
        self.head = nn.Conv2d(base_ch, out_ch, 1)
    def forward(self, x):
        x = self.enc1(x)
        return self.head(x)

class UNet25DLite(nn.Module):
    # 2 downs + 2 ups
    def __init__(self, in_ch=4, base_ch=32, out_ch=1):
        super().__init__()
        c1, c2, c3 = base_ch, base_ch*2, base_ch*4
        self.enc1 = ConvBNAct(in_ch, c1);  self.down1 = Down()
        self.enc2 = ConvBNAct(c1, c2);     self.down2 = Down()
        self.bott = ConvBNAct(c2, c3)
        self.up2  = Up(c3, c2)
        self.up1  = Up(c2, c1)
        self.head = nn.Conv2d(c1, out_ch, 1)
    def forward(self, x):
        e1 = self.enc1(x); x = self.down1(e1)
        e2 = self.enc2(x); x = self.down2(e2)
        x  = self.bott(x)
        x  = self.up2(x, e2)
        x  = self.up1(x, e1)
        return self.head(x)

class UNet25DDeep(nn.Module):
    # 3 downs + bottleneck + 3 ups
    def __init__(self, in_ch=4, base_ch=32, out_ch=1):
        super().__init__()
        c1, c2, c3, c4 = base_ch, base_ch*2, base_ch*4, base_ch*8
        self.enc1 = ConvBNAct(in_ch, c1);  self.down1 = Down()
        self.enc2 = ConvBNAct(c1, c2);     self.down2 = Down()
        self.enc3 = ConvBNAct(c2, c3);     self.down3 = Down()
        self.bott = ConvBNAct(c3, c4)
        self.up3  = Up(c4, c3)
        self.up2  = Up(c3, c2)
        self.up1  = Up(c2, c1)
        self.head = nn.Conv2d(c1, out_ch, 1)
    def forward(self, x):
        e1 = self.enc1(x); x = self.down1(e1)
        e2 = self.enc2(x); x = self.down2(e2)
        e3 = self.enc3(x); x = self.down3(e3)
        x  = self.bott(x)
        x  = self.up3(x, e3)
        x  = self.up2(x, e2)
        x  = self.up1(x, e1)
        return self.head(x)

# -------------------- diffusion --------------------
class Diffusion:
    def __init__(self, timesteps=1000, device="cuda"):
        self.device = device
        self.timesteps = timesteps
        self.betas = self._cosine_beta_schedule(timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def _cosine_beta_schedule(self, T, s=0.008):
        steps = T + 1
        x = torch.linspace(0, T, steps)
        alphas_cumprod = torch.cos(((x/T) + s) / (1+s) * math.pi/2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 1e-5, 0.999)

    @torch.no_grad()
    def sample(self, net, cond, shape):
        device = self.device
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.timesteps)):
            x_in = torch.cat([x, cond], dim=1)  # concat cond mask channel
            eps = net(x_in)
            beta_t = self.betas[t].view(1,1,1,1)
            alpha_t = self.alphas[t].view(1,1,1,1)
            alpha_cum = self.alphas_cumprod[t].view(1,1,1,1)
            mean = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_cum)) * eps)
            if t > 0:
                x = mean + torch.sqrt(beta_t) * torch.randn_like(x)
            else:
                x = mean
        return x

# -------------------- checkpoint loading --------------------
def infer_in_ch_and_depth(sd):
    """Heuristics: infer input channels (k+1) and choose model depth based on keys present."""
    # infer in_ch from first conv weight we can find
    key_4d = None
    for k, v in sd.items():
        if isinstance(v, torch.Tensor) and v.ndim == 4:
            key_4d = k; break
    in_ch = int(sd[key_4d].shape[1]) if key_4d else 4

    keys = set(sd.keys())
    has_bott = any(k.startswith("bott.") for k in keys)
    has_up3  = any(k.startswith("up3.") for k in keys)
    has_enc3 = any(k.startswith("enc3.") for k in keys)
    has_enc2 = any(k.startswith("enc2.") for k in keys)

    if has_bott and has_up3 and has_enc3:
        depth = "deep"
    elif has_bott and has_enc2:
        depth = "lite"
    else:
        depth = "tiny"
    return in_ch, depth

def maybe_remap_head(sd):
    """Map 'out.*' -> 'head.*' if needed."""
    if any(k.startswith("out.") for k in sd.keys()):
        remapped = {}
        for k, v in sd.items():
            if k.startswith("out."):
                remapped["head." + k[len("out."):]] = v
            else:
                remapped[k] = v
        return remapped
    return sd

def load_model_from_ckpt(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    # pick the most likely dict
    for k in ["ema", "state_dict_ema", "model", "state_dict"]:
        if k in ckpt and isinstance(ckpt[k], dict):
            sd = ckpt[k] if k != "ema" else ckpt["ema"].get("state_dict", ckpt["ema"])
            break
    else:
        sd = ckpt  # assume raw state_dict

    sd = maybe_remap_head(sd)
    in_ch, depth = infer_in_ch_and_depth(sd)
    base_ch = int(next(v for v in sd.values() if isinstance(v, torch.Tensor) and v.ndim==4).shape[0])

    if depth == "deep":
        net = UNet25DDeep(in_ch=in_ch, base_ch=base_ch, out_ch=1).to(device)
    elif depth == "lite":
        net = UNet25DLite(in_ch=in_ch, base_ch=base_ch, out_ch=1).to(device)
    else:
        net = UNet25DTiny(in_ch=in_ch, base_ch=base_ch, out_ch=1).to(device)

    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing:
        print(f"[warn] missing keys when loading ({len(missing)}), e.g.: {missing[:5]}")
    if unexpected:
        print(f"[warn] unexpected keys in checkpoint ({len(unexpected)}), e.g.: {unexpected[:5]}")
    net.eval()
    return net, in_ch

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="labelsTr folder (gt masks)")
    ap.add_argument("--ref", required=False, help="imagesTr folder (optional, for pairing)")
    ap.add_argument("--ckpt", required=True, help="path to best.pt/last.pt")
    ap.add_argument("--out", required=True, help="output root")
    ap.add_argument("--num_vols", type=int, default=20)
    ap.add_argument("--k", type=int, default=3, help="2.5D stack depth used in training (neighbors+center= in_ch, so in_ch-1==k)")
    ap.add_argument("--timesteps", type=int, default=250, help="sampling steps")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    out_img = os.path.join(args.out, "imagesTr"); ensure_dir(out_img)
    out_lab = os.path.join(args.out, "labelsTr"); ensure_dir(out_lab)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net, in_ch = load_model_from_ckpt(args.ckpt, device)
    k_detect = in_ch - 1
    if k_detect != args.k:
        print(f"[warn] checkpoint suggests k={k_detect}; using k={k_detect}")
    args.k = k_detect

    pairs = list_pairs(args.labels, args.ref)
    if len(pairs) == 0:
        raise RuntimeError("No masks found in labelsTr.")
    random.shuffle(pairs)
    pairs = pairs[:args.num_vols]

    diff = Diffusion(timesteps=args.timesteps, device=device)

    meta = []
    for (lab_p, ref_p) in pairs:
        stem = os.path.basename(lab_p).replace(".nii.gz","")
        print(f"Sampling {stem} ...")

        lab, aff, hdr = load_vol(lab_p)     # (Z,Y,X)
        patches = vol_to_patches(lab, k=args.k)
        synth_slices = []
        for stack in patches:
            # stack is (k, H, W) from vol_to_patches(...)
            cond_np = stack.astype(np.float32)                  # (k,H,W)
            cond_t  = torch.from_numpy(cond_np[None, ...]).to(device)   # (1,k,H,W)

            # sample 1-channel image; concatenation inside diffusion will be [1 + k, H, W]
            img_shape = (cond_t.shape[0], 1, cond_t.shape[-2], cond_t.shape[-1])  # (1,1,H,W)
            img_t = diff.sample(net, cond_t, shape=img_shape)                      # (1,1,H,W)

            synth_slices.append(img_t.squeeze().detach().cpu().numpy())

        vol = reassemble_slices(synth_slices)  # (Z,Y,X)
        # robust rescale to [0,1]
        vmin, vmax = np.percentile(vol, (1, 99))
        vol = np.clip((vol - vmin) / max(1e-6, (vmax - vmin)), 0, 1)

        # save image and copy mask
        save_nii(vol, aff, hdr, os.path.join(out_img, f"{stem}_0000.nii.gz"))
        shutil.copy2(lab_p, os.path.join(out_lab, f"{stem}.nii.gz"))
        meta.append({"case": stem, "label": lab_p, "ref": ref_p})

    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Done. Wrote {len(pairs)} volumes to {args.out}")

if __name__ == "__main__":
    main()
