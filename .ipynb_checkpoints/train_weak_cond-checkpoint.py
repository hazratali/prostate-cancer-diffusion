import argparse, os, json, random, pathlib, time, glob, functools
import numpy as np
import nibabel as nib
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize 


def set_seed(s):
    """Set random seeds for reproducibility."""
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

def ensure_dir(p):
    """Ensure a directory exists."""
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

def _robust_norm(x: np.ndarray):
    """Clip to robust range (0.5-99.5 percentile) then z-score."""
    lo, hi = np.percentile(x, [0.5, 99.5])
    x = np.clip(x, lo, hi)
    m, s = x.mean(), x.std() + 1e-8
    return (x - m) / s

def norm_slice(x):
    """Normalize a 2D slice to [0, 1] using 1-99th percentile."""
    p1, p99 = np.percentile(x, (1, 99))
    if p99 <= p1:
        mn, mx = float(x.min()), float(x.max())
    else:
        mn, mx = p1, p99
    return np.clip((x - mn) / (mx - mn + 1e-6), 0, 1)

def js_divergence(a, b, bins=64):
    """Calculate Jensen-Shannon divergence between two 0-1 normalized images."""
    ha, _ = np.histogram(a.ravel(), bins=bins, range=(0, 1), density=True)
    hb, _ = np.histogram(b.ravel(), bins=bins, range=(0, 1), density=True)
    ha /= ha.sum() + 1e-8
    hb /= hb.sum() + 1e-8
    m = 0.5 * (ha + hb)
    kl = lambda p, q: (p * np.log((p + 1e-8) / (q + 1e-8))).sum()
    return 0.5 * kl(ha, m) + 0.5 * kl(hb, m)

def _center_crop_to_match(a, b):
    """Center crop the larger image to match the smaller one's spatial dimensions."""
    Ha, Wa = a.shape[-2], a.shape[-1]
    Hb, Wb = b.shape[-2], b.shape[-1]
    H, W = min(Ha, Hb), min(Wa, Wb)
    def cc(x):
        sy, sx = max((x.shape[-2] - H) // 2, 0), max((x.shape[-1] - W) // 2, 0)
        return x[..., sy : sy + H, sx : sx + W]
    return cc(a), cc(b)

def _to_tensors_from_batch(b, device):
    if isinstance(b, dict):
        cond = b["mask"].to(device)
        target = b["img"].to(device)
    else:
        raise TypeError(f"Unexpected batch type/shape: {type(b)}")
    return cond, target


def load_vol(img_path: str, lab_path: str):
    img_nii = nib.load(img_path)
    lab_nii = nib.load(lab_path)
    img = img_nii.get_fdata().astype(np.float32)
    lab = lab_nii.get_fdata().astype(np.float32)
    if img.ndim == 4: img = img[..., 0]
    img = np.transpose(img, (2, 1, 0)) 
    lab = np.transpose(lab, (2, 1, 0)) 
    img = _robust_norm(img)
    lab = (lab > 0).astype(np.float32)
    return img, lab, None

class CaseDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        img_p, lab_p = self.pairs[idx]
        vol_img, vol_lab, _ = load_vol(img_p, lab_p)
        return vol_img, vol_lab


def make_k_slab(vol_img, vol_lab, k):
    """
    Given full volumes [S,H,W], choose a random center slice z0,
    build a [k+1,H,W] condition (k-mask + 1-low-res-img),
    and return the [1,H,W] full-res target.
    """
    S, H, W = vol_img.shape
    z0 = np.random.randint(0, S)
    half = k // 2
    zs = [np.clip(z0 + d, 0, S - 1) for d in range(-half, half + 1)]
    slab_mask = (vol_lab[zs, ...] > 0).astype(np.float32)
    target_img = norm_slice(vol_img[z0, ...]).astype(np.float32)
    low_res_shape = (H // 8, W // 8)
    low_res_img = resize(target_img, low_res_shape, anti_aliasing=True)
    low_res_img = resize(low_res_img, (H, W), order=0, anti_aliasing=False, preserve_range=True) 
    
    cond = np.concatenate([
        slab_mask,                       
        low_res_img[None, ...]           
    ], axis=0).astype(np.float32)
    
    return cond, target_img[None, ...] 

def collate_fn(batch, k):
    conds = []
    targets = []
    for vol_img, vol_lab in batch:
        c, t = make_k_slab(vol_img, vol_lab, k)
        conds.append(torch.from_numpy(c))
        targets.append(torch.from_numpy(t))
    cond = torch.stack(conds, 0)   
    tgt = torch.stack(targets, 0)  
    return {"mask": cond, "img": tgt}

def build_loaders(images_dir, labels_dir, k, batch, n_workers, seed):
    rng = random.Random(seed)
    imgs_modality = glob.glob(os.path.join(images_dir, "*_0000.nii.gz"))
    imgs_plain = glob.glob(os.path.join(images_dir, "*.nii.gz"))
    imgs = sorted(list(set(imgs_modality + imgs_plain)))
    labs = sorted(glob.glob(os.path.join(labels_dir, "**", "*.nii.gz"), recursive=True))

    img_stems = {}
    for p in imgs:
        fname = os.path.basename(p)
        if fname.endswith("_0000.nii.gz"):
            stem = fname.replace("_0000.nii.gz", "")
        else:
            stem = fname.replace(".nii.gz", "")
        img_stems[stem] = p
    lab_stems = {os.path.basename(p).replace(".nii.gz", ""): p for p in labs}
    
    common = sorted(img_stems.keys() & lab_stems.keys())
    print(f"[pairing] images={len(imgs)} labels={len(labs)} paired={len(common)}")
    if not common:
        raise RuntimeError("No paired cases found.")
    pairs = [(img_stems[s], lab_stems[s]) for s in common]
    rng.shuffle(pairs)

    n = len(pairs)
    n_val = max(1, int(0.1 * n))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    tr_ds = CaseDataset(train_pairs)
    va_ds = CaseDataset(val_pairs)
    print(f"[dataset] train_cases={len(tr_ds)} val_cases={len(va_ds)} k={k} batch={batch}")
    g = torch.Generator(); g.manual_seed(seed)
    collate_with_k = functools.partial(collate_fn, k=k)
    dl_tr = DataLoader(
        tr_ds, batch_size=batch, shuffle=True, num_workers=n_workers,
        pin_memory=True, drop_last=True, generator=g, collate_fn=collate_with_k
    )
    dl_va = DataLoader(
        va_ds, batch_size=batch, shuffle=False, num_workers=n_workers,
        pin_memory=True, drop_last=False, collate_fn=collate_with_k
    )
    return dl_tr, dl_va


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

    def q_sample(self, x0, t, noise=None):
        if noise is None: noise = torch.randn_like(x0)
        ac = self.alphas_cum[t].view(-1, 1, 1, 1)
        return (ac.sqrt() * x0) + (1 - ac).sqrt() * noise, noise

    def p_losses(self, cond, x0, t):
        x_t, noise = self.q_sample(x0, t)  
        eps = self.model(torch.cat([cond, x_t], dim=1))
        return F.mse_loss(eps, noise)


    @torch.no_grad()
    def sample(self, cond, steps=50):
        b, _, h, w = cond.shape 
        device = cond.device
        x = torch.randn(b, 1, h, w, device=device)
        ts = torch.linspace(self.T - 1, 0, steps, dtype=torch.long, device=device)

        for ti in ts:
            t = torch.full((b,), int(ti), device=device, dtype=torch.long)
            eps = self.model(torch.cat([cond, x], dim=1))
            beta = self.betas[t].view(-1,1,1,1)
            alpha_c = self.alphas_cum[t].view(-1,1,1,1)
            mean = (x - (1 - alpha_c).sqrt() * eps) / alpha_c.sqrt()
            x = mean + (beta.sqrt() * torch.randn_like(x) if ti > 0 else 0)
            x = x.clamp(0, 1) 
        return x


class EMA:
    def __init__(self, net, decay=0.9999):
        self.shadow = {k: v.detach().clone() for k, v in net.state_dict().items()}
        self.decay = decay
    @torch.no_grad()
    def update(self, net):
        for k, v in net.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)
    @torch.no_grad()
    def copy_to(self, net):
        net.load_state_dict(self.shadow, strict=True)


@torch.no_grad()
def quick_eval(diff_model, val_loader, device):
    ssim_l, js_l = [], []
    for b in val_loader:
        cond, target = _to_tensors_from_batch(b, device)
        pred = diff_model.sample(cond, steps=25)
        
        for i in range(pred.size(0)):
            p = pred[i, 0].cpu().numpy()
            r = target[i, 0].cpu().numpy()
            p, r = _center_crop_to_match(p, r)
            ssim_l.append(ssim(p, r, data_range=1.0))
            js_l.append(js_divergence(p, r))
        
        if len(ssim_l) >= 32:
            break
            
    return float(np.mean(ssim_l)), float(np.mean(js_l))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--base", type=int, default=24)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=8)
    args = ap.parse_args()

    set_seed(args.seed)
    ensure_dir(args.out)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        print("Enabling TF32 matmul precision")
        torch.set_float32_matmul_precision('high')

    dl_tr, dl_va = build_loaders(
        args.images, args.labels, args.k, args.batch, args.num_workers, args.seed
    )


    net = UNet2D(in_ch=args.k + 2, base=args.base, out_ch=1, grad_ckpt=args.grad_ckpt).to(device)
    diff = Diffusion(net, args.timesteps).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    ema = EMA(net)
    scaler = GradScaler(device=device, enabled=(device == "cuda"))

    best_score = -1e9
    last_improve = 0
    it = 0
    
    print(f"Starting training on {device.upper()}...")
    
    for ep in range(args.epochs):
        net.train()
        pbar = tqdm(dl_tr, desc=f"Epoch {ep+1}/{args.epochs}")
        
        for b in pbar:
            it += 1
            cond, target = _to_tensors_from_batch(b, device)
            t = torch.randint(0, diff.T, (cond.size(0),), device=device)

            with autocast(device, enabled=(device == "cuda")):
                loss = diff.p_losses(cond, target, t)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            ema.update(net)
            
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if it % args.eval_every == 0:
                net.eval()
                ema.copy_to(net) 
                ssim_v, js_v = quick_eval(diff, dl_va, device)
                score = ssim_v - js_v
                net.train() 

                print(f"[eval] it={it} ssim={ssim_v:.4f} js={js_v:.4f} score={score:.4f}")
                with open(os.path.join(args.out, "train_log.jsonl"), "a") as f:
                    f.write(json.dumps({"it": it, "ssim": ssim_v, "js": js_v, "score": score}) + "\n")

                if score > best_score:
                    best_score = score
                    last_improve = it
                    print("  ** New best score! Saving model... **")
                    torch.save(
                        {"net": net.state_dict(), "ema": ema.shadow, "it": it, "args": vars(args)},
                        os.path.join(args.out, "best.pt")
                    )
                    with open(os.path.join(args.out, "best_metrics.json"), "w") as f:
                        json.dump({"it": it, "ssim": ssim_v, "js": js_v, "score": score}, f, indent=2)
                
                if (it - last_improve) >= args.patience * args.eval_every:
                    print(f"Early stopping: No improvement in {args.patience * args.eval_every} iterations.")
                    return

    print("Training complete.")

if __name__ == "__main__":
    main()