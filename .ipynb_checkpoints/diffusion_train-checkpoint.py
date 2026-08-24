#!/usr/bin/env python3
import os, json, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Runtime / CUDA setup 
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:64")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHANNELS_LAST = True

# paths
ROOT = "/space/local/cug/nnUNet_raw/Dataset001_PROSTATE/slices_axial"
SAVE = "./diffusion_ckpts_prostate"
os.makedirs(SAVE, exist_ok=True)


CROP = 320  

def _crop_center(t: torch.Tensor, size: int = CROP) -> torch.Tensor:
    h, w = t.shape[-2:]
    y0 = max((h - size) // 2, 0)
    x0 = max((w - size) // 2, 0)
    return t[..., y0:y0+size, x0:x0+size]

class SlicePairs(Dataset):
    def __init__(self, root: str):
        self.root = Path(root)
        self.index = json.loads((self.root / "index.json").read_text())

    def __len__(self): return len(self.index)

    def __getitem__(self, i: int):
        d = np.load(self.root / self.index[i])
        img = torch.from_numpy(d["img"]).float() / 127.5 - 1.0   # [-1, 1]
        msk = torch.from_numpy(d["mask"]).float()                # {0,1}
        # center crop to reduce HxW → big memory win
        img = _crop_center(img)
        msk = _crop_center(msk)
        return torch.stack([img, msk], 0)                        # [2,H,W]

train_ds = SlicePairs(f"{ROOT}/train")
val_ds   = SlicePairs(f"{ROOT}/val")

# per-GPU batch kept tiny; gradient accumulation preserves effective batch size
GPU_BATCH   = 1
ACCUM_STEPS = 32

train_dl = DataLoader(
    train_ds, batch_size=GPU_BATCH, shuffle=True,
    num_workers=4, pin_memory=True, persistent_workers=False
)
val_dl = DataLoader(
    val_ds, batch_size=GPU_BATCH, shuffle=False,
    num_workers=2, pin_memory=True, persistent_workers=False
)

# diffusion model
class DoubleConv(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        # Use up to 8 groups
        g = min(8, c_out)
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, 1, 1), nn.GroupNorm(g, c_out), nn.SiLU(),
            nn.Conv2d(c_out, c_out, 3, 1, 1), nn.GroupNorm(g, c_out), nn.SiLU()
        )
    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    """
    3-level UNet to save memory
    """
    def __init__(self, in_ch=2, base=32):  # narrower than 64
        super().__init__()
        ch = [base, base*2, base*4]  # enc1 -> enc2 -> mid

        # encoder
        self.enc1 = DoubleConv(in_ch, ch[0]); self.down1 = nn.Conv2d(ch[0], ch[0], 4, 2, 1)
        self.enc2 = DoubleConv(ch[0], ch[1]); self.down2 = nn.Conv2d(ch[1], ch[1], 4, 2, 1)

        # bottleneck
        self.mid  = DoubleConv(ch[1], ch[2])
        self.mid_channels = ch[2]

        # time embedding projection (128 -> mid channels)
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(128, self.mid_channels))

        # decoder
        self.up2  = nn.ConvTranspose2d(ch[2], ch[1], 4, 2, 1); self.dec2 = DoubleConv(ch[1]*2, ch[1])
        self.up1  = nn.ConvTranspose2d(ch[1], ch[0], 4, 2, 1); self.dec1 = DoubleConv(ch[0]*2, ch[0])
        self.out  = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x, t_emb):
        e1 = self.enc1(x); x = self.down1(e1)
        e2 = self.enc2(x); x = self.down2(e2)

        x = self.mid(x)
        te = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)  # [B, Cmid, 1, 1]
        x = x + te

        x = self.up2(x); x = self.dec2(torch.cat([x, e2], 1))
        x = self.up1(x); x = self.dec1(torch.cat([x, e1], 1))
        return self.out(x)

# DDPM
def cosine_beta_schedule(T, s=0.008):
    steps = np.arange(T+1)
    alphas_cumprod = np.cos(((steps / T) + s) / (1 + s) * math.pi / 2)**2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 1e-6, 0.999)

T = 1000
betas = torch.tensor(cosine_beta_schedule(T), dtype=torch.float32, device=DEVICE)
alphas = 1.0 - betas
a_bar  = torch.cumprod(alphas, dim=0)

def timestep_embed(t, dim=128):  # simple Fourier features
    half = dim // 2
    freqs = torch.exp(torch.arange(half, device=DEVICE) * (-math.log(10000.0) / half))
    args  = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

# build model
net = UNet(in_ch=2, base=32).to(DEVICE)
if CHANNELS_LAST:
    net = net.to(memory_format=torch.channels_last)

opt    = torch.optim.AdamW(net.parameters(), lr=2e-4)
scaler = torch.amp.GradScaler('cuda')

# -------------------- Train loop --------------------
def train_epoch(dl, epoch):
    net.train()
    pbar = tqdm(dl, desc=f"train {epoch}")
    opt.zero_grad(set_to_none=True)

    for step, batch in enumerate(pbar):
        # batch: [N,2,H,W]
        x = batch[:, 0].unsqueeze(1)  # [N,1,H,W]
        m = batch[:, 1].unsqueeze(1)  # [N,1,H,W]

        if CHANNELS_LAST:
            x = x.to(memory_format=torch.channels_last)
            m = m.to(memory_format=torch.channels_last)

        x = x.to(DEVICE, non_blocking=True)
        m = m.to(DEVICE, non_blocking=True)

        N = x.size(0)
        t = torch.randint(0, T, (N,), device=DEVICE)
        noise = torch.randn_like(x)
        a_t = a_bar[t].view(N, 1, 1, 1)
        x_t = torch.sqrt(a_t) * x + torch.sqrt(1 - a_t) * noise
        inp = torch.cat([x_t, m], dim=1)

        with torch.amp.autocast('cuda'):
            t_emb = timestep_embed(t, dim=128)
            pred  = net(inp, t_emb)
            loss  = F.mse_loss(pred, noise) / ACCUM_STEPS

        scaler.scale(loss).backward()

        if (step + 1) % ACCUM_STEPS == 0:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

        pbar.set_postfix(loss=float(loss) * ACCUM_STEPS)

    torch.cuda.empty_cache()

# inference
@torch.no_grad()
def sample(mask_np, n_steps=50):
    net.eval()
    x = torch.randn(1, 1, *mask_np.shape, device=DEVICE)
    m = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    if CHANNELS_LAST:
        x = x.to(memory_format=torch.channels_last)
        m = m.to(memory_format=torch.channels_last)

    ts = torch.linspace(T-1, 0, n_steps, dtype=torch.long, device=DEVICE)
    for ti in ts:
        with torch.amp.autocast('cuda'):
            t_emb = timestep_embed(ti.unsqueeze(0), 128)
            eps   = net(torch.cat([x, m], 1), t_emb)
            a     = a_bar[ti]
            a_prev= a_bar[max(ti-1, 0)]
            beta  = betas[ti]
            x0_hat= (x - torch.sqrt(1 - a) * eps) / torch.sqrt(a)
            mean  = torch.sqrt(a_prev) * x0_hat + torch.sqrt(1 - a_prev) * 0
        x = mean + (torch.sqrt(beta) * torch.randn_like(x) if ti > 0 else 0)

    out = x.clamp(-1, 1).cpu().numpy()[0, 0]
    return ((out + 1) * 127.5).astype(np.uint8)

# run training
if __name__ == "__main__":
 EPOCHS = 50
 for e in range(EPOCHS):
    train_epoch(train_dl, e)

 torch.save(net.state_dict(), f"{SAVE}/ddpm_mask2image.pth")
 print("Saved:", f"{SAVE}/ddpm_mask2image.pth")
