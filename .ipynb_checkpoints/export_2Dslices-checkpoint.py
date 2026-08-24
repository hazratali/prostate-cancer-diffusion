import os, json, numpy as np, nibabel as nib
from pathlib import Path
from skimage.exposure import rescale_intensity
from tqdm import tqdm

# dataset paths
DS="/space/local/cug/nnUNet_raw/Dataset001_PROSTATE"  
IMT=Path(f"{DS}/imagesTr"); LBT=Path(f"{DS}/labelsTr")
OUT=Path(f"{DS}/slices_axial"); (OUT/"train").mkdir(parents=True, exist_ok=True); (OUT/"val").mkdir(parents=True, exist_ok=True)

# train/validation split (90/10)
cases = sorted([
    p.with_suffix('').with_suffix('').name.replace('_0000', '')
    for p in IMT.glob('*.nii.gz')
])
split=int(len(cases)*0.9); train_cases, val_cases = cases[:split], cases[split:]

#image normalisation
def to_uint8(x):
    x = rescale_intensity(
        x,
        in_range=(np.percentile(x,0.5), np.percentile(x,99.5)),
        out_range=(0,1)
    )
    x = np.clip(x, 0, 1) 
    return (x * 255).astype(np.uint8)


#coversion function
def export_split(cases, sub):
    out = OUT/sub
    index=[]
    for c in tqdm(cases, desc=sub):
        img = nib.load(IMT/f"{c}_0000.nii.gz").get_fdata()  # (X,Y,Z)
        seg = nib.load(LBT/f"{c}.nii.gz").get_fdata()
        # transpose to (Z,Y,X) for axial slicing (z first)
        img = np.moveaxis(img, -1, 0); seg = np.moveaxis(seg, -1, 0)
        for z in range(img.shape[0]):
            # keep slices that have any label to train on anatomy
            if seg[z].max() < 0.5: 
                continue
            I = to_uint8(img[z])
            M = (seg[z] > 0.5).astype(np.uint8)
            np.savez_compressed(out/f"{c}_z{z:03d}.npz", img=I, mask=M)
            index.append(f"{c}_z{z:03d}.npz")
    with open(out/"index.json","w") as f: json.dump(index,f,indent=2)

export_split(train_cases,"train")
export_split(val_cases,"val")
print("Done:", OUT)
