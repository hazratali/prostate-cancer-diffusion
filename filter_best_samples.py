#!/usr/bin/env python3

import os
import glob
import csv
import shutil

import numpy as np
import nibabel as nib
from scipy.stats import entropy
from skimage.metrics import structural_similarity as ssim

# Configuration 
ALPHA       = 0.85   # SSIM weight
KEEP_RATIO  = 0.30
N_BINS      = 64

# Paths 
SYNTH_DIR   = os.path.expanduser(
    "~/medicalimaging/nnunet/results/synth_from_weak_cond_normalized")
REAL_DIR    = "/space/local/cug/nnUNet_raw/Dataset001_PROSTATE/imagesTr"
LABEL_DIR   = "/space/local/cug/nnUNet_raw/Dataset001_PROSTATE/labelsTr"
OUTPUT_BASE = os.path.expanduser(
    "~/medicalimaging/nnunet/results/composite_filter_volumes")

# Output folders 
TOP30_IMG  = os.path.join(OUTPUT_BASE, "top30",    "imagesTr")
TOP30_LAB  = os.path.join(OUTPUT_BASE, "top30",    "labelsTr")
REJECT_IMG = os.path.join(OUTPUT_BASE, "rejected", "imagesTr")
REJECT_LAB = os.path.join(OUTPUT_BASE, "rejected", "labelsTr")
CSV_PATH   = os.path.join(OUTPUT_BASE, "composite_scores.csv")
COMP_PATH  = os.path.join(OUTPUT_BASE, "selection_comparison.csv")

for d in [TOP30_IMG, TOP30_LAB, REJECT_IMG, REJECT_LAB]:
    os.makedirs(d, exist_ok=True)

#  Helpers 
def preprocess(vol):
    """1st-99th percentile clip + min-max normalise to [0, 1]."""
    p1, p99 = np.percentile(vol, 1), np.percentile(vol, 99)
    vol = np.clip(vol, p1, p99)
    mn, mx = vol.min(), vol.max()
    if mx - mn < 1e-8:
        return vol
    return (vol - mn) / (mx - mn)


def compute_ssim(synth, real, label):
    """
    Mean SSIM across all axial slices where label > 0.
    Restricts evaluation to prostate-containing slices only.
    """
    valid = [z for z in range(label.shape[2])
             if label[..., z].max() > 0]
    if not valid:
        return 0.0
    return float(np.mean([
        ssim(synth[..., z], real[..., z], data_range=1.0)
        for z in valid
    ]))


def compute_js(synth, real, label):
    """
    Jensen-Shannon divergence between foreground intensity distributions
    (label > 0). Base-2 log — bounded in [0, 1].
    """
    mask     = label > 0
    synth_fg = synth[mask]
    real_fg  = real[mask]

    if synth_fg.size == 0:
        return 1.0

    bins = np.linspace(0.0, 1.0, N_BINS + 1)
    eps  = 1e-10

    p, _ = np.histogram(synth_fg, bins=bins, density=False)
    q, _ = np.histogram(real_fg,  bins=bins, density=False)

    p = p.astype(np.float64) + eps;  p /= p.sum()
    q = q.astype(np.float64) + eps;  q /= q.sum()

    m  = 0.5 * (p + q)
    js = 0.5 * entropy(p, m, base=2) + 0.5 * entropy(q, m, base=2)
    return float(np.clip(js, 0.0, 1.0))


def find_real(stem):
    for suffix in [f"{stem}_0000.nii.gz", f"{stem}.nii.gz"]:
        path = os.path.join(REAL_DIR, suffix)
        if os.path.exists(path):
            return path
    return None


def find_label(stem):
    path = os.path.join(LABEL_DIR, f"{stem}.nii.gz")
    return path if os.path.exists(path) else None


# Score all cases 
synth_files = sorted(glob.glob(os.path.join(SYNTH_DIR, "*.nii.gz")))
print(f"Found {len(synth_files)} synthetic volumes.")
print(f"Computing S = {ALPHA}·SSIM − {1-ALPHA}·JS for all cases...\n")

results = []

for sf in synth_files:
    fname = os.path.basename(sf)
    stem  = fname.replace("_synth_weakcond.nii.gz", "")

    real_path  = find_real(stem)
    label_path = find_label(stem)

    if real_path is None:
        print(f"  [SKIP] {stem} — real image not found")
        continue
    if label_path is None:
        print(f"  [SKIP] {stem} — label not found")
        continue

    try:
        s_vol = preprocess(nib.load(sf).get_fdata().astype(np.float32))
        r_vol = preprocess(nib.load(real_path).get_fdata().astype(np.float32))
        l_vol = nib.load(label_path).get_fdata().astype(np.float32)

        ssim_score = compute_ssim(s_vol, r_vol, l_vol)
        js_score   = compute_js(s_vol, r_vol, l_vol)
        composite  = ALPHA * ssim_score - (1 - ALPHA) * js_score

        results.append({
            "stem":       stem,
            "synth_path": sf,
            "label_path": label_path,
            "ssim":       ssim_score,
            "js":         js_score,
            "composite":  composite,
        })

        print(f"  {stem}  SSIM={ssim_score:.4f}  "
              f"JS={js_score:.4f}  S={composite:+.4f}")

    except Exception as e:
        print(f"  [ERROR] {stem} — {e}")
        continue
    finally:
        try:
            del s_vol, r_vol, l_vol
        except:
            pass

print(f"\nScored {len(results)} / {len(synth_files)} cases successfully.")

# Sort by composite and split 
cutoff = max(1, int(len(results) * KEEP_RATIO))

# Weighted composite ranking
results_by_composite = sorted(
    results, key=lambda x: x["composite"], reverse=True)
kept_composite = set(r["stem"] for r in results_by_composite[:cutoff])

# Pure SSIM ranking
results_by_ssim = sorted(
    results, key=lambda x: x["ssim"], reverse=True)
kept_ssim = set(r["stem"] for r in results_by_ssim[:cutoff])

# Final kept set = weighted composite top 30%
kept     = results_by_composite[:cutoff]
rejected = results_by_composite[cutoff:]

# ── Comparison: what differs between SSIM-only and composite selection ─────────
only_in_ssim      = kept_ssim - kept_composite      # SSIM kept, composite rejected
only_in_composite = kept_composite - kept_ssim      # composite kept, SSIM rejected
in_both           = kept_ssim & kept_composite      # same in both

print(f"\n{'─'*58}")
print(f"  Selection Comparison  (α={ALPHA})")
print(f"{'─'*58}")
print(f"  In both selections         : {len(in_both)}")
print(f"  Only in SSIM top 30%       : {len(only_in_ssim)}")
print(f"  Only in composite top 30%  : {len(only_in_composite)}")
print(f"{'─'*58}")

if only_in_ssim:
    print(f"\n  Cases kept by SSIM but displaced by JS penalty:")
    for r in results_by_ssim[:cutoff]:
        if r["stem"] in only_in_ssim:
            print(f"    {r['stem']}  SSIM={r['ssim']:.4f}  "
                  f"JS={r['js']:.4f}  S={r['composite']:+.4f}")

if only_in_composite:
    print(f"\n  Cases kept by composite but not pure SSIM:")
    for r in results_by_composite[:cutoff]:
        if r["stem"] in only_in_composite:
            print(f"    {r['stem']}  SSIM={r['ssim']:.4f}  "
                  f"JS={r['js']:.4f}  S={r['composite']:+.4f}")

# Save composite scores CSV 
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=["stem", "ssim", "js", "composite",
                       "composite_decision", "ssim_only_decision"])
    writer.writeheader()
    for r in results_by_composite:
        writer.writerow({
            "stem":                 r["stem"],
            "ssim":                 f"{r['ssim']:.4f}",
            "js":                   f"{r['js']:.4f}",
            "composite":            f"{r['composite']:.4f}",
            "composite_decision":   "top30" if r["stem"] in kept_composite
                                    else "rejected",
            "ssim_only_decision":   "top30" if r["stem"] in kept_ssim
                                    else "rejected",
        })
print(f"\nSaved scores CSV:       {CSV_PATH}")

#  Save comparison CSV 
with open(COMP_PATH, "w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=["stem", "ssim", "js", "composite", "status"])
    writer.writeheader()
    for r in results:
        if r["stem"] in in_both:
            status = "both"
        elif r["stem"] in only_in_ssim:
            status = "ssim_only"
        elif r["stem"] in only_in_composite:
            status = "composite_only"
        else:
            status = "rejected_both"
        writer.writerow({
            "stem":      r["stem"],
            "ssim":      f"{r['ssim']:.4f}",
            "js":        f"{r['js']:.4f}",
            "composite": f"{r['composite']:.4f}",
            "status":    status,
        })
print(f"Saved comparison CSV:   {COMP_PATH}")

# Copy files 
print(f"\nCopying top 30% ({len(kept)} cases) to top30/...")
for item in kept:
    shutil.copy2(item["synth_path"],
                 os.path.join(TOP30_IMG,
                              f"{item['stem']}_synth_weakcond.nii.gz"))
    shutil.copy2(item["label_path"],
                 os.path.join(TOP30_LAB, f"{item['stem']}.nii.gz"))

print(f"Copying rejected ({len(rejected)} cases) to rejected/...")
for item in rejected:
    shutil.copy2(item["synth_path"],
                 os.path.join(REJECT_IMG,
                              f"{item['stem']}_synth_weakcond.nii.gz"))
    shutil.copy2(item["label_path"],
                 os.path.join(REJECT_LAB, f"{item['stem']}.nii.gz"))

#Final summary 
composites = [r["composite"] for r in results]
ssims      = [r["ssim"]      for r in results]
js_list    = [r["js"]        for r in results]

print(f"\n{'─'*58}")
print(f"  Composite Fidelity Filter v2")
print(f"  S = {ALPHA}·SSIM − {1-ALPHA}·JS")
print(f"{'─'*58}")
print(f"  Total scored              : {len(results)}")
print(f"  Kept  (top 30%)           : {len(kept)}")
print(f"  Rejected                  : {len(rejected)}")
print(f"{'─'*58}")
print(f"  Mean SSIM                 : {np.mean(ssims):.4f}")
print(f"  Mean JS                   : {np.mean(js_list):.4f}")
print(f"  Mean composite S          : {np.mean(composites):.4f}")
print(f"{'─'*58}")
print(f"  Best  S kept              : {kept[0]['composite']:+.4f}"
      f"  ({kept[0]['stem']})")
print(f"  Worst S kept              : {kept[-1]['composite']:+.4f}"
      f"  ({kept[-1]['stem']})")
print(f"  Best  S rejected          : {rejected[0]['composite']:+.4f}"
      f"  ({rejected[0]['stem']})")
print(f"{'─'*58}")
print(f"  Overlap with SSIM-only    : {len(in_both)} / {cutoff} cases")
print(f"  Displaced by JS penalty   : {len(only_in_ssim)} cases")
print(f"{'─'*58}")
print(f"\nOutput: {OUTPUT_BASE}")


