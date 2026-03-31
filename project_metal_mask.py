import sys
sys.path.append(r"C:\dev\LEAP\build\lib")

import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from leapctype import *
import pydicom

from from_dcm import extract_cbct_geometry_from_dcm

leapct = tomographicModels()


def normalize_image(x):
    x = x.astype(np.float32)
    x_min = np.percentile(x, 1)
    x_max = np.percentile(x, 99)
    if x_max <= x_min:
        return np.zeros_like(x, dtype=np.float32)
    x = np.clip((x - x_min) / (x_max - x_min), 0, 1)
    return x


def make_red_overlay(raw_2d, metal_mask_2d, alpha=0.35):
    """
    raw_2d: float/grayscale image
    metal_mask_2d: bool mask
    returns RGB image
    """
    base = normalize_image(raw_2d)
    rgb = np.stack([base, base, base], axis=-1)

    # red overlay
    rgb[..., 0] = np.where(metal_mask_2d, (1 - alpha) * rgb[..., 0] + alpha * 1.0, rgb[..., 0])
    rgb[..., 1] = np.where(metal_mask_2d, (1 - alpha) * rgb[..., 1], rgb[..., 1])
    rgb[..., 2] = np.where(metal_mask_2d, (1 - alpha) * rgb[..., 2], rgb[..., 2])

    return rgb


def setup_conebeam(
    num_views,
    det_rows,
    det_cols,
    pixel_height,
    pixel_width,
    center_row,
    center_col,
    angles_deg,
    sod,
    sdd,
):
    leapct.set_conebeam(
        numAngles=num_views,
        numRows=det_rows,
        numCols=det_cols,
        pixelHeight=pixel_height,
        pixelWidth=pixel_width,
        centerRow=center_row,
        centerCol=center_col,
        phis=np.asarray(angles_deg, dtype=np.float32),
        sod=sod,
        sdd=sdd,
    )


def project_metal_volume(metal_vol):
    g = leapct.allocate_projections()
    f = leapct.allocate_volume()

    print("LEAP volume shape:", f.shape)
    print("Metal volume shape:", metal_vol.shape)

    if f.shape != metal_vol.shape:
        raise ValueError(f"Volume shape mismatch: LEAP {f.shape} vs metal {metal_vol.shape}")

    f[:] = metal_vol.astype(np.float32)
    leapct.project(g, f)
    return g.copy()


def save_overlay_examples(raw_proj, metal_proj, out_dir, every=30):
    os.makedirs(out_dir, exist_ok=True)

    metal_trace = metal_proj > 1e-6

    num_views = raw_proj.shape[0]
    for v in range(0, num_views, every):
        overlay = make_red_overlay(raw_proj[v], metal_trace[v])

        plt.figure(figsize=(8, 6))
        plt.imshow(overlay)
        plt.title(f"View {v}")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"overlay_view_{v:04d}.png"), dpi=150)
        plt.close()

    return metal_trace


if __name__ == "__main__":
    # -------------------------
    # 1) load your data
    # -------------------------
    dcm_path = r"D:\NMAR\case_01_MFOV\pyrecon\raw\case_13_40006_CEUE00388452_wrist_R_UC.dcm"
    ds = pydicom.dcmread(dcm_path)
    raw_proj = ds.pixel_array

    # raw_proj = np.load(r"D:\NMAR\raw_proj.npy")   # shape [views, rows, cols]
    metal_vol = tifffile.imread(r"D:\NMAR\case_01_MFOV\mar_metal_mask_vol.tif")
    metal_vol = (metal_vol > 0).astype(np.float32)

    print("raw_proj shape:", raw_proj.shape)
    print("metal_vol shape:", metal_vol.shape)

    num_views, det_rows, det_cols = raw_proj.shape

    geom = extract_cbct_geometry_from_dcm(ds, raw_proj)

    # PanelUOffset = 0.392
    PanelUOffset = 0
    # PanelVOffset = 37.746
    PanelVOffset = 0

    u_offset_pix =  PanelUOffset / geom["pixel_height"]

    v_offset_pix = PanelVOffset / geom["pixel_height"]

    print("u_offset_pix", u_offset_pix)
    print("v_offset_pix", v_offset_pix)

    num_views = geom["num_views"]
    det_rows = geom["det_rows"]
    det_cols = geom["det_cols"]
    pixel_height = geom["pixel_height"]
    pixel_width = geom["pixel_width"]
    center_row = geom["center_row"] + u_offset_pix
    center_col = geom["center_col"] + v_offset_pix
    # angles_deg = geom["angles_deg"]
    angles_deg = np.mod(geom["angles_deg"] - 90.0, 360.0).astype(np.float32)
    sod = geom["sod"]
    sdd = geom["sdd"]

    leapct.set_conebeam(
        numAngles=num_views,
        numRows=det_rows,
        numCols=det_cols,
        pixelHeight=pixel_height,
        pixelWidth=pixel_width,
        centerRow=center_row,
        centerCol=center_col,
        phis=angles_deg.astype(np.float32),
        sod=sod,
        sdd=sdd,
    )

    # -------------------------
    # 3) set volume model
    # IMPORTANT: replace with the correct volume setup
    # -------------------------
    # leapct.set_default_volume()
    nz, ny, nx = metal_vol.shape
    leapct.set_volume(numZ=nz, numY=ny, numX=nx,
                      voxelHeight=0.5, voxelWidth=0.5)
    #
    # # -------------------------
    # 4) project metal mask
    # -------------------------
    metal_proj = project_metal_volume(metal_vol)
    #
    # -------------------------
    # 5) save outputs
    # -------------------------
    out_dir = r"D:\NMAR\project_metal_mask"
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "metal_proj.npy"), metal_proj)

    metal_trace = metal_proj > 1e-6

    os.makedirs(out_dir, exist_ok=True)

    for v in range(0, raw_proj.shape[0], 20):  # every=20
        raw = raw_proj[v]
        mask = metal_trace[v]

        # normalize raw for display
        raw_norm = raw.astype(np.float32)
        lo, hi = np.percentile(raw_norm, (1, 99))
        raw_norm = np.clip((raw_norm - lo) / (hi - lo), 0, 1)

        # convert to RGB
        rgb = np.stack([raw_norm, raw_norm, raw_norm], axis=-1)

        # overlay metal in RED
        alpha = 0.4
        rgb[..., 0] = np.where(mask, (1 - alpha) * rgb[..., 0] + alpha * 1.0, rgb[..., 0])
        rgb[..., 1] = np.where(mask, (1 - alpha) * rgb[..., 1], rgb[..., 1])
        rgb[..., 2] = np.where(mask, (1 - alpha) * rgb[..., 2], rgb[..., 2])

        # save as PNG
        filename = os.path.join(out_dir, f"overlay_{v:04d}.png")
        plt.imsave(filename, rgb)
    #
    # metal_trace = save_overlay_examples(raw_proj, metal_proj, out_dir, every=20)
    # np.save(os.path.join(out_dir, "metal_trace.npy"), metal_trace.astype(np.uint8))
    #
    # print("Done.")