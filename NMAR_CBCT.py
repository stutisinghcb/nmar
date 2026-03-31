import sys
sys.path.append(r"C:\dev\LEAP\build\lib")

import os
import numpy as np
import pydicom
import tifffile
from scipy.ndimage import gaussian_filter
from leapctype import *

leapct = tomographicModels()


def load_raw_projection_dicom(dcm_path):
    ds = pydicom.dcmread(dcm_path)
    raw_proj = ds.pixel_array.astype(np.float32)
    return ds, raw_proj


def extract_cbct_geometry_from_dcm(ds, raw_proj):
    num_views, det_rows, det_cols = raw_proj.shape

    val = getattr(ds, "ImagerPixelSpacing", None)
    if val is None:
        val = getattr(ds, "PixelSpacing", None)

    if isinstance(val, (list, tuple)):
        pixel_height, pixel_width = map(float, val)
    elif val is not None:
        pixel_height = pixel_width = float(val)
    else:
        pixel_height = pixel_width = 1.0

    center_row = 0.5 * (det_rows - 1)
    center_col = 0.5 * (det_cols - 1)

    sod = float(getattr(ds, "DistanceSourceToPatient", 462.817))
    sdd = float(getattr(ds, "DistanceSourceToDetector", 724.296))

    angles_deg = np.linspace(0, 360, num_views, endpoint=False).astype(np.float32)

    return {
        "num_views": num_views,
        "det_rows": det_rows,
        "det_cols": det_cols,
        "pixel_height": pixel_height,
        "pixel_width": pixel_width,
        "center_row": center_row,
        "center_col": center_col,
        "angles_deg": angles_deg,
        "sod": sod,
        "sdd": sdd,
    }


def setup_conebeam(geom):
    # PanelUOffset = 0.392
    # PanelVOffset = 37.746
    # PanelUOffset = 0

    PanelUOffset = 0
    PanelVOffset = 0


    print('geom["pixel_width]: ', geom["pixel_width"])
    u_offset_pix = PanelUOffset / geom["pixel_width"]
    v_offset_pix = PanelVOffset / geom["pixel_height"]

    # center_col = 0.5 * (geom["det_cols"] - 1) + u_offset_pix
    # center_row = 0.5 * (geom["det_rows"] - 1) + v_offset_pix

    leapct.set_conebeam(
        numAngles=geom["num_views"],
        numRows=geom["det_rows"],
        numCols=geom["det_cols"],
        pixelHeight=geom["pixel_height"],
        pixelWidth=geom["pixel_width"],
        # centerRow=geom["center_row"],
        # centerCol=geom["center_col"],
        # centerRow= 0.5 * (geom["det_cols"] - 1) + u_offset_pix,
        # centerCol = 0.5 * (geom["det_rows"] - 1) + v_offset_pix,
        centerRow = geom["center_row"] + u_offset_pix,
        centerCol = geom["center_col"] + v_offset_pix,
        phis=geom["angles_deg"].astype(np.float32),
        # phis=np.mod(geom["angles_deg"] - 90.0, 360.0).astype(np.float32),
        sod=geom["sod"],
        sdd=geom["sdd"],
    )


def setup_volume_from_mask(metal_vol, voxel_size_mm=0.5):
    nz, ny, nx = metal_vol.shape

    # Adjust this call to your LEAP volume API if the signature differs.
    leapct.set_volume(
        numX=nx,
        numY=ny,
        numZ=nz,
        voxelWidth=voxel_size_mm,
        voxelHeight=voxel_size_mm
    )


def allocate_and_project(volume):
    g = leapct.allocate_projections()
    f = leapct.allocate_volume()

    if f.shape != volume.shape:
        raise ValueError(f"Volume shape mismatch: LEAP {f.shape} vs input {volume.shape}")

    f[:] = volume.astype(np.float32)
    leapct.project(g, f)
    return g.copy()


def reconstruct_fdk(proj):
    g = leapct.allocate_projections()
    f = leapct.allocate_volume()

    if g.shape != proj.shape:
        raise ValueError(f"Projection shape mismatch: LEAP {g.shape} vs proj {proj.shape}")

    g[:] = proj.astype(np.float32)
    f[:] = 0
    leapct.FBP(g, f)
    return f.copy()


def interp_1d_over_mask(signal_1d, mask_1d):
    out = signal_1d.copy()
    metal_idx = np.where(mask_1d)[0]
    nonmetal_idx = np.where(~mask_1d)[0]

    if len(metal_idx) == 0:
        return out
    if len(nonmetal_idx) < 2:
        return out

    out[metal_idx] = np.interp(metal_idx, nonmetal_idx, signal_1d[nonmetal_idx])
    return out


def interp_proj_stack(proj, metal_trace):
    out = proj.copy()
    num_views, det_rows, _ = proj.shape

    for v in range(num_views):
        for r in range(det_rows):
            out[v, r] = interp_1d_over_mask(proj[v, r], metal_trace[v, r])

    return out


def hu_to_mu(vol_hu, mu_water=0.19):
    return mu_water * (1.0 + vol_hu / 1000.0)


def mu_to_hu(vol_mu, mu_water=0.19):
    return (vol_mu / mu_water - 1.0) * 1000.0


def build_prior_from_li(vol_li_mu, metal_vol, mu_air=0.0, mu_water=0.19, bone_thresh=0.25):
    prior = gaussian_filter(vol_li_mu, sigma=1.0)

    prior[prior < mu_air] = mu_air

    # remove metal from prior
    prior[metal_vol > 0] = mu_water

    # simple 3-class style prior
    air_mask = prior <= 0.02
    soft_mask = (prior > 0.02) & (prior < bone_thresh)

    prior[air_mask] = mu_air
    prior[soft_mask] = mu_water
    # bone is left as-is

    return prior


def run_nmar(raw_proj, proj_prior, metal_trace, eps=1e-6):
    proj_prior = np.clip(proj_prior, eps, None)

    proj_norm = raw_proj / proj_prior
    proj_norm_interp = interp_proj_stack(proj_norm, metal_trace)

    proj_nmar = proj_norm_interp * proj_prior
    proj_nmar[~metal_trace] = raw_proj[~metal_trace]

    return proj_nmar

import numpy as np
import pydicom
from pydicom.uid import ExplicitVRLittleEndian


def save_projection_stack_as_dcm(template_ds, proj_stack, out_path):
    """
    Save projection stack (views, rows, cols) into a DICOM file
    using a template dataset.
    """

    ds = template_ds.copy()

    # ---- ensure dtype ----
    if proj_stack.dtype != np.uint16:
        # normalize to 16-bit range
        proj_min = proj_stack.min()
        proj_max = proj_stack.max()

        proj_scaled = (proj_stack - proj_min) / (proj_max - proj_min + 1e-6)
        proj_scaled = (proj_scaled * 65535).astype(np.uint16)
    else:
        proj_scaled = proj_stack

    # ---- update dimensions ----
    num_views, rows, cols = proj_scaled.shape

    ds.NumberOfFrames = str(num_views)
    ds.Rows = rows
    ds.Columns = cols

    # ---- flatten pixel data ----
    ds.PixelData = proj_scaled.tobytes()

    # ---- critical: make it uncompressed ----
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    # ---- required fields ----
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0  # unsigned

    # ---- save ----
    ds.save_as(out_path)

    print(f"Saved DICOM: {out_path}")

if __name__ == "__main__":
    raw_dcm_path = r"D:\NMAR\case_01_MFOV\pyrecon\raw\case_13_40006_CEUE00388452_wrist_R_UC.dcm"
    metal_mask_path = r"D:\NMAR\case_01_MFOV\mar_metal_mask_vol.tif"
    out_dir = r"D:\NMAR\cbct_nmar_no_hu"
    os.makedirs(out_dir, exist_ok=True)

    # 1) Load raw projections
    ds, raw_proj = load_raw_projection_dicom(raw_dcm_path)
    print("raw_proj shape:", raw_proj.shape)

    # I = raw_proj.astype(np.float32)
    # I0 = np.percentile(I, 99.9)
    # proj = -np.log(np.clip(I / I0, 1e-6, None))

 ###### raw proj test
    print("raw_proj.dtype, raw_proj.min(), raw_proj.max():" , raw_proj.dtype, raw_proj.min(), raw_proj.max())
    print("np.percentile(raw_proj, [0.1, 1, 5, 50, 95, 99, 99.9]) ", np.percentile(raw_proj, [0.1, 1, 5, 50, 95, 99, 99.9]))

    I = raw_proj.astype(np.float32)
    I0 = np.percentile(I, 99.9)
    proj_log = -np.log(np.clip(I / I0, 1e-6, None))





###### raw proj test

    geom = extract_cbct_geometry_from_dcm(ds, raw_proj)
    setup_conebeam(geom)
    metal_vol = tifffile.imread(metal_mask_path)
    metal_vol = (metal_vol > 0).astype(np.float32)
    print("metal_vol shape:", metal_vol.shape)
    setup_volume_from_mask(metal_vol, voxel_size_mm=0.5)
    vol_test = reconstruct_fdk(proj_log)


    # z = vol_test.shape[0] // 2
    # slice_2d = vol_test[z]
    # # output path
    # out_path = os.path.join(out_dir, "vol_li_mid_slice_u3v37.tif")
    #
    # # save
    # tifffile.imwrite(out_path, slice_2d.astype(np.float32))
    #
    # print(f"Saved: {out_path}")

###### raw proj test

    # # # 2) Geometry from DICOM
    # geom = extract_cbct_geometry_from_dcm(ds, raw_proj)
    # #
    # # # Apply the angle fix you found
    geom["angles_deg"] = np.mod(geom["angles_deg"] - 90.0, 360.0).astype(np.float32)
    #
    #
    #
    # # # 3) Load metal mask
    metal_vol = tifffile.imread(metal_mask_path)
    metal_vol = (metal_vol > 0).astype(np.float32)
    print("metal_vol shape:", metal_vol.shape)
    #
    # # # 4) Configure LEAP
    setup_conebeam(geom)
    setup_volume_from_mask(metal_vol, voxel_size_mm=0.5)
    # #
    # # # 5) Forward project metal mask
    metal_proj = allocate_and_project(metal_vol)
    # metal_trace = metal_proj > 1e-6
    # #
    metal_trace = metal_proj > (0.02 * metal_proj.max())
    np.save(os.path.join(out_dir, "metal_proj.npy"), metal_proj)
    np.save(os.path.join(out_dir, "metal_trace.npy"), metal_trace.astype(np.uint8))

    # pick a representative view (middle angle)
    v = metal_trace.shape[0] // 2
              
    slice_2d = metal_trace[v].astype(np.uint8)  # 0/1 mask

    out_path = os.path.join(out_dir, "metal_trace_mid_view.tif")

    # scale for visibility (0 → 0, 1 → 255)
    tifffile.imwrite(out_path, (slice_2d * 255).astype(np.uint8))

    print(f"Saved: {out_path}")

    for v in [0, 120, 240, 360]:
        raw = proj_log[v] if 'proj_log' in locals() else raw_proj[v]
        mask = metal_trace[v]

        p1, p99 = np.percentile(raw, (1, 99))
        raw_norm = np.clip((raw - p1) / (p99 - p1 + 1e-6), 0, 1)

        rgb = np.stack([raw_norm, raw_norm, raw_norm], axis=-1)

        alpha = 0.5
        rgb[..., 0] = np.where(mask, (1 - alpha) * rgb[..., 0] + alpha * 1.0, rgb[..., 0])
        rgb[..., 1] = np.where(mask, (1 - alpha) * rgb[..., 1], rgb[..., 1])
        rgb[..., 2] = np.where(mask, (1 - alpha) * rgb[..., 2], rgb[..., 2])

        out_path = os.path.join(out_dir, f"metal_trace02_overlay_{v:04d}.tif")
        tifffile.imwrite(out_path, (rgb * 255).astype(np.uint8))

        print(f"Saved: {out_path}")
    # #
    # # # 6) Linear interpolation MAR in projection space
    proj_li = interp_proj_stack(proj_log, metal_trace)
    np.save(os.path.join(out_dir, "proj_li.npy"), proj_li)
    #
    # #7) Reconstruct LI volume
    vol_li = reconstruct_fdk(proj_li)
    np.save(os.path.join(out_dir, "vol_li.npy"), vol_li)
    #
    # import numpy as np
    # import tifffile
    # import os
    #
    # pick middle slice
    z = vol_li.shape[0] // 2
    slice_2d = vol_li[z]
    out_path = os.path.join(out_dir, "vol_li_LI.tif")
    tifffile.imwrite(out_path, slice_2d.astype(np.float32))
    #
    print(f"Saved: {out_path}")
    # z = vol_test.shape[0] // 2
    # slice_2d = vol_test[z]
    # # output path
    # out_path = os.path.join(out_dir, "vol_li_mid_slice_u3v37.tif")
    #
    # # save
    # tifffile.imwrite(out_path, slice_2d.astype(np.float32))
    #
    # print(f"Saved: {out_path}")



    # #8) Build prior volume
    # #IMPORTANT:
    # #If vol_li is already in attenuation units, use directly.
    # #If it is HU-like, convert first:
    # vol_li_mu = hu_to_mu(vol_li)
    vol_li_mu = vol_li.copy()
    #
    prior_vol = build_prior_from_li(vol_li_mu, metal_vol)
    np.save(os.path.join(out_dir, "prior_vol.npy"), prior_vol)
    #
    # # 9) Forward project prior
    proj_prior = allocate_and_project(prior_vol)
    np.save(os.path.join(out_dir, "proj_prior.npy"), proj_prior)

    import matplotlib.pyplot as plt

    plt.imshow(proj_log[180], cmap='gray')
    plt.title("proj_log")

    plt.figure()
    plt.imshow(proj_prior[180], cmap='gray')
    plt.title("proj_prior")
    #
    # # 10) NMAR
    proj_nmar = run_nmar(proj_log, proj_prior, metal_trace)
    np.save(os.path.join(out_dir, "proj_nmar.npy"), proj_nmar)
    #
    # print("Saved NMAR projection stack.")

    import matplotlib.pyplot as plt
    import numpy as np

    for v in [0, 120, 240, 360]:
        diff = proj_nmar[v] - raw_proj[v]

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(raw_proj[v], cmap='gray')
        plt.title("Original")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(proj_nmar[v], cmap='gray')
        plt.title("NMAR")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(diff, cmap='bwr',
                   vmin=-np.max(np.abs(diff)),
                   vmax=np.max(np.abs(diff)))
        plt.title("Difference")
        plt.axis('off')

        plt.tight_layout()

        save_path = os.path.join(out_dir, f"diff_{v:04d}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved: {save_path}")
    #

    # ds, raw_proj = load_raw_projection_dicom(raw_dcm_path)
    #
    # save_projection_stack_as_dcm(
    #     template_ds=ds,
    #     proj_stack=proj_nmar,
    #     out_path=r"D:\NMAR\case_01_MFOV\nmar_projections\nmar_proj.dcm"
    # )

    vol_nmar = reconstruct_fdk(proj_nmar)
    np.save(os.path.join(out_dir, "vol_nmar.npy"), vol_nmar)

    z = vol_nmar.shape[0] // 2
    slice_2d = vol_nmar[z]
    out_path = os.path.join(out_dir, "vol_nmar_mid_slice.tif")
    tifffile.imwrite(out_path, slice_2d.astype(np.float32))

    print(f"Saved: {out_path}")

    # -----------------------------
    # Difference: NMAR vs uncorrected
    # -----------------------------
    diff_vol = vol_nmar - vol_test
    np.save(os.path.join(out_dir, "vol_diff_nmar_vs_raw.npy"), diff_vol)

    # pick same slice index
    z = vol_nmar.shape[0] // 2

    slice_raw = vol_test[z]
    slice_nmar = vol_nmar[z]
    slice_diff = diff_vol[z]


    # -----------------------------
    # Normalize for visualization
    # -----------------------------
    def normalize(img):
        p1, p99 = np.percentile(img, (1, 99))
        return np.clip((img - p1) / (p99 - p1 + 1e-6), 0, 1)


    raw_norm = normalize(slice_raw)
    nmar_norm = normalize(slice_nmar)

    # symmetric scaling for difference
    vmax = np.percentile(np.abs(slice_diff), 99)
    diff_vis = np.clip(slice_diff / (vmax + 1e-6), -1, 1)

    # -----------------------------
    # Save outputs
    # -----------------------------
    tifffile.imwrite(os.path.join(out_dir, "slice_raw.tif"), (raw_norm * 255).astype(np.uint8))
    tifffile.imwrite(os.path.join(out_dir, "slice_nmar.tif"), (nmar_norm * 255).astype(np.uint8))

    # difference as float (for analysis)
    tifffile.imwrite(
        os.path.join(out_dir, "slice_diff_float.tif"),
        slice_diff.astype(np.float32)
    )

    # difference as visualization (shift to 0–255)
    diff_uint8 = ((diff_vis + 1) / 2 * 255).astype(np.uint8)
    tifffile.imwrite(
        os.path.join(out_dir, "slice_diff_vis.tif"),
        diff_uint8
    )

    print("Saved raw, NMAR, and difference slices.")