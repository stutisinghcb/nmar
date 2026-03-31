import numpy as np
import pydicom

def extract_cbct_geometry_from_dcm(ds, raw_proj):
    num_views, det_rows, det_cols = raw_proj.shape

    # detector pixel size
    pixel_height = 1.0
    pixel_width = 1.0
    # if hasattr(ds, "ImagerPixelSpacing") and ds.ImagerPixelSpacing is not None:
    #     pixel_height, pixel_width = map(float, ds.ImagerPixelSpacing)
    # elif hasattr(ds, "PixelSpacing") and ds.PixelSpacing is not None:
    #     pixel_height, pixel_width = map(float, ds.PixelSpacing)

    if hasattr(ds, "ImagerPixelSpacing") and ds.ImagerPixelSpacing is not None:
        val = ds.ImagerPixelSpacing
    elif hasattr(ds, "PixelSpacing") and ds.PixelSpacing is not None:
        val = ds.PixelSpacing
    else:
        val = None

    if val is None:
        pixel_height = 1.0
        pixel_width = 1.0
    else:
        if isinstance(val, (list, tuple)):
            pixel_height, pixel_width = map(float, val)
        else:
            # single value → assume square pixels
            pixel_height = float(val)
            pixel_width = float(val)

    # detector center
    center_row = 0.5 * (det_rows - 1)
    center_col = 0.5 * (det_cols - 1)

    # distances
    sod = getattr(ds, "DistanceSourceToPatient", None)
    sdd = getattr(ds, "DistanceSourceToDetector", None)

    sod = float(sod) if sod is not None else None
    sdd = float(sdd) if sdd is not None else None

    # angles: fallback only
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

if __name__ == "__main__":

    dcm_path = r"D:\NMAR\case_01_MFOV\pyrecon\raw\case_13_40006_CEUE00388452_wrist_R_UC.dcm"
    ds = pydicom.dcmread(dcm_path)
    raw_proj = ds.pixel_array

    print(extract_cbct_geometry_from_dcm(ds, raw_proj))
