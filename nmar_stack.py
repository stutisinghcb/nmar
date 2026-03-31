import os
import copy
import uuid
import numpy as np
import pydicom
from pydicom.uid import generate_uid

from nmar import *

# assumes your existing NMAR code is already imported or in the same file:
# from leapctype import *
# leapct = tomographicModels()
# ...
# def mar(im, show_result=True, save_path=None): ...

def load_dicom_series(input_dir):
    files = []
    for name in os.listdir(input_dir):
        path = os.path.join(input_dir, name)
        if not os.path.isfile(path):
            continue
        try:
            ds = pydicom.dcmread(path)
            if hasattr(ds, "PixelData"):
                files.append(ds)
        except Exception:
            pass

    if not files:
        raise ValueError(f"No readable DICOM slices found in: {input_dir}")

    def sort_key(ds):
        if hasattr(ds, "ImagePositionPatient") and len(ds.ImagePositionPatient) >= 3:
            return float(ds.ImagePositionPatient[2])
        if hasattr(ds, "InstanceNumber"):
            return int(ds.InstanceNumber)
        return 0

    files.sort(key=sort_key)
    return files


def dcm_to_hu(ds):
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    return arr * slope + intercept


def hu_to_stored(hu, ds):
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))

    stored = np.round((hu - intercept) / slope)

    bits_stored = int(getattr(ds, "BitsStored", 16))
    pixel_repr = int(getattr(ds, "PixelRepresentation", 1))  # 0=unsigned, 1=signed

    if pixel_repr == 0:
        min_val = 0
        max_val = 2**bits_stored - 1
        dtype = np.uint16 if bits_stored > 8 else np.uint8
    else:
        min_val = -(2 ** (bits_stored - 1))
        max_val = 2 ** (bits_stored - 1) - 1
        dtype = np.int16 if bits_stored > 8 else np.int8

    stored = np.clip(stored, min_val, max_val).astype(dtype)
    return stored


def pad_to_square(img, pad_value=-1000):
    h, w = img.shape
    if h == w:
        return img, (0, 0, 0, 0)

    size = max(h, w)
    out = np.full((size, size), pad_value, dtype=img.dtype)

    top = (size - h) // 2
    left = (size - w) // 2
    out[top:top + h, left:left + w] = img

    return out, (top, top + h, left, left + w)


def unpad_from_square(img, crop_box):
    top, bottom, left, right = crop_box
    if top == 0 and bottom == 0 and left == 0 and right == 0:
        return img
    return img[top:bottom, left:right]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def run_nmar_on_dicom_series(input_dir, output_dir, show_result=False, save_preview_dir=None):
    ensure_dir(output_dir)
    if save_preview_dir is not None:
        ensure_dir(save_preview_dir)

    series = load_dicom_series(input_dir)

    new_series_uid = generate_uid()

    for i, ds in enumerate(series):
        print(f"Processing slice {i+1}/{len(series)}")

        hu = dcm_to_hu(ds)

        # Your mar() requires square 2D input, so pad if needed.
        hu_sq, crop_box = pad_to_square(hu, pad_value=-1000)

        preview_path = None
        if save_preview_dir is not None:
            preview_path = os.path.join(save_preview_dir, f"slice_{i:04d}.png")

        result_sq = mar(hu_sq, show_result=False, save_dir=preview_path)
        result_hu = unpad_from_square(result_sq, crop_box)

        out_ds = copy.deepcopy(ds)

        stored = hu_to_stored(result_hu, out_ds)
        out_ds.PixelData = stored.tobytes()
        out_ds.Rows, out_ds.Columns = stored.shape

        # Give the output a new series identity
        out_ds.SeriesInstanceUID = new_series_uid
        out_ds.SOPInstanceUID = generate_uid()

        if hasattr(out_ds, "SeriesDescription"):
            out_ds.SeriesDescription = f"{out_ds.SeriesDescription} [NMAR]"
        else:
            out_ds.SeriesDescription = "NMAR"

        # Optional: keep instance numbering consistent
        if hasattr(out_ds, "InstanceNumber"):
            out_ds.InstanceNumber = i + 1

        out_path = os.path.join(output_dir, f"IM_{i+1:04d}.dcm")
        out_ds.save_as(out_path)

    print(f"Saved NMAR DICOM series to: {output_dir}")


if __name__ == "__main__":
    input_dir = r"D:\NMAR\case_01_MFOV\pyrecon\recon\recon_fix_compress"
    output_dir = r"D:\NMAR\test_01\nmar_vol"
    preview_dir = r"D:\NMAR\test_01\nmar_vol_previews"

    run_nmar_on_dicom_series(
        input_dir=input_dir,
        output_dir=output_dir,
        show_result=False,
        save_preview_dir=preview_dir,   # or None
    )