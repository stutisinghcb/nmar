import os
import numpy as np
import pydicom
from pydicom.uid import ExplicitVRLittleEndian

def save_corrected_dicom(template_path, out_path, image):
    ds = pydicom.dcmread(template_path)

    image = np.asarray(image)

    if image.dtype != np.int16:
        image = image.astype(np.int16)

    ds.Rows, ds.Columns = image.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1

    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    ds.PixelData = image.tobytes()

    if "PixelData" in ds:
        ds["PixelData"].is_undefined_length = False

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ds.save_as(out_path)

if __name__ == "__main__":
    input_dir = r"D:\NMAR\case_01_MFOV\pyrecon\recon\recon"
    output_dir = r"D:\NMAR\test_01\nmar_vol"
    preview_dir = r"D:\NMAR\test_01\nmar_vol_previews"

    run_nmar_on_dicom_series(
        input_dir=input_dir,
        output_dir=output_dir,
        show_result=False,
        save_preview_dir=preview_dir,   # or None
    )