import os
import pydicom
import numpy as np
from pydicom.uid import ExplicitVRLittleEndian

def convert_dcm(in_path, out_dir):
    # Read DICOM
    ds = pydicom.dcmread(in_path)

    # Force pixel data decoding (important for compressed inputs)
    arr = ds.pixel_array

    # Ensure int16 (typical for CT/CBCT HU)
    if arr.dtype != np.int16:
        arr = arr.astype(np.int16)

    # Update dataset with uncompressed format
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    # Update pixel-related fields
    ds.Rows, ds.Columns = arr.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1  # signed

    # Assign pixel data
    ds.PixelData = arr.tobytes()
    ds["PixelData"].is_undefined_length = False

    # Output path
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.basename(in_path))

    # Save
    ds.save_as(out_path)

    print(f"Saved: {out_path}")

#
# if __name__ == "__main__":
#     input_file = r"D:\NMAR\input\slice.dcm"
#     output_dir = r"D:\NMAR\output_uncompressed"
#
#     convert_dcm(input_file, output_dir)

def convert_folder(in_dir, out_dir):
    files = [f for f in os.listdir(in_dir) if f.lower().endswith(".dcm")]

    for i, f in enumerate(files):
        in_path = os.path.join(in_dir, f)
        try:
            convert_dcm(in_path, out_dir)
        except Exception as e:
            print(f"Failed: {f} -> {e}")


if __name__ == "__main__":
    input_dir = r"D:\NMAR\cbct_dicom"
    output_dir = r"D:\NMAR\cbct_uncompressed"

    convert_folder(input_dir, output_dir)