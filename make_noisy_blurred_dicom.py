import os
import uuid
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter

# -------- CONFIG --------
INPUT_DICOM = r"C:\Users\alexs\Desktop\test"   # MUST contain .dcm files
OUTPUT_DICOM = r"C:\Users\alexs\Desktop\test_NOISY_BLUR"

NOISE_STD = 15
BLUR_SIGMA = 1.2
# ------------------------

def new_uid():
    return f"2.25.{uuid.uuid4().int}"

os.makedirs(OUTPUT_DICOM, exist_ok=True)

reader = sitk.ImageSeriesReader()
uids = reader.GetGDCMSeriesIDs(INPUT_DICOM)
if not uids:
    raise RuntimeError("No DICOM series found in INPUT_DICOM")

files = reader.GetGDCMSeriesFileNames(INPUT_DICOM, uids[0])
reader.SetFileNames(files)
reader.MetaDataDictionaryArrayUpdateOn()
reader.LoadPrivateTagsOn()

img = reader.Execute()
vol = sitk.GetArrayFromImage(img).astype(np.float32)  # (z, y, x)

# ---- add blur ----
for z in range(vol.shape[0]):
    vol[z] = gaussian_filter(vol[z], sigma=BLUR_SIGMA)

# ---- add noise ----
vol += np.random.normal(0, NOISE_STD, size=vol.shape).astype(np.float32)

# Clamp
vmin, vmax = float(vol.min()), float(vol.max())
vol = np.clip(vol, vmin, vmax)

# Back to image
out_img = sitk.GetImageFromArray(vol.astype(np.int16))
out_img.CopyInformation(img)

writer = sitk.ImageFileWriter()

# One UID for whole series
series_uid = new_uid()

# FrameOfReferenceUID (reuse if possible)
try:
    frame_uid = reader.GetMetaData(0, "0020|0052")
except Exception:
    frame_uid = new_uid()

for k in range(out_img.GetSize()[2]):
    sl = out_img[:, :, k]

    # Copy metadata (portable way)
    for key in reader.GetMetaDataKeys(k):
        sl.SetMetaData(key, reader.GetMetaData(k, key))

    # Required derived-series tags
    sl.SetMetaData("0008|0008", "DERIVED\\SECONDARY")
    sl.SetMetaData("0008|103e", "Synthetic Noisy+Blurred MRI (Test)")
    sl.SetMetaData("0020|000e", series_uid)        # SeriesInstanceUID
    sl.SetMetaData("0020|0013", str(k + 1))        # InstanceNumber
    sl.SetMetaData("0020|0052", frame_uid)         # FrameOfReferenceUID
    sl.SetMetaData("0008|0018", new_uid())         # SOPInstanceUID

    # Windowing (prevents black images)
    center = int((vmax + vmin) / 2)
    width = int(max(1, vmax - vmin))
    sl.SetMetaData("0028|1050", str(center))
    sl.SetMetaData("0028|1051", str(width))

    writer.SetFileName(os.path.join(OUTPUT_DICOM, f"IM_{k+1:04d}.dcm"))
    writer.Execute(sl)

print("✅ Noisy & blurred DICOM series created")
print("Output folder:", OUTPUT_DICOM)
print("SeriesInstanceUID:", series_uid)
