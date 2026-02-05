
import argparse
import json
import os
import uuid
from typing import Any, Dict, Optional
import numpy as np
import SimpleITK as sitk
from skimage.restoration import denoise_nl_means, estimate_sigma, denoise_tv_chambolle, richardson_lucy


def new_uid() -> str:
    return f"2.25.{uuid.uuid4().int}"




def gaussian_psf_2d(sigma: float, size: Optional[int] = None) -> np.ndarray:
    if sigma <= 0:
        raise ValueError("psf_gauss must be > 0")
    if size is None:
        size = int(np.ceil(sigma * 6)) | 1
        size = max(size, 7)
    ax = np.arange(-(size // 2), size // 2 + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    psf = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    psf /= float(psf.sum())
    return psf.astype(np.float32)


def normalize01(x: np.ndarray):
    x = x.astype(np.float32, copy=False)
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32), mn, mx
    return (x - mn) / (mx - mn), mn, mx


def unnormalize01(y: np.ndarray, mn: float, mx: float):
    return (y * (mx - mn) + mn).astype(np.float32)


def clamp_to_dtype(arr: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(np.rint(arr), info.min, info.max).astype(dtype)
    return arr.astype(dtype)



def denoise2d(x01: np.ndarray, method: str, nlm_h: Optional[float], tv_weight: float) -> np.ndarray:
    if method == "none":
        return x01
    if method == "tv":
        return denoise_tv_chambolle(x01, weight=float(tv_weight), channel_axis=None)

    # nlm
    sigma_est = float(np.mean(estimate_sigma(x01, channel_axis=None)))
    h = (1.0 * sigma_est) if nlm_h is None else float(nlm_h)
    return denoise_nl_means(
        x01,
        h=h,
        sigma=sigma_est,
        fast_mode=True,
        patch_size=5,
        patch_distance=7,
        channel_axis=None,
    )


def deblur2d(x01: np.ndarray, method: str, psf: Optional[np.ndarray], iters: int, clip: bool) -> np.ndarray:
    if method == "none":
        return x01
    if method != "rl":
        raise ValueError(f"Unknown deblur method: {method}")
    if psf is None:
        raise ValueError("PSF is required for Richardson–Lucy deconvolution.")
    x01 = np.clip(x01, 0, 1)
    y = richardson_lucy(x01, psf, num_iter=int(iters), clip=bool(clip))
    return np.clip(y, 0, 1)


def pick_series_uid(dicom_folder: str) -> str:
    reader = sitk.ImageSeriesReader()
    uids = reader.GetGDCMSeriesIDs(dicom_folder)
    if not uids:
        raise RuntimeError(f"No DICOM series found in folder: {dicom_folder}")



























































    best_uid = max(uids, key=lambda uid: len(reader.GetGDCMSeriesFileNames(dicom_folder, uid)))
    return best_uid


def read_series(dicom_folder: str, series_uid: str):
    reader = sitk.ImageSeriesReader()
    files = reader.GetGDCMSeriesFileNames(dicom_folder, series_uid)
    if not files:
        raise RuntimeError("Series UID exists but has no files. Check folder/UID.")
    reader.SetFileNames(files)

    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()

    img3d = reader.Execute()
    return reader, img3d, files


def write_derived_dicom_series(
    out_folder: str,
    out_img3d: sitk.Image,
    reader: sitk.ImageSeriesReader,
    series_description: str,
    global_min: float,
    global_max: float,
):
    os.makedirs(out_folder, exist_ok=True)

    new_series_uid = new_uid()

    try:
        frame_uid = reader.GetMetaData(0, "0020|0052")
    except Exception:
        frame_uid = new_uid()

    center = int((global_max + global_min) / 2.0)
    width = int(max(1.0, global_max - global_min))

    writer = sitk.ImageFileWriter()

    depth = out_img3d.GetSize()[2]
    for k in range(depth):
        sl = out_img3d[:, :, k]

        try:
            keys = reader.GetMetaDataKeys(k)
        except Exception:
            keys = []

        for key in keys:
            try:
                sl.SetMetaData(key, reader.GetMetaData(k, key))
            except Exception:
                pass

        sl.SetMetaData("0008|0008", "DERIVED\\SECONDARY")
        sl.SetMetaData("0008|103e", series_description)

        sl.SetMetaData("0020|000e", new_series_uid)
        sl.SetMetaData("0008|0018", new_uid())

        sl.SetMetaData("0020|0052", frame_uid)
        sl.SetMetaData("0020|0013", str(k + 1))

        sl.SetMetaData("0028|1050", str(center))
        sl.SetMetaData("0028|1051", str(width))

        out_path = os.path.join(out_folder, f"IM_{k+1:04d}.dcm")
        writer.SetFileName(out_path)
        writer.Execute(sl)

    return new_series_uid



def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_value(cli_val, cfg: Dict[str, Any], key: str, default=None):
    return cli_val if cli_val is not None else cfg.get(key, default)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="Path to config.json")

    ap.add_argument("--dicom", help="Input DICOM folder")
    ap.add_argument("--out_dicom", help="Output DICOM folder")

    ap.add_argument("--series_uid", default=None, help="Optional specific SeriesInstanceUID")
    ap.add_argument("--series_desc", default=None, help="SeriesDescription for output")

    ap.add_argument("--denoise", choices=["none", "nlm", "tv"], default=None)
    ap.add_argument("--nlm_h", type=float, default=None)
    ap.add_argument("--tv_weight", type=float, default=None)

    ap.add_argument("--deblur", choices=["none", "rl"], default=None)
    ap.add_argument("--psf_gauss", type=float, default=None)
    ap.add_argument("--psf_size", type=int, default=None)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--clip", action="store_true")

    args = ap.parse_args()

    cfg: Dict[str, Any] = {}
    if args.config:
        cfg = load_config(args.config)

    dicom_folder = get_value(args.dicom, cfg, "dicom")
    out_folder = get_value(args.out_dicom, cfg, "out_dicom")
    if not dicom_folder or not out_folder:
        raise RuntimeError("You must provide dicom and out_dicom (via --config or CLI).")

    denoise = get_value(args.denoise, cfg, "denoise", "nlm")
    nlm_h = get_value(args.nlm_h, cfg, "nlm_h", None)
    tv_weight = float(get_value(args.tv_weight, cfg, "tv_weight", 0.07))

    deblur = get_value(args.deblur, cfg, "deblur", "rl")
    psf_gauss = float(get_value(args.psf_gauss, cfg, "psf_gauss", 1.0))
    psf_size = get_value(args.psf_size, cfg, "psf_size", None)
    iters = int(get_value(args.iters, cfg, "iters", 12))
    clip = bool(get_value(True if args.clip else None, cfg, "clip", False))

    series_uid = get_value(args.series_uid, cfg, "series_uid", None)
    series_desc = get_value(args.series_desc, cfg, "series_desc", "MRI Denoise+Deblur (Derived)")

    if not os.path.isdir(dicom_folder):
        raise RuntimeError(f"Input path is not a folder: {dicom_folder}")

    if series_uid is None:
        series_uid = pick_series_uid(dicom_folder)

    reader, img3d, files = read_series(dicom_folder, series_uid)


    vol = sitk.GetArrayFromImage(img3d)
    in_dtype = vol.dtype

    vol01, mn, mx = normalize01(vol)

    psf = None
    if deblur != "none":
        psf = gaussian_psf_2d(psf_gauss, psf_size)

    out01 = np.empty_like(vol01, dtype=np.float32)
    for z in range(vol01.shape[0]):
        sl = vol01[z]
        sl = denoise2d(sl, denoise, nlm_h, tv_weight)
        sl = deblur2d(sl, deblur, psf, iters, clip)
        out01[z] = sl

    out_vol = unnormalize01(out01, mn, mx)
    out_vol = clamp_to_dtype(out_vol, in_dtype)

    out_img3d = sitk.GetImageFromArray(out_vol)
    out_img3d.CopyInformation(img3d)

    new_series_uid = write_derived_dicom_series(
        out_folder,
        out_img3d,
        reader,
        series_desc,
        global_min=float(np.min(out_vol)),
        global_max=float(np.max(out_vol)),
    )

    print("✅ Done.")
    print("Input folder:", dicom_folder)
    print("Input SeriesInstanceUID:", series_uid)
    print("Output folder:", out_folder)
    print("Output SeriesInstanceUID:", new_series_uid)
    print("Slices written:", out_img3d.GetSize()[2])


if __name__ == "__main__":
    main()