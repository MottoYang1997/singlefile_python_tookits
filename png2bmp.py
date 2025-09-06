#!/usr/bin/env python3
# png2binbmp.py
# Convert PNG (with optional alpha) to 1-bit (binary) BMP.
# Supports: dithering, fixed threshold, Otsu, inversion, DPI tagging, batch.

import argparse
from pathlib import Path
from typing import Iterable, Tuple, Optional

from PIL import Image, ImageOps

try:
    import numpy as np  # for Otsu
except Exception:
    np = None


def composite_to_opaque(img: Image.Image, bg: str = "white") -> Image.Image:
    """If PNG has alpha, composite onto a solid background (white/black/#RRGGBB)."""
    if img.mode in ("LA", "RGBA"):
        if bg == "white":
            bg_color = (255, 255, 255)
        elif bg == "black":
            bg_color = (0, 0, 0)
        elif isinstance(bg, str) and bg.startswith("#") and len(bg) in (4, 7):
            bg_color = ImageColor.getrgb(bg)  # type: ignore
        else:
            bg_color = (255, 255, 255)
        base = Image.new("RGB", img.size, bg_color)
        base.paste(img.convert("RGBA"), mask=img.getchannel("A"))
        return base
    elif img.mode == "P":
        # Convert palette images to RGBA then composite if they secretly carry transparency
        if "transparency" in img.info:
            return composite_to_opaque(img.convert("RGBA"), bg=bg)
        return img.convert("RGB")
    else:
        return img.convert("RGB")


def otsu_threshold(gray: Image.Image) -> int:
    """Compute Otsu threshold (0-255)."""
    if np is None:
        raise RuntimeError("Otsu method requires numpy. Install with: pip install numpy")
    hist = gray.histogram()  # 256 bins for L mode
    hist = np.array(hist, dtype=np.float64)
    total = hist.sum()
    if total <= 0:
        return 128
    prob = hist / total
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_t = mu[-1]
    sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1 - omega) + 1e-12)
    idx = int(np.nanargmax(sigma_b2))
    return idx


def to_binary_bmp(
    in_path: Path,
    out_path: Path,
    method: str = "dither",
    threshold: int = 128,
    invert: bool = False,
    bg: str = "white",
    dpi: Optional[int] = None,
) -> None:
    """
    Convert one PNG into a 1-bit BMP using the selected method.
    method: dither | threshold | otsu
    """
    img = Image.open(in_path)
    img = composite_to_opaque(img, bg=bg)
    gray = img.convert("L")

    if method == "dither":
        # Floyd–Steinberg dithering to 1-bit
        bin_img = gray.convert("1", dither=Image.FLOYDSTEINBERG)
    elif method == "threshold":
        thr = max(0, min(255, int(threshold)))
        # Create binary via fixed threshold
        bin_img = gray.point(lambda p: 255 if p >= thr else 0, mode="1")
    elif method == "otsu":
        thr = otsu_threshold(gray)
        bin_img = gray.point(lambda p: 255 if p >= thr else 0, mode="1")
    else:
        raise ValueError(f"Unknown method: {method}")

    if invert:
        bin_img = ImageOps.invert(bin_img.convert("L")).convert("1", dither=Image.NONE)

    # Ensure parent dir exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_kwargs = {}
    if dpi is not None:
        # Pillow can embed DPI in BMP as pixels-per-meter; it converts internally.
        save_kwargs["dpi"] = (dpi, dpi)

    bin_img.save(out_path, format="BMP", **save_kwargs)


def expand_inputs(inputs: Iterable[str]) -> Iterable[Path]:
    """Expand files/globs/directories into a flat list of PNG files."""
    paths = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            paths.extend(sorted(p.glob("**/*.png")))
        else:
            # glob pattern or file
            matches = list(Path().glob(item))
            if matches:
                paths.extend(matches)
            else:
                paths.append(p)
    # Filter PNG-ish
    return [p for p in paths if p.suffix.lower() in (".png",)]


def main():
    ap = argparse.ArgumentParser(description="Convert PNG to 1-bit (binary) BMP.")
    ap.add_argument("inputs", nargs="+", help="PNG files, directories, or glob patterns")
    ap.add_argument("-o", "--out", help="Output file (single input only)")
    ap.add_argument("-d", "--out-dir", help="Output directory (for batch)")
    ap.add_argument("--method", choices=["dither", "threshold", "otsu"], default="dither",
                    help="Binarization method (default: dither)")
    ap.add_argument("--threshold", type=int, default=128,
                    help="Threshold 0-255 if --method threshold (default: 128)")
    ap.add_argument("--invert", action="store_true", help="Invert black/white")
    ap.add_argument("--bg", default="white",
                    help="Background for alpha: white|black|#RRGGBB (default: white)")
    ap.add_argument("--dpi", type=int, help="DPI to tag in BMP metadata (optional)")
    ap.add_argument("--suffix", default="_bin", help="Suffix for output filenames (batch)")
    args = ap.parse_args()

    in_files = expand_inputs(args.inputs)
    if not in_files:
        raise SystemExit("No PNG inputs found.")

    # Single-file mode
    if len(in_files) == 1 and args.out:
        out_path = Path(args.out)
        to_binary_bmp(
            in_files[0],
            out_path,
            method=args.method,
            threshold=args.threshold,
            invert=args.invert,
            bg=args.bg,
            dpi=args.dpi,
        )
        print(f"Done: {out_path}")
        return

    # Batch mode
    out_dir = Path(args.out_dir) if args.out_dir else None
    for p in in_files:
        if out_dir:
            out_path = out_dir / (p.stem + args.suffix + ".bmp")
        else:
            out_path = p.with_name(p.stem + args.suffix + ".bmp")
        to_binary_bmp(
            p,
            out_path,
            method=args.method,
            threshold=args.threshold,
            invert=args.invert,
            bg=args.bg,
            dpi=args.dpi,
        )
        print(f"Done: {out_path}")


if __name__ == "__main__":
    main()
