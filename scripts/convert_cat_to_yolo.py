import os, glob, pathlib
import cv2
import numpy as np

SRC = "data/kaggle"                     # recurse from here (e.g., data/kaggle/cats/CAT_00/...)
IMG_OUT = "data/raw_kaggle/cats"        # detector images
LBL_OUT = "data/labels/raw_kaggle/cats" # YOLO txt labels (class 0)

pathlib.Path(IMG_OUT).mkdir(parents=True, exist_ok=True)
pathlib.Path(LBL_OUT).mkdir(parents=True, exist_ok=True)

def parse_cat_file(path):
    # .cat format: usually "k x1 y1 x2 y2 ..." but sometimes just coords
    toks=[]
    for t in open(path, "r").read().split():
        try:
            toks.append(int(t))
        except:
            pass
    if not toks:
        return None
    # try "k + coords"
    if len(toks) >= 3:
        k = toks[0]
        coords = toks[1:]
        if len(coords) < 2 * k:
            # fall back: treat all as coords
            k = len(toks) // 2
            coords = toks[:2 * k]
    else:
        coords = toks
        k = len(coords) // 2
    pts = np.array(coords[: 2 * k], dtype=np.float32).reshape(-1, 2)
    return pts

def box_from_points(pts, w, h, pad=0.10):
    x1, y1 = pts.min(0); x2, y2 = pts.max(0)
    bw, bh = x2 - x1, y2 - y1
    x1 -= pad * bw; y1 -= pad * bh
    x2 += pad * bw; y2 += pad * bh
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w - 1, x2); y2 = min(h - 1, y2)
    return float(x1), float(y1), float(x2), float(y2)

def to_yolo(x1, y1, x2, y2, w, h):
    cx = (x1 + x2) / 2.0 / w
    cy = (y1 + y2) / 2.0 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return cx, cy, bw, bh

def find_image_for_cat(cat_path):
    # .cat files are like ".../000123.jpg.cat" → image is the same path without the last ".cat"
    base = os.path.splitext(cat_path)[0]   # strips only the last extension → ends with .jpg/.png
    if os.path.exists(base):
        return base
    # Fallbacks: try common extensions and also strip a possible double extension
    candidates = [base] + [base + ext for ext in (".jpg",".JPG",".jpeg",".png",".PNG")]
    # If base already contains ".jpg" before ".cat", also try removing that extra ".jpg"
    dbl = os.path.splitext(base)[0]        # remove a second extension if present
    candidates += [dbl, dbl + ".jpg", dbl + ".jpeg", dbl + ".png"]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def main():
    ok = skip = 0
    for cat_path in glob.glob(os.path.join(SRC, "**", "*.cat"), recursive=True):
        img_path = find_image_for_cat(cat_path)
        if img_path is None:
            skip += 1
            continue
        im = cv2.imread(img_path)
        if im is None:
            skip += 1
            continue
        h, w = im.shape[:2]
        pts = parse_cat_file(cat_path)
        if pts is None or pts.size == 0:
            skip += 1
            continue
        x1, y1, x2, y2 = box_from_points(pts, w, h, pad=0.10)
        cx, cy, bw, bh = to_yolo(x1, y1, x2, y2, w, h)

        base_name = os.path.basename(img_path)
        out_img = os.path.join(IMG_OUT, base_name)
        out_lbl = os.path.join(LBL_OUT, os.path.splitext(base_name)[0] + ".txt")

        if not cv2.imwrite(out_img, im):
            skip += 1
            continue
        with open(out_lbl, "w") as f:
            f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        ok += 1

    print(f"Converted: {ok} | Skipped: {skip}")

if __name__ == "__main__":
    main()
