import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, ".")
from src.hand_tip import TipDetector

files = sorted(glob.glob("data/tip_rec/*.jpg"))
out = "data/tip_debug"
os.makedirs(out, exist_ok=True)
det = TipDetector()
tiles = []
for i, fn in enumerate(files):
    t = int(os.path.basename(fn)[:6]) / 1000.0
    img = cv2.imread(fn)
    d = det.detect(img, t)
    if i % 6 == 0:
        v = img.copy()
        if d is not None:
            cv2.circle(v, (int(d.x), int(d.y)), 14, (0, 0, 255), 3)
            cv2.circle(v, (int(d.entry[0]), int(d.entry[1])), 14, (255, 0, 0), 3)
            cv2.line(v, (int(d.x), int(d.y)), (int(d.entry[0]), int(d.entry[1])), (255, 0, 0), 2)
        cv2.putText(v, f"{t:.1f}s bg={det.has_background}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        m = det.mask if det.mask is not None else np.zeros((10, 10), np.uint8)
        m = cv2.cvtColor(cv2.resize(m, (v.shape[1], v.shape[0]), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
        tiles.append(cv2.resize(np.hstack([v, m]), None, fx=0.25, fy=0.25))
    print(f"{t:6.2f}s bg={det.has_background} det={'-' if d is None else (round(d.x), round(d.y), round(d.area))}")
rows = [np.hstack(tiles[k:k + 5] + [np.zeros_like(tiles[0])] * (5 - len(tiles[k:k + 5]))) for k in range(0, len(tiles), 5)]
for j in range(0, len(rows), 3):
    cv2.imwrite(f"{out}/sheet_{j // 3}.jpg", np.vstack(rows[j:j + 3]))
