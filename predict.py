"""
辨識單張圖片中的一個手寫符號（需先執行 train.py 產生 models/symbol_cnn.pt）。

    python predict.py 我的字.png
    python predict.py 我的字.png --show     # 另存前處理後的 32×32 圖，方便檢查

前處理盡量貼近 HASYv2 的格式：轉灰階 → 判斷是否需要反色 → 二值化 →
裁掉留白 → 依長邊等比例縮放到 32×32 並置中。
注意：模型是用 HASYv2 的筆跡訓練的，手機照片若有陰影、格線或多個符號，
結果不一定可靠；請一張圖只寫一個符號、白紙黑字、光線均勻。
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from model import SymbolCNN

SIZE = 32


def otsu_threshold(gray: np.ndarray) -> float:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    omega, mu = np.cumsum(p), np.cumsum(p * np.arange(256))
    sigma_b = (mu[-1] * omega - mu) ** 2 / (omega * (1 - omega) + 1e-12)
    return float(np.argmax(sigma_b))


def preprocess(path: Path) -> np.ndarray:
    gray = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    t = otsu_threshold(gray)
    ink = gray <= t  # 暗色像素當作筆畫（Otsu 門檻以下）
    if ink.mean() > 0.5:  # 筆畫比背景多，代表是深色背景、淺色字
        ink = ~ink
    ys, xs = np.nonzero(ink)
    if len(ys) == 0:
        raise ValueError("圖片中找不到筆畫")
    crop = ink[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]

    # 依長邊等比例縮放，HASYv2 的符號會撐滿 32×32 的其中一邊
    h, w = crop.shape
    scale = SIZE / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    img = Image.fromarray((crop * 255).astype(np.uint8)).resize((nw, nh), Image.BILINEAR)
    canvas = np.zeros((SIZE, SIZE), dtype=np.float32)
    top, left = (SIZE - nh) // 2, (SIZE - nw) // 2
    canvas[top : top + nh, left : left + nw] = np.asarray(img, dtype=np.float32) / 255.0
    return (canvas > 0.3).astype(np.float32)  # HASYv2 是純黑白圖，這裡也二值化


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=Path("models/symbol_cnn.pt"))
    parser.add_argument("--show", action="store_true", help="另存前處理後的圖片")
    args = parser.parse_args()

    ckpt = torch.load(args.model, weights_only=False)
    names = ckpt["class_names"]
    model = SymbolCNN(len(names))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    x = preprocess(args.image)
    if args.show:
        out = args.image.with_name(args.image.stem + "_32x32.png")
        Image.fromarray(((1 - x) * 255).astype(np.uint8)).resize((128, 128), Image.NEAREST).save(out)
        print(f"前處理結果已存到 {out}")

    with torch.no_grad():
        prob = F.softmax(model(torch.from_numpy(x)[None, None]), 1)[0].numpy()
    top = np.argsort(-prob)[:3]
    print(f"辨識結果：{names[top[0]]}（信心 {prob[top[0]]:.1%}）")
    print("前三名：" + "、".join(f"{names[i]} {prob[i]:.1%}" for i in top))


if __name__ == "__main__":
    main()
