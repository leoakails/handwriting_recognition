"""
準備資料集：從 HASYv2 取出手寫數字 0–9 與運算符號 +、−、×、÷。

HASYv2（Thoma, 2017）是 32×32 的手寫符號資料集，數字與運算符號都由同一套
流程產生（筆跡 → 黑白點陣圖 → 縮放置中），所以不會出現「數字和符號來自不同
資料集、背景或筆畫粗細不同」造成的來源線索問題。每張圖附有書寫者編號
（user_id），可用來依書寫者切分資料。

資料集沒有「=」，因此本專題只辨識 14 類。

使用方式：
    python prepare_data.py                 # 自動下載 HASYv2 並輸出 data/dataset.npz
    python prepare_data.py --hasy-dir 路徑  # 使用已下載解壓縮的資料夾
"""
import argparse
import csv
import json
import tarfile
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

HASY_URL = "https://zenodo.org/records/259444/files/HASYv2.tar.bz2?download=1"

# HASYv2 中的 LaTeX 標籤 → 本專題使用的顯示名稱（依此順序編號 0–13）
CLASS_MAP = {
    "0": "0", "1": "1", "2": "2", "3": "3", "4": "4",
    "5": "5", "6": "6", "7": "7", "8": "8", "9": "9",
    "+": "+", "-": "−", "\\times": "×", "\\div": "÷",
}
CLASS_NAMES = list(CLASS_MAP.values())

# 同一位書寫者在同一類別最多保留幾張。
# HASYv2 中「×」有 1,396/1,509 張、「÷」有 300/335 張來自同一位書寫者，
# 不設上限的話模型等於在學這個人的筆跡，評估結果也會失真。
MAX_PER_WRITER_PER_CLASS = 10


def download_hasy(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "HASYv2.tar.bz2"
    if not archive.exists():
        print(f"下載 HASYv2（約 34 MB）→ {archive}")
        urllib.request.urlretrieve(HASY_URL, archive)
    if not (root / "hasy-data-labels.csv").exists():
        print("解壓縮中……")
        with tarfile.open(archive) as tar:
            try:
                tar.extractall(root, filter="data")
            except TypeError:  # Python 3.11.3 以前的版本沒有 filter 參數
                tar.extractall(root)
    return root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hasy-dir", type=Path, default=Path("data_raw/hasy"))
    parser.add_argument("--out", type=Path, default=Path("data/dataset.npz"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    hasy_dir = download_hasy(args.hasy_dir)
    rng = np.random.default_rng(args.seed)

    with open(hasy_dir / "hasy-data-labels.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["latex"] in CLASS_MAP]

    # 依（書寫者, 類別）分組，每組最多隨機保留 MAX_PER_WRITER_PER_CLASS 張
    groups = defaultdict(list)
    for r in rows:
        groups[(r["user_id"], r["latex"])].append(r)
    kept = []
    for key in sorted(groups):
        g = groups[key]
        if len(g) > MAX_PER_WRITER_PER_CLASS:
            idx = rng.choice(len(g), MAX_PER_WRITER_PER_CLASS, replace=False)
            g = [g[i] for i in sorted(idx)]
        kept.extend(g)

    latex_list = list(CLASS_MAP.keys())
    images, labels, writers = [], [], []
    for r in kept:
        img = Image.open(hasy_dir / r["path"]).convert("L")
        arr = np.asarray(img, dtype=np.uint8)
        # 原圖為白底黑字；反轉成黑底白字，讓「筆畫 = 1、背景 = 0」
        images.append(255 - arr)
        labels.append(latex_list.index(r["latex"]))
        writers.append(int(r["user_id"]))

    X = np.stack(images)
    y = np.array(labels, dtype=np.int64)
    w = np.array(writers, dtype=np.int64)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, X=X, y=y, writer=w, class_names=np.array(CLASS_NAMES))

    before = Counter(CLASS_MAP[r["latex"]] for r in rows)
    after = Counter(CLASS_NAMES[i] for i in y)
    n_writers = {
        CLASS_MAP[k]: len({r["user_id"] for r in rows if r["latex"] == k}) for k in CLASS_MAP
    }
    stats = {
        "classes": CLASS_NAMES,
        "count_before_cap": {c: before[c] for c in CLASS_NAMES},
        "count_after_cap": {c: after[c] for c in CLASS_NAMES},
        "writers_per_class": n_writers,
        "total_images": int(len(y)),
        "total_writers": int(len(set(w.tolist()))),
        "image_shape": list(X.shape[1:]),
        "max_per_writer_per_class": MAX_PER_WRITER_PER_CLASS,
    }
    (args.out.parent / "dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"輸出 {args.out}：{len(y)} 張、{stats['total_writers']} 位書寫者、影像 {X.shape[1:]}")
    print(f"{'類別':<4}{'原始':>6}{'上限後':>8}{'書寫者':>8}")
    for c in CLASS_NAMES:
        print(f"{c:<5}{before[c]:>6}{after[c]:>8}{n_writers[c]:>8}")


if __name__ == "__main__":
    main()
