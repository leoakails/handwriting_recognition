"""
產出報告用的圖表與指標（需先執行 prepare_data.py 與 train.py）。

輸出到 figures/：
  01_class_counts.png      各類別樣本數（設上限前後）
  02_samples.png           各類別隨機樣本
  03_augmentation.png      資料增強示意
  04_learning_curves.png   訓練／驗證的損失與準確率曲線（依書寫者切分，第 1 折）
  05_confusion_matrix.png  混淆矩陣（依書寫者切分，5 折合併）
  06_per_class_f1.png      各類別精確率、召回率、F1
  07_split_comparison.png  隨機切分 vs 依書寫者切分
  08_misclassified.png     辨識錯誤的案例
  09_correct_examples.png  辨識正確的案例
  10_feature_maps.png      第一層卷積的特徵圖
另輸出 results/metrics_summary.json。
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from model import SymbolCNN
from train import augment

FIG = Path("figures")
RES = Path("results")

# 配色（色盲友善的分類色；文字使用中性墨色，不用系列色）
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

# 依作業系統挑一個已安裝的繁體中文字型（Linux / Windows / macOS）
_installed = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
CJK_FONT = next(
    (f for f in ["Noto Sans CJK TC", "Microsoft JhengHei", "PingFang TC", "Heiti TC"] if f in _installed),
    "sans-serif",
)

plt.rcParams.update(
    {
        "font.family": [CJK_FONT, "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
        "axes.titlecolor": INK,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "figure.dpi": 150,
    }
)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    print("已輸出", FIG / name)


def show_img(ax, img, title=None, color=INK):
    ax.imshow(img, cmap="gray_r", vmin=0, vmax=1)
    ax.set_xticks([]), ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(GRID)
    if title:
        ax.set_title(title, fontsize=10, fontweight="normal", color=color)


def main():
    FIG.mkdir(exist_ok=True)
    d = np.load("data/dataset.npz")
    X = d["X"].astype(np.float32) / 255.0
    y, writers = d["y"], d["writer"]
    names = [str(c) for c in d["class_names"]]
    nc = len(names)
    stats = json.loads(Path("data/dataset_stats.json").read_text(encoding="utf-8"))
    cv = {m: json.loads((RES / f"cv_{m}.json").read_text()) for m in ["writer", "random"]}
    oof = np.load(RES / "oof_writer.npz")
    probs = oof["probs"]
    pred = probs.argmax(1)
    rng = np.random.default_rng(0)

    # 01 各類別樣本數
    before = [stats["count_before_cap"][c] for c in names]
    after = [stats["count_after_cap"][c] for c in names]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(nc)
    ax.bar(x - 0.2, before, 0.38, color=GRID, label="HASYv2 原始張數")
    ax.bar(x + 0.2, after, 0.38, color=BLUE, label="每人每類上限 10 張後")
    ax.set_yscale("log")
    ax.set_xticks(x, names, fontsize=12)
    ax.set_ylabel("張數（對數刻度）")
    ax.set_title("各類別樣本數：「×」「÷」原本幾乎都來自同一位書寫者")
    for i, (b, a) in enumerate(zip(before, after)):
        if b != a:
            ax.annotate(f"{b:,}", (i - 0.2, b), ha="center", va="bottom", fontsize=9, color=INK2)
        ax.annotate(str(a), (i + 0.2, a), ha="center", va="bottom", fontsize=9, color=INK)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    save(fig, "01_class_counts.png")

    # 02 各類別隨機樣本
    k = 8
    fig, axes = plt.subplots(nc, k, figsize=(k * 0.85, nc * 0.9))
    for c in range(nc):
        idx = rng.choice(np.where(y == c)[0], k, replace=False)
        for j, i in enumerate(idx):
            show_img(axes[c, j], X[i])
        axes[c, 0].set_ylabel(names[c], rotation=0, labelpad=14, fontsize=13, color=INK, va="center")
    fig.suptitle("資料集樣本（每列一類，每張來自不同書寫者或不同筆跡）", fontsize=13, fontweight="bold")
    save(fig, "02_samples.png")

    # 03 資料增強示意
    torch.manual_seed(1)
    picks = [np.where(y == names.index(s))[0][0] for s in ["2", "7", "+", "÷"]]
    fig, axes = plt.subplots(len(picks), 6, figsize=(7.5, 5.4))
    for row, i in enumerate(picks):
        show_img(axes[row, 0], X[i], "原圖" if row == 0 else None)
        aug = augment(torch.from_numpy(np.repeat(X[i][None, None], 5, 0))).numpy()[:, 0]
        for j in range(5):
            show_img(axes[row, j + 1], aug[j], f"增強 {j + 1}" if row == 0 else None)
    fig.suptitle("資料增強：±10° 旋轉、0.9–1.1 倍縮放、±2 像素平移", fontsize=13, fontweight="bold")
    save(fig, "03_augmentation.png")

    # 04 學習曲線
    h = cv["writer"]["folds"][0]["history"]
    ep = np.arange(1, len(h["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title in [(axes[0], "loss", "損失（越低越好）"), (axes[1], "acc", "準確率")]:
        ax.plot(ep, h[f"train_{key}"], color=BLUE, lw=2, label="訓練（含資料增強）")
        ax.plot(ep, h[f"val_{key}"], color=ORANGE, lw=2, label="驗證（未見過的書寫者）")
        ax.axvline(h["best_epoch"], color=INK2, lw=1, ls="--")
        ax.set_xlabel("Epoch")
        ax.set_title(title)
        ax.legend()
    axes[0].annotate(
        f"採用第 {h['best_epoch']} 輪權重\n（驗證損失最低）",
        (h["best_epoch"], max(h["val_loss"]) * 0.6),
        xytext=(-10, 0), textcoords="offset points", ha="right", fontsize=9, color=INK2,
    )
    axes[1].set_ylim(0, 1.02)
    fig.suptitle("學習曲線（依書寫者切分，第 1 折）", fontsize=14, fontweight="bold")
    save(fig, "04_learning_curves.png")

    # 05 混淆矩陣（列 = 真實類別，以列正規化成召回率）
    cm = confusion_matrix(y, pred, labels=range(nc))
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    for i in range(nc):
        for j in range(nc):
            if cm[i, j]:
                ax.text(
                    j, i, str(cm[i, j]), ha="center", va="center", fontsize=9,
                    color="white" if cmn[i, j] > 0.5 else INK,
                )
    ax.set_xticks(range(nc), names, fontsize=11)
    ax.set_yticks(range(nc), names, fontsize=11)
    ax.set_xlabel("模型預測")
    ax.set_ylabel("真實類別")
    acc = cv["writer"]["overall_accuracy"]
    ax.set_title(f"混淆矩陣（依書寫者 5 折交叉驗證，共 {len(y)} 張，準確率 {acc:.1%}）", fontsize=12)
    save(fig, "05_confusion_matrix.png")

    # 06 各類別指標
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=range(nc), zero_division=0)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    w = 0.27
    ax.bar(x - w, p, w * 0.92, color=BLUE, label="精確率 Precision")
    ax.bar(x, r, w * 0.92, color=ORANGE, label="召回率 Recall")
    ax.bar(x + w, f, w * 0.92, color=AQUA, label="F1")
    ax.set_xticks(x, names, fontsize=12)
    ax.set_ylim(0.7, 1.0)
    worst = int(np.argmin(f))
    ax.set_title(
        f"各類別辨識表現（依書寫者切分）：「{names[worst]}」最弱，"
        f"資料只來自 {stats['writers_per_class'][names[worst]]} 位書寫者"
    )
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    save(fig, "06_per_class_f1.png")

    # 07 切分方式比較
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, (m, label, color) in enumerate(
        [("random", "隨機切分\n（同一人可能同時在訓練與測試）", ORANGE),
         ("writer", "依書寫者切分\n（測試者完全沒出現在訓練）", BLUE)]
    ):
        accs = [fd["accuracy"] for fd in cv[m]["folds"]]
        ax.scatter(np.full(5, i) + rng.uniform(-0.06, 0.06, 5), accs, s=50, color=color,
                   edgecolor=SURFACE, linewidth=1.5, zorder=3)
        mean = np.mean(accs)
        ax.hlines(mean, i - 0.22, i + 0.22, color=INK, lw=2, zorder=4)
        ax.annotate(f"平均 {mean:.1%}", (i + 0.25, mean), va="center", fontsize=10, color=INK)
        ax.text(i, 0.935, label, ha="center", va="bottom", fontsize=9.5, color=INK2)
    ax.set_xlim(-0.6, 1.8)
    ax.set_xticks([])
    ax.set_ylim(0.93, 1.0)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("各折測試準確率")
    ax.set_title("切分方式比較：遇到新筆跡時準確率略降")
    ax.grid(axis="x", visible=False)
    save(fig, "07_split_comparison.png")

    # 08 錯誤案例（全部列出，最多 24 張）
    wrong = np.where(pred != y)[0]
    wrong = wrong[np.argsort(-probs[wrong].max(1))][:24]
    cols = 8
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.3, rows * 1.95))
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, wrong):
        ax.axis("on")
        show_img(ax, X[i], f"真 {names[y[i]]} → 判 {names[pred[i]]}\n信心 {probs[i].max():.0%}", color=INK)
        ax.title.set_fontsize(9)
    fig.suptitle(f"辨識錯誤的案例（共 {int((pred != y).sum())} 張，依模型信心排序）",
                 fontsize=13, fontweight="bold")
    save(fig, "08_misclassified.png")

    # 09 正確案例（每類一張）
    fig, axes = plt.subplots(2, 7, figsize=(10, 3.4))
    for c, ax in zip(range(nc), axes.flat):
        ok = np.where((y == c) & (pred == c))[0]
        i = rng.choice(ok)
        show_img(ax, X[i], f"判 {names[c]}（{probs[i, c]:.0%}）")
    fig.suptitle("辨識正確的案例（每類隨機一張）", fontsize=13, fontweight="bold")
    save(fig, "09_correct_examples.png")

    # 10 第一層卷積特徵圖
    ckpt = torch.load("models/symbol_cnn.pt", weights_only=False)
    model = SymbolCNN(nc)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    sample = [np.where(y == names.index(s))[0][3] for s in ["+", "×", "7"]]
    with torch.no_grad():
        fmap = model.features[:3](torch.from_numpy(X[sample][:, None])).numpy()
    chans = np.argsort(-fmap.mean((0, 2, 3)))[:7]
    fig, axes = plt.subplots(len(sample), 8, figsize=(10.5, 4.2))
    for row, i in enumerate(sample):
        show_img(axes[row, 0], X[i], "輸入" if row == 0 else None)
        for j, ch in enumerate(chans):
            ax = axes[row, j + 1]
            ax.imshow(fmap[row, ch], cmap="viridis")
            ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
            if row == 0:
                ax.set_title(f"濾鏡 {ch}", fontsize=10, fontweight="normal")
    fig.suptitle("第一層卷積的特徵圖：不同濾鏡對不同方向的筆畫有反應", fontsize=13, fontweight="bold")
    save(fig, "10_feature_maps.png")

    # 指標摘要
    summary = {
        "dataset": {k: stats[k] for k in ["total_images", "total_writers", "image_shape"]},
        "writer_split": {k: cv["writer"][k] for k in cv["writer"] if k != "folds"},
        "random_split": {k: cv["random"][k] for k in cv["random"] if k != "folds"},
        "per_class_writer_split": {
            names[c]: {"precision": float(p[c]), "recall": float(r[c]), "f1": float(f[c]),
                       "support": int(s[c])}
            for c in range(nc)
        },
        "top_confusions": [
            {"true": names[i], "pred": names[j], "count": int(cm[i, j])}
            for i, j in sorted(
                [(i, j) for i in range(nc) for j in range(nc) if i != j and cm[i, j]],
                key=lambda t: -cm[t],
            )[:8]
        ],
    }
    (RES / "metrics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("已輸出", RES / "metrics_summary.json")


if __name__ == "__main__":
    main()
