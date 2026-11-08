"""
訓練與評估 CNN。

資料只有約 1,200 張，單次切出的測試集太小（「÷」可能只剩兩三位書寫者），
所以採 5 折交叉驗證：每張圖都恰好在某一折當過一次測試資料，把五折的預測
合起來就能得到涵蓋全部資料的混淆矩陣與指標。

兩種切分方式都會跑，用來比較：
  writer：依書寫者分組切分，測試折的書寫者完全沒在訓練中出現（主要結果）
  random：隨機切分，同一人的筆跡可能同時出現在訓練與測試（對照組）

每一折內再從訓練資料切出約 1/6 當驗證集，用來畫學習曲線與挑選最佳 epoch；
資料增強只套用在訓練批次上，驗證與測試資料一律使用原圖。

最後用全部資料（同樣保留一部分書寫者當驗證）訓練一個最終模型，存成
models/symbol_cnn.pt，供 predict.py 使用。

使用方式：
    python train.py               # 預設 40 epochs
    python train.py --epochs 20   # 快速試跑
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from torch import nn

from model import SymbolCNN, count_parameters


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_dataset(path: Path):
    d = np.load(path)
    X = d["X"].astype(np.float32)[:, None, :, :] / 255.0  # (N, 1, 32, 32)，像素正規化到 0–1
    return X, d["y"], d["writer"], [str(c) for c in d["class_names"]]


def augment(x: torch.Tensor, max_deg=10.0, scale=(0.9, 1.1), max_shift_px=2.0) -> torch.Tensor:
    """
    隨機小幅旋轉、縮放、平移（只用在訓練批次）。
    旋轉上限刻意設在 ±10°：「+」轉 45° 就變成「×」，角度太大會把類別改掉。
    也不做左右翻轉，翻轉後「2」「5」「7」等數字會變成不存在的字形。
    """
    n = x.shape[0]
    angle = (torch.rand(n) * 2 - 1) * math.radians(max_deg)
    s = torch.empty(n).uniform_(*scale)
    shift = (torch.rand(n, 2) * 2 - 1) * (max_shift_px / (x.shape[-1] / 2))
    cos, sin = torch.cos(angle) / s, torch.sin(angle) / s
    theta = torch.stack(
        [torch.stack([cos, -sin, shift[:, 0]], 1), torch.stack([sin, cos, shift[:, 1]], 1)], 1
    )
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")


def class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    """類別權重 ∝ 1 / 樣本數，讓樣本較少的「÷」在損失中不被忽略。"""
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    w = counts.sum() / (num_classes * np.maximum(counts, 1))
    return torch.tensor(w)


@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray, batch=256) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        out.append(F.softmax(model(torch.from_numpy(X[i : i + batch])), 1).numpy())
    return np.concatenate(out)


def train_model(Xtr, ytr, Xva, yva, num_classes, epochs, seed, lr=1e-3, batch=64):
    set_seed(seed)
    model = SymbolCNN(num_classes)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(ytr, num_classes))
    Xtr_t, ytr_t = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    Xva_t, yva_t = torch.from_numpy(Xva), torch.from_numpy(yva)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_state, best_val = None, float("inf")
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        tot_loss, tot_correct = 0.0, 0
        for i in range(0, len(perm), batch):
            idx = perm[i : i + batch]
            xb, yb = augment(Xtr_t[idx]), ytr_t[idx]
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += loss.item() * len(idx)
            tot_correct += (logits.argmax(1) == yb).sum().item()
        sched.step()

        model.eval()
        with torch.no_grad():
            logits = model(Xva_t)
            val_loss = loss_fn(logits, yva_t).item()
            val_acc = (logits.argmax(1) == yva_t).float().mean().item()
        history["train_loss"].append(tot_loss / len(perm))
        history["train_acc"].append(tot_correct / len(perm))
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        # 保留驗證損失最低的那一輪權重（等同提前停止，但不必猜要停在第幾輪）
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            history["best_epoch"] = epoch + 1

    model.load_state_dict(best_state)
    return model, history


def make_splits(mode, y, writers, n_splits, seed):
    if mode == "writer":
        return StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed), writers
    return StratifiedKFold(n_splits, shuffle=True, random_state=seed), None


def first_split(mode, y, writers, n_splits, seed):
    splitter, groups = make_splits(mode, y, writers, n_splits, seed)
    return next(splitter.split(np.zeros(len(y)), y, groups))


def cross_validate(mode, X, y, writers, num_classes, epochs, seed, out_dir: Path):
    splitter, groups = make_splits(mode, y, writers, 5, seed)
    probs = np.zeros((len(y), num_classes), dtype=np.float32)
    fold_id = np.zeros(len(y), dtype=np.int64)
    folds = []
    for k, (tr, te) in enumerate(splitter.split(np.zeros(len(y)), y, groups)):
        t0 = time.time()
        # 訓練折內再切 1/6 當驗證集（writer 模式同樣依書寫者切）
        sub_tr, sub_va = first_split(mode, y[tr], writers[tr], 6, seed + k)
        tr_idx, va_idx = tr[sub_tr], tr[sub_va]
        model, hist = train_model(
            X[tr_idx], y[tr_idx], X[va_idx], y[va_idx], num_classes, epochs, seed + k
        )
        p = predict_proba(model, X[te])
        probs[te], fold_id[te] = p, k
        pred = p.argmax(1)
        fold = {
            "fold": k + 1,
            "n_train": int(len(tr_idx)),
            "n_val": int(len(va_idx)),
            "n_test": int(len(te)),
            "test_writers": int(len(set(writers[te].tolist()))),
            "train_test_writer_overlap": int(
                len(set(writers[tr].tolist()) & set(writers[te].tolist()))
            ),
            "accuracy": float(accuracy_score(y[te], pred)),
            "macro_f1": float(f1_score(y[te], pred, average="macro")),
            "history": hist,
        }
        folds.append(fold)
        print(
            f"[{mode}] fold {k + 1}: acc={fold['accuracy']:.3f} macro-F1={fold['macro_f1']:.3f} "
            f"best epoch={hist['best_epoch']} 書寫者重疊={fold['train_test_writer_overlap']} "
            f"({time.time() - t0:.0f}s)"
        )

    pred = probs.argmax(1)
    summary = {
        "mode": mode,
        "overall_accuracy": float(accuracy_score(y, pred)),
        "overall_macro_f1": float(f1_score(y, pred, average="macro")),
        "fold_accuracy_mean": float(np.mean([f["accuracy"] for f in folds])),
        "fold_accuracy_std": float(np.std([f["accuracy"] for f in folds])),
        "fold_macro_f1_mean": float(np.mean([f["macro_f1"] for f in folds])),
        "fold_macro_f1_std": float(np.std([f["macro_f1"] for f in folds])),
        "folds": folds,
    }
    (out_dir / f"cv_{mode}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(out_dir / f"oof_{mode}.npz", probs=probs, y=y, fold=fold_id, writer=writers)
    print(
        f"[{mode}] 整體 acc={summary['overall_accuracy']:.3f} "
        f"macro-F1={summary['overall_macro_f1']:.3f} "
        f"（各折 {summary['fold_accuracy_mean']:.3f} ± {summary['fold_accuracy_std']:.3f}）"
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/dataset.npz"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    args = parser.parse_args()

    torch.set_num_threads(max(1, torch.get_num_threads()))
    args.results.mkdir(parents=True, exist_ok=True)
    args.models.mkdir(parents=True, exist_ok=True)
    X, y, writers, class_names = load_dataset(args.data)
    nc = len(class_names)
    print(f"資料 {len(y)} 張、{nc} 類；模型參數 {count_parameters(SymbolCNN(nc)):,} 個")

    for mode in ["writer", "random"]:
        cross_validate(mode, X, y, writers, nc, args.epochs, args.seed, args.results)

    # 最終模型：用全部資料訓練，保留約 1/6 書寫者當驗證集挑選最佳 epoch
    tr, va = first_split("writer", y, writers, 6, args.seed)
    model, hist = train_model(X[tr], y[tr], X[va], y[va], nc, args.epochs, args.seed)
    torch.save(
        {"state_dict": model.state_dict(), "class_names": class_names, "history": hist},
        args.models / "symbol_cnn.pt",
    )
    print(f"最終模型已存到 {args.models / 'symbol_cnn.pt'}（best epoch {hist['best_epoch']}）")


if __name__ == "__main__":
    main()
