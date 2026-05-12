"""
resplit_isic2019.py
--------------------
Re-split les données ISIC-2019 pour augmenter le nombre d'images
dans le test set des classes rares (VASC, DF), afin d'obtenir des
intervalles de confiance interprétables après TTA.

Stratégie :
  - Combine train.npz + val_ss.npz + test_ss.npz en un pool global par classe.
  - Test   : max(50, 20% du pool de la classe)
  - Val    : 10% du pool (min 20)
  - Train  : le reste (source du sensitivity set et du HVP batch au runtime)
  - Sensitivity cap recommandé dans CONFIG : 50/classe (au lieu de 250)

Usage :
    python3.12 resplit_isic2019.py
    python3.12 resplit_isic2019.py --test-min 75 --val-frac 0.10

Sorties :
    data/isic2019_cache/train_v2.npz
    data/isic2019_cache/val_v2.npz
    data/isic2019_cache/test_v2.npz
"""

import argparse
import numpy as np
import os

CLASSES   = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
CACHE_DIR = "data/isic2019_cache"

def load_npz(name):
    path = os.path.join(CACHE_DIR, name)
    d = np.load(path)
    return d["imgs"], d["labels"].astype(np.int32)

def main(test_min: int, test_frac: float, val_frac: float, seed: int):
    rng = np.random.default_rng(seed)

    print("[1] Chargement des caches existants ...")
    imgs_list, labels_list = [], []
    for name in ["train.npz", "val_ss.npz", "test_ss.npz"]:
        x, y = load_npz(name)
        imgs_list.append(x)
        labels_list.append(y)
        print(f"    {name}: {len(y)} imgs")

    x_all = np.concatenate(imgs_list, axis=0)
    y_all = np.concatenate(labels_list, axis=0)
    print(f"    Total : {len(y_all)} imgs\n")

    test_idx, val_idx, train_idx = [], [], []

    print("[2] Re-split par classe :")
    print(f"  {'Classe':6s}  {'total':>6s}  {'test':>5s}  {'val':>5s}  {'train':>6s}")
    for c, name in enumerate(CLASSES):
        idx = np.where(y_all == c)[0]
        n   = len(idx)
        rng.shuffle(idx)

        n_test  = max(test_min, int(np.floor(n * test_frac)))
        n_test  = min(n_test, n)
        n_val   = max(20, int(np.floor(n * val_frac)))
        n_val   = min(n_val, n - n_test)
        n_train = n - n_test - n_val

        test_idx .extend(idx[:n_test].tolist())
        val_idx  .extend(idx[n_test:n_test + n_val].tolist())
        train_idx.extend(idx[n_test + n_val:].tolist())

        print(f"  {name:6s}  {n:>6d}  {n_test:>5d}  {n_val:>5d}  {n_train:>6d}")

    test_idx  = np.array(test_idx)
    val_idx   = np.array(val_idx)
    train_idx = np.array(train_idx)

    print(f"\n  Total → test:{len(test_idx)}  val:{len(val_idx)}  train:{len(train_idx)}")

    print("\n[3] Sauvegarde ...")
    os.makedirs(CACHE_DIR, exist_ok=True)
    for split, idx, fname in [
        ("test",  test_idx,  "test_v2.npz"),
        ("val",   val_idx,   "val_v2.npz"),
        ("train", train_idx, "train_v2.npz"),
    ]:
        path = os.path.join(CACHE_DIR, fname)
        np.savez_compressed(path,
                            imgs=x_all[idx],
                            labels=y_all[idx])
        size_mb = x_all[idx].nbytes / 1e6
        print(f"    {fname}: {len(idx)} imgs  ({size_mb:.0f} MB)")

    print("\n[4] Distribution du test set :")
    y_test = y_all[test_idx]
    for c, name in enumerate(CLASSES):
        n = (y_test == c).sum()
        # CI width approximation (Wilson 95%): 1.96 * sqrt(p(1-p)/n) ≈ 0.25/sqrt(n)
        ci_half = 1.96 * 0.5 / np.sqrt(n) * 100
        print(f"  {name:6s}: {n:>4d} imgs  CI±≈{ci_half:.1f}%")

    print("\nDone. Pour utiliser dans spectral_surgery.py :")
    print('  "cache_val"  : "data/isic2019_cache/val_v2.npz"')
    print('  "cache_test" : "data/isic2019_cache/test_v2.npz"')
    print('  # Le train_v2.npz remplace train.npz pour le sensitivity set')
    print('  "sens_max_per_class": 50  # recommandé avec ce re-split')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-min",  type=int,   default=50,
                        help="Nombre minimum d'images test par classe (défaut: 50)")
    parser.add_argument("--test-frac", type=float, default=0.20,
                        help="Fraction du pool allouée au test (défaut: 0.20)")
    parser.add_argument("--val-frac",  type=float, default=0.10,
                        help="Fraction du pool allouée à la validation (défaut: 0.10)")
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()
    main(args.test_min, args.test_frac, args.val_frac, args.seed)
