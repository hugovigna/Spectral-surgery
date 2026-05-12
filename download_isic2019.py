"""
download_isic2019.py
--------------------
Télécharge ISIC 2019 Training Set (~9 Go) via Kaggle API.

Prérequis :
    1. Compte Kaggle + token API dans ~/.kaggle/kaggle.json
    2. pip3 install kaggle

Usage :
    python3 download_isic2019.py
"""

import os
import sys
import zipfile
import pathlib

DEST = pathlib.Path("data/isic2019")

def check_kaggle_credentials():
    cred = pathlib.Path.home() / ".kaggle" / "kaggle.json"
    if not cred.exists():
        print("ERREUR : credentials Kaggle absents.")
        print()
        print("  1. Va sur https://www.kaggle.com/settings")
        print("  2. Section 'API' → 'Create New Token' → télécharge kaggle.json")
        print("  3. Dans ton terminal :")
        print("       mkdir -p ~/.kaggle")
        print("       mv ~/Downloads/kaggle.json ~/.kaggle/")
        print("       chmod 600 ~/.kaggle/kaggle.json")
        print()
        sys.exit(1)

def download_via_kaggle(dest: pathlib.Path):
    import kaggle  # noqa — déclenche l'auth automatique
    dest.mkdir(parents=True, exist_ok=True)

    print("[1] Téléchargement du dataset ISIC 2019 depuis Kaggle ...")
    print("    Dataset : andrewmvd/isic-2019  (~9 Go, patience)")
    os.system(
        f"/Library/Frameworks/Python.framework/Versions/3.13/bin/kaggle "
        f"datasets download -d andrewmvd/isic-2019 -p {dest} --unzip"
    )

def verify(dest: pathlib.Path):
    img_dir = dest / "ISIC_2019_Training_Input"
    gt_csv  = dest / "ISIC_2019_Training_GroundTruth.csv"

    if not img_dir.exists():
        # cherche un sous-dossier alternatif
        subdirs = [d for d in dest.iterdir() if d.is_dir()]
        print(f"    Sous-dossiers trouvés : {[d.name for d in subdirs]}")
    else:
        n = len(list(img_dir.glob("*.jpg")))
        print(f"    Images trouvées : {n:,}  (attendu ~25 331)")

    if gt_csv.exists():
        import pandas as pd
        gt = pd.read_csv(gt_csv)
        print(f"    Classes : {list(gt.columns[1:])}")
        print(f"    Distribution :")
        for col in gt.columns[1:]:
            print(f"      {col:4s} : {int(gt[col].sum()):5d}")
    else:
        print(f"    Fichier ground truth non trouvé dans {dest}")


if __name__ == "__main__":
    check_kaggle_credentials()

    if DEST.exists() and len(list((DEST / "ISIC_2019_Training_Input").glob("*.jpg"))) > 1000:
        print(f"Dataset déjà présent dans {DEST}/")
    else:
        download_via_kaggle(DEST)

    print("\n[2] Vérification ...")
    verify(DEST)
    print("\nDataset prêt. Lance maintenant :")
    print("    python3 train_resnet50_isic2019.py")
