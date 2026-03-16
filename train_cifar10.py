"""
train_cifar10.py
----------------
Fine-tune ResNet50 (poids ImageNet) sur CIFAR-10.

Pourquoi pas de phases séparées ?
  La stratégie classique "geler backbone → entraîner tête → débloquer tout" suppose
  que le backbone produit de bonnes features pour la tâche cible. Ici, ce n'est pas
  le cas : ResNet50 ImageNet est conçu pour des images 224×224. Sur 32×32, le premier
  Conv (stride=2) + MaxPool (stride=2) réduit la résolution à 8×8 dès l'entrée,
  rendant les features du backbone quasi-aléatoires pour CIFAR-10.
  → On entraîne tout le réseau dès le début (ImageNet sert juste d'initialisation).

Performances attendues :
  - 32×32 (rapide)  : ~65-75% val_accuracy en 30 epochs (~20 min CPU)
  - 96×96 (meilleur): ~88-92% val_accuracy en 30 epochs (~2h CPU ou ~15 min GPU)
  Passer '--img-size 96' pour activer le resize.

Usage :
  python train_cifar10.py                      # 32×32, ~20 min CPU
  python train_cifar10.py --img-size 96        # 96×96, bien meilleur
  python train_cifar10.py --epochs 50 --lr 5e-4
"""

import os
import argparse
import time

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


DEFAULTS = {
    "out"        : "resnet50_cifar10.keras",
    "img_size"   : 32,
    "epochs"     : 40,
    "lr"         : 1e-3,
    "weight_decay": 1e-4,   # L2 via AdamW — pénalise les grands poids
    "dropout"    : 0.4,     # dropout avant la tête Dense
    "batch"      : 128,
    "seed"       : 42,
}


# ---------------------------------------------------------------------------
# 1. Chargement et prétraitement CIFAR-10
# ---------------------------------------------------------------------------

def load_cifar10(img_size: int = 32):
    """
    Charge CIFAR-10 (nativement 32×32 RGB uint8) et :
      - Normalise en [0, 1]
      - Redimensionne à img_size×img_size si img_size != 32
    """
    print("[1] Chargement CIFAR-10 ...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    x_train = x_train.astype(np.float32) / 255.0
    x_test  = x_test.astype(np.float32)  / 255.0
    y_train = y_train.squeeze().astype(np.int32)
    y_test  = y_test.squeeze().astype(np.int32)

    # Resize si demandé (utile pour mieux exploiter les poids ImageNet)
    if img_size != 32:
        print(f"    Resize 32 → {img_size} ...")
        x_train = tf.image.resize(x_train, [img_size, img_size]).numpy()
        x_test  = tf.image.resize(x_test,  [img_size, img_size]).numpy()

    print(f"    x_train : {x_train.shape}, plage=[{x_train.min():.2f}, {x_train.max():.2f}]")
    return x_train, y_train, x_test, y_test


# ---------------------------------------------------------------------------
# 2. Construction du modèle
# ---------------------------------------------------------------------------

def build_augmentation(img_size: int = 32) -> tf.keras.Sequential:
    """
    Pipeline de data augmentation appliqué uniquement à l'entraînement.

    Transformations choisies pour CIFAR-10 :
      - RandomFlip horizontal : les objets naturels sont symétriques gauche/droite
      - RandomTranslation ±10% : simule de petits décalages de cadrage
      - RandomContrast ±20%   : simule des variations d'éclairage

    Ces augmentations forcent le réseau à apprendre des features invariantes,
    réduisant l'overfitting sans changer la distribution des labels.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1,
                                          fill_mode="reflect"),
        tf.keras.layers.RandomContrast(factor=0.2),
    ], name="augmentation")


def build_model(img_size: int = 32, dropout: float = 0.4) -> tf.keras.Model:
    """
    ResNet50 (ImageNet) + Dropout + tête Dense(10).

    Régularisations :
      - Dropout(dropout) avant Dense : coupe aléatoirement des activations
        → force le réseau à ne pas sur-dépendre d'un seul neurone
      - Weight decay (AdamW) appliqué à la compilation : pénalise les grands poids
        → poussée vers des minima plus plats (ζ plus faible)

    Tout le réseau est entraînable dès le départ (ImageNet comme init).
    """
    print("[2] Construction du modèle ...")

    augment = build_augmentation(img_size)

    base = tf.keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(img_size, img_size, 3),
        pooling="avg",           # GlobalAveragePooling → vecteur (2048,)
    )
    base.trainable = True

    inputs  = tf.keras.Input(shape=(img_size, img_size, 3))
    x       = augment(inputs, training=True)   # augmentation active en train seulement
    x       = base(x, training=True)
    x       = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="ResNet50_CIFAR10")

    n_params = sum(int(tf.size(v)) for v in model.trainable_variables)
    print(f"    Paramètres entraînables : {n_params:,}")
    print(f"    Dropout : {dropout}  |  Augmentation : flip + translation + contrast")
    return model


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",          default=DEFAULTS["out"])
    parser.add_argument("--img-size",     type=int,   default=DEFAULTS["img_size"])
    parser.add_argument("--epochs",       type=int,   default=DEFAULTS["epochs"])
    parser.add_argument("--lr",           type=float, default=DEFAULTS["lr"])
    parser.add_argument("--weight-decay", type=float, default=DEFAULTS["weight_decay"])
    parser.add_argument("--dropout",      type=float, default=DEFAULTS["dropout"])
    parser.add_argument("--batch",        type=int,   default=DEFAULTS["batch"])
    args = parser.parse_args()

    tf.random.set_seed(DEFAULTS["seed"])
    np.random.seed(DEFAULTS["seed"])

    print(f"\n=== Fine-tuning ResNet50 → CIFAR-10 ({args.img_size}×{args.img_size}) ===")
    print(f"    lr={args.lr}  wd={args.weight_decay}  dropout={args.dropout}  "
          f"epochs={args.epochs}  batch={args.batch}  out={args.out}\n")

    x_train, y_train, x_test, y_test = load_cifar10(args.img_size)
    model = build_model(args.img_size, dropout=args.dropout)

    # AdamW = Adam + weight decay découplé (L2 sur les poids, pas sur les gradients)
    # Pousse vers des minima plus plats que Adam seul.
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=args.out,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        # Divise lr par 2 si val_loss stagne 5 epochs → convergence plus fine
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    out_dir = os.path.splitext(args.out)[0] + "_results"
    os.makedirs(out_dir, exist_ok=True)

    print(f"[3] Entraînement ({args.epochs} epochs, lr={args.lr}) ...")
    t0 = time.time()
    hist = model.fit(
        x_train, y_train,
        batch_size=args.batch,
        epochs=args.epochs,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
    )
    elapsed = time.time() - t0
    print(f"\n    Terminé en {elapsed/60:.1f} min")

    # -- Sauvegarde de l'historique d'entraînement --
    df = pd.DataFrame(hist.history)
    df.index = df.index + 1          # epoch commence à 1
    df.index.name = "epoch"
    csv_path = os.path.join(out_dir, "training_history.csv")
    df.to_csv(csv_path)
    print(f"    Historique sauvegardé : {csv_path}")

    # -- Courbes d'entraînement --
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs_range = df.index.tolist()

    ax = axes[0]
    ax.plot(epochs_range, df["loss"],     label="train loss",     color="steelblue")
    ax.plot(epochs_range, df["val_loss"], label="val loss",       color="crimson", ls="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Loss — ResNet50 / CIFAR-10"); ax.legend(); ax.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.plot(epochs_range, df["accuracy"],     label="train acc",  color="steelblue")
    ax2.plot(epochs_range, df["val_accuracy"], label="val acc",    color="crimson", ls="--")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy — ResNet50 / CIFAR-10"); ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(out_dir, "training_curves.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"    Courbes sauvegardées  : {fig_path}")

    print("\n[4] Évaluation finale ...")
    model = tf.keras.models.load_model(args.out)   # meilleur checkpoint
    loss, acc = model.evaluate(x_test, y_test, verbose=1, batch_size=256)
    print(f"\n    loss     = {loss:.4f}")
    print(f"    accuracy = {acc:.4f}  ({acc*100:.2f} %)")
    print(f"\n    Modèle sauvegardé    : {args.out}")
    print(f"    Résultats sauvegardés : {out_dir}/")
    print("\n=== Terminé ===\n")


if __name__ == "__main__":
    main()
