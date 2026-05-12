"""
focal_loss_cifar10.py
---------------------
Fine-tune ResNet-50/CIFAR-10 avec Focal Loss (γ=2, 3 epochs, lr=1e-4).
Reproduit exactement les conditions du tableau de comparaison de l'article.

Usage :
    python3.12 -u focal_loss_cifar10.py 2>&1 | tee results/cifar10/focal_loss/log.txt
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
import pandas as pd

CLASSES  = ["avion","auto","oiseau","chat","cerf","chien","grenouille","cheval","bateau","camion"]
C        = 10
SEED     = 0
OUT_DIR  = "results/cifar10/focal_loss"

def focal_loss(gamma=2.0):
    def loss_fn(y_true, y_pred):
        y_true  = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        xent    = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        p_t     = tf.reduce_sum(y_pred * tf.one_hot(y_true, C), axis=-1)
        return tf.reduce_mean((1.0 - p_t) ** gamma * xent)
    return loss_fn

def per_class_accuracy(model, x, y):
    preds = model.predict(x, verbose=0, batch_size=256).argmax(axis=1)
    return np.array([(preds[y == c] == c).mean() for c in range(C)])

os.makedirs(OUT_DIR, exist_ok=True)

print("[1] Chargement CIFAR-10 ...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
y_train = y_train.flatten().astype(np.int32)
y_test  = y_test.flatten().astype(np.int32)
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32")  / 255.0

rng    = np.random.default_rng(SEED)
idx    = rng.permutation(len(x_test))
x_eval, y_eval = x_test[idx[5000:]], y_test[idx[5000:]]
print(f"    Train : {len(x_train)}  Eval held-out : {len(x_eval)}")

print("[2] Chargement modèle ...")
model = tf.keras.models.load_model("resnet50_cifar10.keras", compile=False)

print("[3] Baseline held-out ...")
pc0  = per_class_accuracy(model, x_eval, y_eval)
g0   = (model.predict(x_eval, verbose=0, batch_size=256).argmax(1) == y_eval).mean()
std0 = float(np.std(pc0))
print(f"    Baseline : glob={g0*100:.2f}%  σ={std0*100:.2f}%")
for c, n in enumerate(CLASSES):
    print(f"      {n:12s}: {pc0[c]*100:.1f}%")

print("[4] Fine-tuning focal loss (γ=2, 3 epochs, lr=1e-4) ...")
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=focal_loss(gamma=2.0),
    metrics=["accuracy"],
)

ds_train = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
            .shuffle(50000, seed=SEED)
            .batch(128)
            .prefetch(tf.data.AUTOTUNE))

model.fit(ds_train, epochs=3, verbose=1)

print("\n[5] Évaluation held-out post-FL ...")
pc1  = per_class_accuracy(model, x_eval, y_eval)
g1   = (model.predict(x_eval, verbose=0, batch_size=256).argmax(1) == y_eval).mean()
std1 = float(np.std(pc1))

print(f"\n  {'Classe':12s}  {'baseline':>9s}  {'FL':>9s}  {'Δ':>8s}")
print("  " + "─"*44)
for c, n in enumerate(CLASSES):
    print(f"  {n:12s}  {pc0[c]*100:>8.1f}%  {pc1[c]*100:>8.1f}%  {(pc1[c]-pc0[c])*100:>+7.1f}pp")
print("  " + "─"*44)
print(f"  {'Global':12s}  {g0*100:>8.2f}%  {g1*100:>8.2f}%  {(g1-g0)*100:>+7.2f}pp")
print(f"  {'σ':12s}  {std0*100:>8.2f}%  {std1*100:>8.2f}%  {(std1-std0)*100:>+7.2f}pp")

results = {
    "global_base": g0, "global_fl": g1,
    "std_base": std0,  "std_fl": std1,
    **{f"base_{n}": pc0[c] for c, n in enumerate(CLASSES)},
    **{f"fl_{n}"  : pc1[c] for c, n in enumerate(CLASSES)},
}
pd.DataFrame([results]).to_csv(f"{OUT_DIR}/eval_heldout.csv", index=False)
model.save(f"{OUT_DIR}/model_fl.keras")
print(f"\n  Sauvegardé → {OUT_DIR}/")
