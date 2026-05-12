"""Quick eval of resnet50_isic2019_ce_ss.keras on test_ss.npz."""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf
import pandas as pd

CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
N_CLASSES = 8

print("[1] Chargement test_ss ...")
d = np.load("data/isic2019_cache/test_ss.npz")
x = d["imgs"].astype(np.float32)
y = d["labels"].astype(np.int32)
print(f"    {len(x)} images")

print("[2] Chargement modèle spiked ...")
model = tf.keras.models.load_model("resnet50_isic2019_ce_ss.keras")

print("[3] Inference ...")
probs = model.predict(x, verbose=0, batch_size=64)
preds = probs.argmax(axis=1)

global_acc = (preds == y).mean()
per_class = np.array([(preds[y == c] == c).mean() if (y == c).sum() > 0 else 0.0
                      for c in range(N_CLASSES)])
bal = per_class.mean()
std = per_class.std()

# Loss
y_oh = tf.keras.utils.to_categorical(y, N_CLASSES)
loss = tf.keras.losses.categorical_crossentropy(y_oh, probs).numpy().mean()

print(f"\n    Global acc       : {global_acc*100:.2f}%")
print(f"    Balanced acc     : {bal*100:.2f}%")
print(f"    Std inter-classes: {std*100:.2f}%")
print(f"    Loss             : {loss:.4f}")
print(f"\n    {'Classe':6s}  {'acc':>6s}")
for i, c in enumerate(CLASSES):
    print(f"    {c:6s}  {per_class[i]*100:>5.1f}%")

out = pd.DataFrame([{
    "method": "SS_baseline_256hvp_iter9",
    "global_acc": global_acc, "balanced_acc": bal, "std": std, "loss": loss,
    **{f"acc_{CLASSES[i]}": per_class[i] for i in range(N_CLASSES)},
}])
out.to_csv("results/isic2019/ce_ss/eval_test_iter9.csv", index=False)
print(f"\n    → results/isic2019/ce_ss/eval_test_iter9.csv")
