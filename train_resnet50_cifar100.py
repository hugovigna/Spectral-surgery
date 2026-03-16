"""
train_resnet50_cifar100.py
--------------------------
Fine-tune ResNet50 (poids ImageNet) sur CIFAR-100.
Même approche que pour CIFAR-10 : tout le réseau entraînable,
ImageNet comme initialisation, AdamW + EarlyStopping.

Usage :
    python3 train_resnet50_cifar100.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf

# ── 1. Données ──────────────────────────────────────────────────────────
print("[1] Chargement de CIFAR-100 ...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
y_train, y_test = y_train.flatten(), y_test.flatten()
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32")  / 255.0

print(f"    Train : {x_train.shape}  Test : {x_test.shape}")

# ── 2. Modèle ──────────────────────────────────────────────────────────
print("[2] Construction du modèle ...")

augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomTranslation(0.1, 0.1, fill_mode="reflect"),
    tf.keras.layers.RandomContrast(0.2),
], name="augmentation")

base = tf.keras.applications.ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(32, 32, 3),
    pooling="avg",
)
base.trainable = True

inputs  = tf.keras.Input(shape=(32, 32, 3))
x       = augment(inputs)
x       = base(x)
x       = tf.keras.layers.Dropout(0.4)(x)
outputs = tf.keras.layers.Dense(100, activation="softmax")(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name="ResNet50_CIFAR100")

model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

n_params = sum(int(tf.size(v)) for v in model.trainable_variables)
print(f"    Paramètres entraînables : {n_params:,}")

# ── 3. Entraînement ────────────────────────────────────────────────────
print("[3] Entraînement (max 40 epochs, early stopping) ...")

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath="resnet50_cifar100_best.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5,
        min_lr=1e-7, verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=8,
        restore_best_weights=True, verbose=1,
    ),
]

history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=40,
    batch_size=128,
    callbacks=callbacks,
    verbose=1,
)

# ── 4. Évaluation ─────────────────────────────────────────────────────
print("\n[4] Évaluation finale ...")
loss, acc = model.evaluate(x_test, y_test, verbose=0, batch_size=256)
print(f"    Loss : {loss:.4f}")
print(f"    Accuracy : {acc:.4f}  ({acc*100:.2f}%)")

# ── 5. Sauvegarde du modèle d'inférence (sans augmentation) ───────────
# On reconstruit sans les couches d'augmentation pour que le HVP soit propre
print("\n[5] Sauvegarde du modèle d'inférence ...")

inference_model = tf.keras.Sequential([
    tf.keras.applications.ResNet50(
        weights=None, include_top=False,
        input_shape=(32, 32, 3), pooling="avg",
    ),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(100, activation="softmax"),
])
inference_model.build((None, 32, 32, 3))

# Copier les poids depuis le modèle entraîné (base + dropout + dense)
inference_model.layers[0].set_weights(base.get_weights())
inference_model.layers[2].set_weights(model.layers[-1].get_weights())

inference_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

loss2, acc2 = inference_model.evaluate(x_test, y_test, verbose=0, batch_size=256)
print(f"    Vérification : {acc2:.4f} (doit ≈ {acc:.4f})")

inference_model.save("resnet50_cifar100.keras")
print(f"    Modèle sauvegardé : resnet50_cifar100.keras")
