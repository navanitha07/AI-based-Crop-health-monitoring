# AI-based-Crop-health-monitoring
#Description
🌾 AI-Based Crop Health Monitoring System
This repository uses the PlantVillage dataset (Resized 224×224 version) to build an AI-based crop health monitoring system. The dataset contains images of plant leaves categorized into healthy and diseased classes, enabling training and evaluation of deep learning models for plant disease detection.

✅ Dataset Source
Dataset: PlantVillage (Resized to 224x224)

Source: Kaggle

Link: https://www.kaggle.com/datasets/bulentsiyah/plantvillage

📂 Dataset Structure
The dataset includes multiple plant species with corresponding disease categories:

Tomato

Potato

Corn (Maize)

Apple

Grape

Pepper

Soybean

Strawberry

Others

Each category contains healthy and various disease-affected leaf images.

🧠 Purpose of This Project
This project aims to:

Detect crop diseases from leaf images using deep learning.

Help farmers and agricultural systems identify plant health issues early.

Support precision farming and smart agriculture systems.

🛠️ Technologies Used
Python

TensorFlow / PyTorch

OpenCV (optional)

NumPy & Pandas

Matplotlib / Seaborn

🌟 Applications
Smart farming systems

Mobile crop disease detection apps

Agricultural advisory platforms

Precision agriculture research


#Codings
!pip install "tensorflow==2.16.1" "keras==3.3.3" --upgrade
!pip install matplotlib opencv-python pillow scikit-learn pandas tqdm
import os, sys, json, math, random, glob, shutil, itertools, pathlib
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import cv2
from PIL import Image

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix

seed = 42
random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)

data_dir = "PlantVillage_resize_224"   # CHANGE if needed
img_size = (224, 224)
batch_size = 32
val_split = 0.2
classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
classes = sorted(classes)
print("Classes:", classes)
assert len(classes) > 1, "No class folders found."

class_to_idx = {c:i for i,c in enumerate(classes)}
idx_to_class = {i:c for c,i in class_to_idx.items()}
with open("class_indices.json", "w") as f:
    json.dump(idx_to_class, f, indent=2)
print("Saved class_indices.json")
valid_exts = {".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}
records = []

def is_image(path):
    return pathlib.Path(path).suffix.lower() in valid_exts

bad_files = []
tiny_files = []
odd_ratio = []

MIN_SIDE = 64       # too small threshold
MIN_RATIO = 0.3     # extreme aspect ratio thresholds
MAX_RATIO = 3.3
for cls in classes:

    folder = os.path.join(data_dir, cls)
    for fn in os.listdir(folder):
        fp = os.path.join(folder, fn)
        if not is_image(fp):
            bad_files.append((fp, "non-image-ext"))
            continue
        
        try:
            with Image.open(fp) as im:
                im.verify()   # quick integrity check
            # reopen to read size (verify closes file)
            im = Image.open(fp)
            w, h = im.size
        
        except Exception as e:
            bad_files.append((fp, f"corrupt: {e}"))
            continue

        if min(w,h) < MIN_SIDE:
            tiny_files.append((fp, f"{w}x{h}"))

        ratio = (w/h) if h else 0
        if ratio < MIN_RATIO or ratio > MAX_RATIO:
            odd_ratio.append((fp, f"{w}x{h}"))

        records.append({"path": fp, "cls": cls, "w": w, "h": h, "ratio": ratio})

      df_meta = pd.DataFrame(records)
    print(f"Scanned {len(df_meta)} images.")
    print(f"Non-image/Corrupt files: {len(bad_files)} | Tiny: {len(tiny_files)} | Odd ratio: {len(odd_ratio)}")

     pd.DataFrame(bad_files, columns=["path","reason"]).to_csv("report_bad_files.csv", index=False)
    pd.DataFrame(tiny_files, columns=["path","size"]).to_csv("report_tiny_files.csv", index=False)
    pd.DataFrame(odd_ratio, columns=["path","size"]).to_csv("report_odd_ratio.csv", index=False)

    cnt = df_meta['cls'].value_counts().sort_index()
    print("\nPer-class counts:\n", cnt)
    cnt.to_csv("report_class_counts.csv")
    os.makedirs("_quarantine", exist_ok=True)
    def move_list(lst, tag):
    for p,_ in lst:
        rel = os.path.relpath(p, start=data_dir)
        dst_dir = os.path.join("_quarantine", tag, os.path.dirname(rel))
        os.makedirs(dst_dir, exist_ok=True)
        try: shutil.move(p, os.path.join(dst_dir, os.path.basename(p)))
        except: pass
    AUTOTUNE = tf.data.AUTOTUNE

    ds_train = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    image_size=img_size,
    batch_size=batch_size,
    validation_split=val_split,
    subset='training',
    seed=seed,
    label_mode='categorical'
    )

    ds_val = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    image_size=img_size,
    batch_size=batch_size,
    validation_split=val_split,
    subset='validation',
    seed=seed,
    label_mode='categorical'
    )

    data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.10),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomContrast(0.1),
    ], name="augmentation")


    class PreprocessLayer(layers.Layer):
    def call(self, x): return preprocess_input(x)

    preprocess_layer = PreprocessLayer(name="mnetv2_preprocess")

    ds_train = ds_train.shuffle(1000).prefetch(AUTOTUNE)
    ds_val   = ds_val.prefetch(AUTOTUNE)

    num_classes = len(classes)
    print("num_classes:", num_classes)

    base_fe = MobileNetV2(include_top=False, weights="imagenet", input_shape=img_size+(3,))
    fe_model = tf.keras.Sequential([preprocess_layer, base_fe, GlobalAveragePooling2D()], name="feature_extractor")
    fe_model.trainable = False

    paths, labels, feats = [], [], []
    for batch_imgs, batch_lbls in tqdm(ds_train.unbatch().batch(64), total=math.ceil(sum(1 for _ in ds_train.unbatch())/64)):
    f = fe_model(batch_imgs, training=False).numpy()
    feats.append(f)
    labels.append(batch_lbls.numpy())


feats = np.vstack(feats)
labels = np.vstack(labels)
y_idx = labels.argmax(1)
train_list = tf.keras.utils.image_dataset_from_directory(
    data_dir, image_size=img_size, batch_size=1,
    validation_split=val_split, subset='training', seed=seed, label_mode='categorical'
)

file_paths = []
for img, lab in train_list.unbatch().take(len(df_meta)):  # will end at training subset size
    break
train_files = []
for c in classes:
    all_files = sorted([os.path.join(data_dir, c, f) for f in os.listdir(os.path.join(data_dir, c)) if is_image(os.path.join(data_dir, c, f))])
    # split deterministically like Keras does: we’ll mimic by proportion
    n = len(all_files)
    val_n = int(round(n*val_split))
    train_n = n - val_n
    random.Random(seed).shuffle(all_files)
    train_files += all_files[:train_n]

paths = train_files[:feats.shape[0]]  # align lengths defensively

df_feats = pd.DataFrame({"path": paths, "y": y_idx})
df_feats["cls"] = df_feats["y"].map(idx_to_class)

centroids = {}
for cidx in range(num_classes):
    m = (y_idx == cidx)
    if m.sum() == 0: continue
    centroids[cidx] = feats[m].mean(axis=0)

dists = np.zeros(len(df_feats), dtype=np.float32)
for i,(p,y) in enumerate(zip(paths, y_idx)):
    if y in centroids:
        dists[i] = np.linalg.norm(feats[i] - centroids[y])
df_feats["dist"] = dists

K = 10  # change per your dataset size
rows = []
for cidx in range(num_classes):
    sub = df_feats[df_feats["y"]==cidx].sort_values("dist", ascending=False).head(K)
    rows.append(sub)
df_outliers = pd.concat(rows) if rows else pd.DataFrame(columns=df_feats.columns)
df_outliers.to_csv("report_visual_outliers_topK.csv", index=False)
print("Saved visual outliers -> report_visual_outliers_topK.csv")
inputs = Input(shape=img_size+(3,))
x = data_augmentation(inputs)
x = preprocess_layer(x)

base = MobileNetV2(include_top=False, weights="imagenet")
base.trainable = False

x = base(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)
outputs = Dense(num_classes, activation="softmax")(x)

model = Model(inputs, outputs)
model.compile(optimizer=Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, mode="max", restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6),
    tf.keras.callbacks.ModelCheckpoint("checkpoints_best.h5", monitor="val_accuracy", save_best_only=True, mode="max"),
    tf.keras.callbacks.CSVLogger("training_log.csv", append=False)
]

history = model.fit(ds_train, validation_data=ds_val, epochs=10, callbacks=callbacks)
# Collect predictions
y_true = []
y_pred = []
y_prob = []

for imgs, labs in ds_val.unbatch().batch(64):
    probs = model.predict(imgs, verbose=0)
    y_prob += list(probs)
    y_pred += list(np.argmax(probs, axis=1))
    y_true += list(np.argmax(labs.numpy(), axis=1))

y_true = np.array(y_true); y_pred = np.array(y_pred); y_prob = np.array(y_prob)
print("Validation accuracy:", (y_true==y_pred).mean())

rep = classification_report(y_true, y_pred, target_names=classes, digits=4, output_dict=True)
pd.DataFrame(rep).transpose().to_csv("report_classification.csv")
print("Saved report_classification.csv")

cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
pd.DataFrame(cm, index=classes, columns=classes).to_csv("report_confusion_matrix.csv")
print("Saved report_confusion_matrix.csv")

plt.figure(figsize=(min(12, 0.6*num_classes), min(12, 0.6*num_classes)))
plt.imshow(cm, interpolation='nearest')
plt.title('Confusion Matrix')
plt.colorbar()
plt.xticks(ticks=np.arange(num_classes), labels=classes, rotation=90)
plt.yticks(ticks=np.arange(num_classes), labels=classes)
plt.tight_layout(); plt.show()
paths_all = []
labels_all = []

 val_files = []
for c in classes:
    all_files = sorted([os.path.join(data_dir, c, f) for f in os.listdir(os.path.join(data_dir, c)) if pathlib.Path(f).suffix.lower() in {".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}])
    n = len(all_files)
    val_n = int(round(n*val_split))
    train_n = n - val_n
    r = random.Random(seed)
    r.shuffle(all_files)
    val_files += all_files[train_n:]
def load_img(path):
    im = tf.keras.utils.load_img(path, target_size=img_size)
    arr = tf.keras.utils.img_to_array(im)
    arr = tf.expand_dims(arr, 0)
    return arr

pred_rows = []
for p in tqdm(val_files):
    true_cls = os.path.basename(os.path.dirname(p))
    arr = load_img(p)
    arr = preprocess_input(arr)
    probs = model.predict(arr, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_cls = idx_to_class[pred_idx]
    conf = float(np.max(probs))
    pred_rows.append({"path": p, "true_cls": true_cls, "pred_cls": pred_cls, "conf": conf})

df_preds = pd.DataFrame(pred_rows)
candidates = df_preds[(df_preds.true_cls != df_preds.pred_cls) & (df_preds.conf >= 0.90)].sort_values("conf", ascending=False)
candidates.to_csv("report_likely_label_mismatches.csv", index=False)
print("Saved likely mismatches -> report_likely_label_mismatches.csv (manual review recommended)")
model.save("plant_disease_mobilenetv2.h5")
tf.saved_model.save(model, "saved_model_mnetv2")
prep_cfg = {
    "backend": "mobilenet_v2",
    "img_size": img_size,
    "scale": "[-1,1]",
    "augmentations": {"flip":"horizontal","rotation":0.05,"zoom":0.1,"translation":[0.1,0.1],"contrast":0.1}
}
with open("preprocessing_config.json","w") as f: json.dump(prep_cfg, f, indent=2)
print("Saved model + SavedModel + preprocessing_config.json")
