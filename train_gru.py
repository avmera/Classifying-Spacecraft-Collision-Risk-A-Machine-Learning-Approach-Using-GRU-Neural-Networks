# ============================================
# Fast MLP Training + Threshold Tuning + Summary Table
# ============================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve,
    precision_recall_curve, classification_report, f1_score, average_precision_score
)
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import mixed_precision
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# --------------------------------------------
# ⚙ Speed-ups
# --------------------------------------------
mixed_precision.set_global_policy('mixed_float16')   # أسرع على GPU/TPU
JIT_COMPILE = True                                    # XLA
BATCH = 1024
EPOCHS = 60

# --------------------------------------------
# 📌 Focal Loss
# --------------------------------------------
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)
        weight = alpha * y_true * K.pow(1 - y_pred, gamma) + \
                 (1 - alpha) * (1 - y_true) * K.pow(y_pred, gamma)
        return K.mean(weight * cross_entropy)
    return focal_loss_fixed

# --------------------------------------------
# 📥 Load & Prepare Data
# --------------------------------------------
df = pd.read_csv('train_data.csv')
df = df.infer_objects(copy=False).interpolate()

FEATURES = [
    'max_risk_scaling', 'time_to_tca', 'mahalanobis_distance', 'max_risk_estimate', 'c_h_per',
    'relative_velocity_t', 'c_recommended_od_span', 'relative_speed', 'c_actual_od_span',
    'c_cd_area_over_mass', 't_j2k_sma', 't_h_per', 'c_ctdot_r', 'c_cr_area_over_mass', 't_h_apo',
    'c_sigma_t', 'c_time_lastob_end', 'c_obs_available', 'c_ctdot_n'
]
X = df[FEATURES].to_numpy()
y = (df['risk'] > -6).astype(int).to_numpy()

# Split: Train / (Val+Test)
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, test_size=0.30, random_state=43, stratify=y
)
# Split: Val / Test
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.50, random_state=43, stratify=y_tmp
)

print("Counts:", "train", Counter(y_train), "val", Counter(y_val), "test", Counter(y_test))

# Scale (fit on Train only)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

# --------------------------------------------
# ⚖ Class Weights (أسرع من SMOTE)
# (لو تبغين SMOTE بدلًا من class_weight، فعّلي SMOTE هنا وعلّقي class_weight)
# --------------------------------------------
classes = np.unique(y_train)
weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

# -- بديل (أبطأ): SMOTE
# from imblearn.over_sampling import SMOTE
# sm = SMOTE(random_state=42)
# X_train_s, y_train = sm.fit_resample(X_train_s, y_train)
# class_weight = None

# --------------------------------------------
# 🧠 Build Fast MLP
# --------------------------------------------
inp = Input(shape=(X_train_s.shape[1],))
x = BatchNormalization()(inp)
x = Dense(256, activation='relu')(x)
x = Dropout(0.30)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.20)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.10)(x)
out = Dense(1, activation='sigmoid', dtype='float32')(x)  # إخراج fp32 لاستقرار العددي

model = Model(inp, out)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=focal_loss(gamma=2., alpha=0.25),
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
    ],
    jit_compile=JIT_COMPILE
)

early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
lr_scheduler   = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1)

# Datasets
train_ds = tf.data.Dataset.from_tensor_slices((X_train_s.astype('float32'), y_train.astype('float32'))) \
                          .shuffle(10000).batch(BATCH).prefetch(tf.data.AUTOTUNE)
val_ds   = tf.data.Dataset.from_tensor_slices((X_val_s.astype('float32'), y_val.astype('float32'))) \
                          .batch(BATCH).prefetch(tf.data.AUTOTUNE)

# --------------------------------------------
# 🚀 Train
# --------------------------------------------
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=[early_stopping, lr_scheduler],
    verbose=2,
    class_weight=class_weight
)

model.save_weights('mlp_fast.weights.h5')
joblib.dump(scaler, 'scaler_fast.save')
print("✅ Training complete. Model and scaler saved!")

# --------------------------------------------
# 🔍 Helper: bootstrap CIs
# --------------------------------------------
def bootstrap_ci_point(metric_func, y_true, y_pred_labels, n_bootstrap=2000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    values = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        values.append(metric_func(y_true[idx], y_pred_labels[idx]))
    values = np.asarray(values)
    return float(np.mean(values)), (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))

def bootstrap_ci_prob(metric_func, y_true, y_scores, n_bootstrap=2000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    values = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        values.append(metric_func(y_true[idx], y_scores[idx]))
    values = np.asarray(values)
    return float(np.mean(values)), (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))

# --------------------------------------------
# 🧪 Predictions (Train/Val/Test)
# --------------------------------------------
y_train_proba = model.predict(X_train_s.astype('float32'), batch_size=4096, verbose=0).ravel()
y_val_proba   = model.predict(X_val_s.astype('float32'), batch_size=4096, verbose=0).ravel()
y_test_proba  = model.predict(X_test_s.astype('float32'), batch_size=4096, verbose=0).ravel()

# --------------------------------------------
# 🎯 Threshold Tuning on Validation (maximize F1)
# --------------------------------------------
prec_v, rec_v, thrs_v = precision_recall_curve(y_val, y_val_proba)
f1_v = 2 * (prec_v * rec_v) / (prec_v + rec_v + 1e-12)
best_idx = np.argmax(f1_v)
best_thresh = thrs_v[best_idx] if len(thrs_v) > 0 else 0.5

# بديل يضمن Recall ≥ 0.80 (اختياري)
target_recall = 0.80
idx_recall_ok = np.where(rec_v >= target_recall)[0]
alt_thresh_r80 = (thrs_v[idx_recall_ok[-1]] if len(idx_recall_ok) > 0 and idx_recall_ok[-1] < len(thrs_v)
                  else best_thresh)

print(f"\n🎯 Best threshold for F1 on VAL: {best_thresh:.4f}")
print(f"🎯 Alt threshold for Recall≥{target_recall:.2f} on VAL: {alt_thresh_r80:.4f}")

# نستخدم العتبة الأفضل لـ F1
THRESH = best_thresh

# --------------------------------------------
# 📈 Plots (ROC & PR + Confusion Matrix on Test)
# --------------------------------------------
# ROC (Test)
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
auc_test = roc_auc_score(y_test, y_test_proba)
plt.plot(fpr, tpr, label=f"ROC (AUC = {auc_test:.3f})")
plt.plot([0,1],[0,1],'--',lw=1)
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Test)"); plt.legend(); plt.grid(True); plt.show()

# PR (Test)
prec_t, rec_t, _ = precision_recall_curve(y_test, y_test_proba)
ap_test = average_precision_score(y_test, y_test_proba)
plt.plot(rec_t, prec_t, label=f"PR (AP = {ap_test:.3f})")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Precision–Recall Curve (Test)"); plt.legend(); plt.grid(True); plt.show()

# Confusion Matrix (Test @ tuned threshold)
y_test_pred = (y_test_proba > THRESH).astype(int)
cm = confusion_matrix(y_test, y_test_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap=plt.cm.Blues)
plt.title(f"Confusion Matrix (Test, threshold={THRESH:.4f})")
plt.show()

print("📊 Classification Report (Test, tuned threshold):\n")
print(classification_report(y_test, y_test_pred, digits=4))

# --------------------------------------------
# 📏 Metrics (Train/Val using current best weights) & CIs (Test)
# --------------------------------------------
# Training metrics (evaluate/recompute on current weights)
# ملاحظة: قيم "Best" هنا تُحسب على كامل Train/Val بعد استرجاع أفضل وزن (val_loss)
# وهذا أدق من max(history.history[...])
y_train_pred = (y_train_proba > THRESH).astype(int)
y_val_pred   = (y_val_proba > THRESH).astype(int)

train_acc = (y_train_pred == y_train).mean()
val_acc   = (y_val_pred == y_val).mean()

train_prec = ( (y_train_pred & (y_train==1)).sum() / max(1, y_train_pred.sum()) )
val_prec   = ( (y_val_pred & (y_val==1)).sum() / max(1, y_val_pred.sum()) )

train_rec  = ( (y_train_pred & (y_train==1)).sum() / max(1, (y_train==1).sum()) )
val_rec    = ( (y_val_pred & (y_val==1)).sum() / max(1, (y_val==1).sum()) )

# Loss/ROC-AUC/PR-AUC على Train/Val
# (Loss نحسبه عبر evaluate لموثوقية أعلى)
train_eval = model.evaluate(X_train_s.astype('float32'), y_train.astype('float32'), batch_size=4096, verbose=0)
val_eval   = model.evaluate(X_val_s.astype('float32'),   y_val.astype('float32'),   batch_size=4096, verbose=0)
# train_eval = [loss, acc, prec, rec, auc]
train_loss = float(train_eval[0]); val_loss = float(val_eval[0])

train_auc = roc_auc_score(y_train, y_train_proba)
val_auc   = roc_auc_score(y_val,   y_val_proba)

train_pr_auc = average_precision_score(y_train, y_train_proba)
val_pr_auc   = average_precision_score(y_val,   y_val_proba)

# Test point metrics @ tuned threshold
test_acc  = (y_test_pred == y_test).mean()
test_prec = ( (y_test_pred & (y_test==1)).sum() / max(1, y_test_pred.sum()) )
test_rec  = ( (y_test_pred & (y_test==1)).sum() / max(1, (y_test==1).sum()) )
test_auc  = auc_test
test_pr_auc = ap_test

# CIs
acc_mean, acc_ci = bootstrap_ci_point(lambda yt, yp: (yt == yp).mean(), y_test, y_test_pred)
f1_mean,  f1_ci  = bootstrap_ci_point(f1_score, y_test, y_test_pred)
roc_mean, roc_ci = bootstrap_ci_prob(roc_auc_score, y_test, y_test_proba)
pr_mean,  pr_ci  = bootstrap_ci_prob(average_precision_score, y_test, y_test_proba)

# --------------------------------------------
# 🧾 Summary Table (same style)
# --------------------------------------------
def ci_str(ci):
    return f"[{ci[0]:.4f}, {ci[1]:.4f}]"

print("\n=== Summary Table ===")
print(f"{'Metric':<12} {'Training (Best)':<18} {'Validation (Best)':<20} {'Test (Point)':<15} {'95% CI (Test)'}")
print(f"{'Accuracy':<12} {train_acc:>10.4f} {val_acc:>18.4f} {test_acc:>15.4f} {ci_str(acc_ci)}")
print(f"{'Loss':<12} {train_loss:>10.4f} {val_loss:>18.4f} {'-':>15} {'-'}")
print(f"{'Precision':<12} {train_prec:>10.4f} {val_prec:>18.4f} {test_prec:>15.4f} {'-'}")
print(f"{'Recall':<12} {train_rec:>10.4f} {val_rec:>18.4f} {test_rec:>15.4f} {'-'}")
print(f"{'ROC-AUC':<12} {train_auc:>10.4f} {val_auc:>18.4f} {test_auc:>15.4f} {ci_str(roc_ci)}")
print(f"{'PR-AUC':<12} {train_pr_auc:>10.4f} {val_pr_auc:>18.4f} {test_pr_auc:>15.4f} {ci_str(pr_ci)}")
print(f"\n(Threshold chosen on VAL for best F1 = {THRESH:.4f} | Alt threshold for Recall≥{target_recall:.2f} = {alt_thresh_r80:.4f})")
print(f"(Also reporting Test F1 CI: mean={f1_mean:.4f}, CI={ci_str(f1_ci)})")


# =========================================
# 📈 Training vs Validation curves
# =========================================
import matplotlib.pyplot as plt

def plot_history(history, metric='loss'):
    plt.figure(figsize=(6,4))
    plt.plot(history.history[metric], label=f"Train {metric}")
    plt.plot(history.history[f"val_{metric}"], label=f"Val {metric}")
    plt.xlabel("Epoch")
    plt.ylabel(metric.capitalize())
    plt.title(f"Training vs Validation {metric.capitalize()}")
    plt.legend()
    plt.grid(True)
    plt.show()

# 🔹 Loss curve
plot_history(history, 'loss')

# 🔹 Accuracy curve
plot_history(history, 'accuracy')

# 🔹 Precision curve
plot_history(history, 'precision')

# 🔹 Recall curve
plot_history(history, 'recall')

# 🔹 AUC curve
plot_history(history, 'auc')



