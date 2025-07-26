import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Masking
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# Load and preprocess data
df = pd.read_csv('train_data.csv')
df = df.interpolate()

FEATURES = [
    'max_risk_scaling', 'time_to_tca', 'mahalanobis_distance', 'max_risk_estimate', 'c_h_per',
    'relative_velocity_t', 'c_recommended_od_span', 'relative_speed', 'c_actual_od_span',
    'c_cd_area_over_mass', 't_j2k_sma', 't_h_per', 'c_ctdot_r', 'c_cr_area_over_mass', 't_h_apo',
    'c_sigma_t', 'c_time_lastob_end', 'c_obs_available', 'c_ctdot_n'
]
X = df[FEATURES]
y = (df['risk'] > -6).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=43, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
cw = {0: class_weights[0], 1: class_weights[1]}

model = Sequential([
    Masking(mask_value=0., input_shape=(1, X_train_reshaped.shape[2])),
    GRU(100, activation='tanh', return_sequences=True),
    Dropout(0.3),
    GRU(80, activation='tanh', return_sequences=True),
    Dropout(0.2),
    GRU(50, activation='tanh', return_sequences=True),
    Dropout(0.1),
    GRU(30, activation='tanh'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(
    X_train_reshaped, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    shuffle=True,
    class_weight=cw,
    callbacks=[early_stopping],
    verbose=2
)
model.save_weights('gru_risk_model.weights.h5')
joblib.dump(scaler, "scaler.save")
print("Training complete. Model and scaler saved!")


