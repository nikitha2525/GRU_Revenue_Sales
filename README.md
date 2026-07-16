# 📈 Day 61 — GRU for Time Series Forecasting: Next-Day Revenue Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data-150458?style=flat-square&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Viz-11557C?style=flat-square)
![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-20BEFF?style=flat-square&logo=kaggle&logoColor=white)
![Challenge](https://img.shields.io/badge/100%20Days%20AI%2FML-Day%2061-blueviolet?style=flat-square)

**Choosing between GRU and LSTM isn't about which is better — it's about the problem. Test both on your data. That's the only reliable way to know.**

</div>

---

## 📌 Overview

Day 48 built time series forecasting with SimpleRNN. Day 49 + 57 used LSTM. Day 61 introduces **GRU (Gated Recurrent Unit)** — a streamlined alternative that achieves comparable performance to LSTM with fewer parameters and faster training, making it a strong default choice for many sequence problems.

The unique focus today: implementing the **same forecasting model in both TensorFlow and PyTorch** side by side — to deeply understand how each framework structures layers, training loops, and data pipelines differently.

Dataset: [Time Series Starter Dataset](https://www.kaggle.com/datasets/podsyp/time-series-starter-dataset) (Kaggle) — historical sales/revenue data.

> **Hard truth learned today:** Choosing between GRU and LSTM isn't about which is better — it's about the problem you're solving. GRUs are often faster and more lightweight, while LSTMs may perform better on tasks requiring longer-term memory. Testing both on your dataset is the only reliable way to know which performs best.

---

## 🧠 GRU vs LSTM vs SimpleRNN

```
SimpleRNN:   hₜ = tanh(Wₓxₜ + Wₕhₜ₋₁ + b)
             One operation. No gating. Vanishing gradient on long sequences.

LSTM:        4 gates: forget · input · cell update · output
             Separate cell state Cₜ carries long-term memory.
             Most parameters. Best for very long-range dependencies.

GRU:         2 gates: reset · update
             No separate cell state — merged into hₜ.
             Fewer parameters. Comparable performance on most tasks.
```

### GRU Gate Equations

$$r_t = \sigma(W_r \cdot [h_{t-1},\ x_t] + b_r) \qquad \text{(Reset Gate)}$$

$$z_t = \sigma(W_z \cdot [h_{t-1},\ x_t] + b_z) \qquad \text{(Update Gate)}$$

$$\tilde{h}_t = \tanh(W \cdot [r_t \odot h_{t-1},\ x_t] + b) \qquad \text{(Candidate Hidden State)}$$

$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t \qquad \text{(Final Hidden State)}$$

**What each gate does:**

| Gate | Formula | Role |
|---|---|---|
| Reset gate $r_t$ | $\sigma(\cdot) \in (0,1)$ | How much of past hidden state to forget when computing candidate |
| Update gate $z_t$ | $\sigma(\cdot) \in (0,1)$ | How much of past vs new candidate state to keep |
| Candidate $\tilde{h}_t$ | $\tanh(\cdot)$ | What new information to potentially write |
| Output $h_t$ | Interpolation | Blend of old memory and new candidate |

> When $z_t \approx 0$: keep old hidden state (long-term memory preserved)
> When $z_t \approx 1$: replace with new candidate (rapid adaptation)

---

## 🆚 GRU vs LSTM — Side-by-Side

| Property | SimpleRNN | LSTM | GRU |
|---|---|---|---|
| Gates | 0 | 4 (forget, input, cell, output) | 2 (reset, update) |
| Cell state | ❌ | ✅ Separate $C_t$ | ❌ Merged into $h_t$ |
| Parameters | Fewest | Most | Moderate |
| Training speed | Fastest | Slowest | Fast |
| Vanishing gradient | ❌ Severe | ✅ Solved | ✅ Solved |
| Long-range memory | ❌ Poor | ✅ Best | ✅ Good |
| Best for | Toy tasks | Very long sequences | Most practical tasks |

---

## 📊 Dataset

```python
import kagglehub
path = kagglehub.dataset_download("podsyp/time-series-starter-dataset")
```

| Property | Detail |
|---|---|
| Source | Kaggle — Time Series Starter Dataset |
| Task | Next-day revenue / sales forecasting |
| Input | Historical daily sales values |
| Target | Next day's revenue |
| Preprocessing | MinMaxScaler → sliding window sequences |

---

## 🔬 What I Implemented

### Data Preparation (Shared)

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import kagglehub, os

# ─── Load Dataset ────────────────────────────────────────────────────────────────
path = kagglehub.dataset_download("podsyp/time-series-starter-dataset")
df   = pd.read_csv(os.path.join(path, 'month_values.csv'),
                    parse_dates=['date'], index_col='date')

print(df.head())
print(f"Shape: {df.shape}")

# ─── Select Revenue Column & Scale ───────────────────────────────────────────────
series = df['sales'].values.reshape(-1, 1)    # or 'revenue' depending on CSV

scaler        = MinMaxScaler(feature_range=(0, 1))
series_scaled = scaler.fit_transform(series)

# ─── Sliding Window Sequences ────────────────────────────────────────────────────
WINDOW_SIZE = 30    # use 30 past days to predict the next day

def create_sequences(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i : i + window])
        y.append(data[i + window])
    return np.array(X), np.array(y)

X, y = create_sequences(series_scaled, WINDOW_SIZE)
# X shape: (N, 30, 1)  y shape: (N, 1)

# ─── Train/Test Split (no shuffle — temporal order must be preserved) ─────────────
split    = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")
```

---

## 🟠 Implementation 1 — TensorFlow / Keras

```python
import tensorflow as tf
from tensorflow import keras

# ─── Model ───────────────────────────────────────────────────────────────────────
tf_model = keras.Sequential([
    keras.layers.GRU(
        units             = 64,
        activation        = 'tanh',
        return_sequences  = False,
        input_shape       = (WINDOW_SIZE, 1)
    ),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1)                        # single regression output
])

tf_model.compile(
    optimizer = keras.optimizers.Adam(learning_rate=0.001),
    loss      = 'mean_squared_error',
    metrics   = ['mae']
)

tf_model.summary()

# ─── Train ───────────────────────────────────────────────────────────────────────
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True
)

tf_history = tf_model.fit(
    X_train, y_train,
    epochs          = 100,
    batch_size      = 32,
    validation_split= 0.1,
    callbacks       = [early_stop],
    verbose         = 1
)

# ─── Evaluate ────────────────────────────────────────────────────────────────────
tf_pred_scaled = tf_model.predict(X_test)
tf_pred        = scaler.inverse_transform(tf_pred_scaled)
y_actual       = scaler.inverse_transform(y_test)

tf_mse  = mean_squared_error(y_actual, tf_pred)
tf_rmse = np.sqrt(tf_mse)
tf_r2   = r2_score(y_actual, tf_pred)

print(f"\n[TensorFlow GRU]")
print(f"  RMSE : {tf_rmse:.4f}")
print(f"  R²   : {tf_r2:.4f}")
```

---

## 🔴 Implementation 2 — PyTorch

```python
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ─── Convert to Tensors ──────────────────────────────────────────────────────────
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_test_t  = torch.tensor(y_test,  dtype=torch.float32)

train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=32, shuffle=False          # ← never shuffle time series
)

# ─── Model ───────────────────────────────────────────────────────────────────────
class GRUForecast(nn.Module):
    def __init__(self, input_size=1, hidden_size=64,
                 num_layers=1, output_size=1):
        super(GRUForecast, self).__init__()
        self.gru = nn.GRU(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = 0.2 if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.gru(x)         # out: (batch, seq_len, hidden_size)
        out    = out[:, -1, :]       # last time step: (batch, hidden_size)
        return self.fc(out)

pt_model  = GRUForecast()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(pt_model.parameters(), lr=0.001)

# ─── Training Loop ───────────────────────────────────────────────────────────────
EPOCHS   = 100
pt_losses = []

for epoch in range(EPOCHS):
    pt_model.train()
    epoch_loss = 0.0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        preds = pt_model(X_batch)
        loss  = criterion(preds, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pt_model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()

    avg = epoch_loss / len(train_loader)
    pt_losses.append(avg)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {avg:.6f}")

# ─── Evaluate ────────────────────────────────────────────────────────────────────
pt_model.eval()
with torch.no_grad():
    pt_pred_scaled = pt_model(X_test_t).numpy()

pt_pred  = scaler.inverse_transform(pt_pred_scaled)
pt_mse   = mean_squared_error(y_actual, pt_pred)
pt_rmse  = np.sqrt(pt_mse)
pt_r2    = r2_score(y_actual, pt_pred)

print(f"\n[PyTorch GRU]")
print(f"  RMSE : {pt_rmse:.4f}")
print(f"  R²   : {pt_r2:.4f}")
```

---

## 🔄 TensorFlow vs PyTorch — Framework Comparison

| Aspect | TensorFlow / Keras | PyTorch |
|---|---|---|
| GRU layer | `keras.layers.GRU(units=64)` | `nn.GRU(input_size=1, hidden_size=64)` |
| Input shape spec | `input_shape=(seq, features)` | `batch_first=True`, inferred |
| Training | `model.fit(X, y, epochs=...)` | Manual epoch + batch loop |
| Early stopping | `EarlyStopping` callback | Manual `val_loss` tracking |
| Gradient clipping | `clipvalue` in optimizer | `clip_grad_norm_()` |
| Evaluation | `model.predict(X_test)` | `model.eval()` + `torch.no_grad()` |
| Shuffle time series | `shuffle=False` in `.fit()` | `shuffle=False` in DataLoader |
| Code verbosity | Lower | Higher (more control) |

---

## 📊 Visualization

```python
# ─── Forecast vs Actual ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

for ax, pred, title in zip(
    axes,
    [tf_pred, pt_pred],
    ['TensorFlow GRU', 'PyTorch GRU']
):
    ax.plot(y_actual,  label='Actual',    color='steelblue', linewidth=1.5)
    ax.plot(pred,      label='Predicted', color='tomato',
            linewidth=1.5, linestyle='--')
    ax.set_title(f'{title} — Revenue Forecast', fontsize=13)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Revenue')
    ax.legend()

plt.tight_layout()
plt.savefig('outputs/gru_forecast_comparison.png', dpi=150)
plt.show()

# ─── Side-by-side metrics ────────────────────────────────────────────────────────
print(f"\n{'Metric':<12} {'TensorFlow':>12} {'PyTorch':>12}")
print("─" * 38)
print(f"{'RMSE':<12} {tf_rmse:>12.4f} {pt_rmse:>12.4f}")
print(f"{'R²':<12} {tf_r2:>12.4f} {pt_r2:>12.4f}")
```

---

## 💡 Key Learnings

- **GRUs capture long-term dependencies with fewer parameters than LSTM** — 2 gates instead of 4 means faster training and less overfitting risk on smaller datasets
- **Implementing the same model in both frameworks reveals their design philosophy** — Keras abstracts the loop; PyTorch exposes it
- **Never shuffle time series data** — this applies in both frameworks; temporal order is the information the model learns from
- **`batch_first=True` in PyTorch** matches Keras convention `(batch, seq, features)` — critical for consistent input shapes
- **Gradient clipping stabilizes GRU training** — especially useful when sequences are long or the dataset has sharp revenue spikes

---

## ⚠️ When to Use GRU vs LSTM

| Scenario | Recommendation |
|---|---|
| Short sequences (< 50 steps) | GRU — faster, fewer params, comparable accuracy |
| Very long sequences (> 200 steps) | LSTM — separate cell state better preserves distant context |
| Limited compute / fast iteration | GRU — trains noticeably faster |
| Production accuracy benchmark | Test both, compare RMSE on your validation set |
| Time series forecasting (general) | GRU is a strong default starting point |

---

## 🗂️ Project Structure

```
day-61-gru-forecasting/
├── tensorflow_gru.py            # TF/Keras GRU implementation
├── pytorch_gru.py               # PyTorch GRU implementation
├── data_prep.py                 # Shared preprocessing & sequence creation
├── compare.py                   # Side-by-side metrics + plots
├── outputs/
│   ├── gru_forecast_comparison.png
│   ├── training_loss_tf.png
│   └── training_loss_pt.png
└── README.md
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/your-username/day-61-gru-forecasting
cd day-61-gru-forecasting
pip install -r requirements.txt

# TensorFlow implementation
python tensorflow_gru.py

# PyTorch implementation
python pytorch_gru.py

# Side-by-side comparison
python compare.py
```

**Requirements:**
```
tensorflow>=2.10
torch
pandas
numpy
scikit-learn
matplotlib
kagglehub
```

---

## 🔗 Part of the 100 Days AI/ML Engineer Challenge

> Day 61 of 100 — GRU Time Series Forecasting in TensorFlow & PyTorch

| ← Previous | Current | Next → |
|---|---|---|
| [Day 60 — BERT Similarity](#) | **Day 61 — GRU Forecasting** | [Day 62](#) |



---

<div align="center">
<sub>Built with curiosity · Part of #100DaysOfAIML · #GRU #TimeSeriesForecasting #TensorFlow #PyTorch #DeepLearning #RevenuePrediction</sub>
</div>
