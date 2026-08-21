# 1174976-COMP663A2

# COMP663 Assignment 2
## Hyperparameter Optimisation and Neural Architecture Search for Forest Cover Classification

### Author
Ned Matheson

---

# Overview

This project investigates several optimisation approaches for improving the performance of a feed-forward neural network on a forest cover type classification problem.

The assignment compares:

1. Baseline Neural Network
2. Random Search
3. Bayesian Optimisation (Optuna)
4. Neural Architecture Search (NAS)

The objective is to determine whether systematic optimisation can improve classification performance relative to a manually configured baseline model.

---

# Dataset

The dataset contains environmental and geographical measurements used to predict forest cover type.

Target Variable:

- `Cover_Type`

Input Features:

- Elevation
- Slope
- Horizontal_Distance_To_Hydrology
- Vertical_Distance_To_Hydrology
- Horizontal_Distance_To_Roadways
- Hillshade_9am
- Hillshade_Noon
- Hillshade_3pm
- Horizontal_Distance_To_Fire_Points
- Wilderness Area indicators
- Engineered Aspect features

Dataset Size:

- 571,012 observations
- 15 processed input features
- 5 target classes

---

# Preprocessing

The following preprocessing pipeline was applied:

### Missing Values

No missing values were present.

### Feature Engineering

The directional Aspect variable was transformed into:

- Aspect_sin
- Aspect_cos

This preserves cyclical directional information while eliminating the discontinuity between 0° and 360°.

### Feature Scaling

Continuous features were standardised using:

```python
StandardScaler()
```

Binary wilderness area indicators were not scaled.

### Train-Test Split

```python
train_test_split(
    test_size=0.2,
    stratify=y,
    random_state=42
)
```

---

# Evaluation Metrics

## Primary Metric

### Macro F1 Score

Macro F1 was selected as the primary metric because:

- Precision and recall are both considered.
- Class imbalance is handled appropriately.
- All classes receive equal importance.

## Secondary Metrics

- Accuracy
- Balanced Accuracy

These metrics provide additional insight into overall and class-specific performance.

---

# Models Evaluated

## Baseline Neural Network

Architecture:

```text
15 → 24 → 12 → 5
```

Activation:

```text
Sigmoid
```

Optimizer:

```text
Adam
```

Learning Rate:

```text
0.001
```

---

## Random Search

Random Search explored combinations of:

- Learning Rate
- Batch Size
- Hidden Layer 1 Size
- Hidden Layer 2 Size
- Optimizer

---

## Bayesian Optimisation

Bayesian Optimisation was implemented using:

```text
Optuna
```

Search Parameters:

- Learning Rate
- Batch Size
- Hidden Layer 1 Size
- Hidden Layer 2 Size
- Weight Decay

Objective:

```text
Maximise Macro F1 Score
```

---

## Neural Architecture Search (NAS)

NAS searched over:

- Number of Hidden Layers
- Hidden Units
- Activation Function
- Dropout Rate

Objective:

```text
Maximise Macro F1 Score
```

---

# Project Structure

```text
project/
│
├── src/
│   └── assignment_notebook.ipynb
│
├── data/
│   └── forest_cover_data.csv
│
├── figures/
│   ├── *.png
│   └── *.pdf
│
├──final_nas_model.pth├── scaler.pkl
├── best_nas_params.json
│
└── README.md
```

---

# Running the Notebook

Install required packages:

```bash
pip install torch
pip install pandas
pip Install numpy
pip install scikit-learn
pip install matplotlib
pip install seaborn
pip install optuna
pip install joblib
```

Launch Jupyter:
```bash
jupyter notebook
```

Open:

```text
assignment_notebook.ipynb
```

Run all cells sequentially.
---

# Saved Model

The final model is stored as:

```text
final_nas_model.pth
```

Associated preprocessing objects:

```text
scaler.pkl
best_nas_params.json
'``

---

# Loading the Final Model
Load the scaler:

```python
import joblib

scaler = joblib.load("scaler.pkl")
```

Load the NAS parameters:

```python
import json

with open("best_nas_params.json", "r") as if:
    best_nas_params = json.load(f)
```

Rebuild the architecture:

```python
model = NASNN(
    input_size=15,
    n_layers=best_nas_params["n_layer"],
    hidden_units=best_nas_params["hidden_units"],
    activation_name=best_nas_params["activation"],   dropout_rate=best_nas_params["dropout"]
)
```

Load the trained weights:

```python
model.load_state dict(
    torch.load(
        "final_nas_model.pth",
        weights_only=True
    )
)

model.eval()
```
---

# Reproducibility

Random seeds were fixed where possible:

```Python
SEED = 42
```

This was used for:

- Train/Test Split
- NumPy Operations
- PyTorch Initialisation
---

# Hardware

Experiments were performed using:

```text
NVIDIA Geforce RTX 4050 Laptop GPU
```

GPU acceleration was automatically enabled when CUDA was available.

---

Outputs

The notebook generates:
- Training loss plots
- Confusion Matrices
- Classification reports
- Optimisation trial summaries
- Performance comparison tables

GeneratEd figures are stored within:

```text
figures/
```

---

# Final Deliverables

The submission includes:
- Jupyter Notebook
- README.md
- Final Saved NAS Model
- Scaler
- Best Hyperparameter Configuration
- Generated Figures

---

