import torch
import joblib
import json

# Load scaler
scaler = joblib.load("scaler.pkl")

# Load parameters
with open("best_nas_params.json", "r") as f:
    best_nas_params = json.load(f)

# Rebuild model
model = NASNN(
    input_size=15,
    n_layers=best_nas_params["n_layers"],
    hidden_units=best_nas_params["hidden_units"],
    activation_name=best_nas_params["activation"],
    dropout_rate=best_nas_params["dropout"]
)

# Load weights
model.load_state_dict(
    torch.load(
        "final_nas_model.pth",
        weights_only=True
    )
)

model.eval()

print("Model loaded successfully.")