#Load Full Model

checkpoint = torch.load(
    "full_final_model.pth",
    weights_only=False
)

model = checkpoint["model"]

optimizer = checkpoint["optimizer"]

X_train = checkpoint["train_data"]