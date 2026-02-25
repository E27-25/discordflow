# ============================================================
#  DiscordFlow × Google Colab Demo
#  Simple MLP trained on MNIST — all metrics go to Discord!
# ============================================================
#
#  HOW TO USE:
#  1. Open a new Colab notebook
#  2. Paste each cell block below into separate Colab cells
#  3. Fill in your WEBHOOK_URL
#  4. Run all cells!
# ============================================================


# ── CELL 1: Install ──────────────────────────────────────────
# !pip install discordflow -q


# ── CELL 2: Imports & Config ─────────────────────────────────
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import io

from discordflow import DiscordFlow

# 🔧 Put your Discord webhook URL here
WEBHOOK_URL = "YOUR_DISCORD_WEBHOOK_URL"

dflow = DiscordFlow(
    webhook_url     = WEBHOOK_URL,
    experiment_name = "MNIST_MLP_Demo",
    username        = "TrainBot 🏋️",           # custom bot name
    avatar_url      = "https://i.imgur.com/AfFp7pu.png",  # ← any public image URL!
)


# ── CELL 3: Define Model ─────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.net(x)


# ── CELL 4: Data Loaders ─────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_ds = datasets.MNIST("/tmp/data", train=True,  download=True, transform=transform)
test_ds  = datasets.MNIST("/tmp/data", train=False, download=True, transform=transform)

BATCH_SIZE = 256
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)


# ── CELL 5: Train ────────────────────────────────────────────
EPOCHS      = 5
LR          = 1e-3
HIDDEN      = 128

device = "cuda" if torch.cuda.is_available() else "cpu"
model  = MLP(hidden=HIDDEN).to(device)
optim_ = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

history = {"train_loss": [], "test_loss": [], "test_acc": []}

with dflow.start_run("mlp_baseline") as run:

    # Log all hyperparameters
    run.log_params({
        "model"      : "MLP",
        "hidden_dim" : HIDDEN,
        "learning_rate": LR,
        "batch_size" : BATCH_SIZE,
        "epochs"     : EPOCHS,
        "optimizer"  : "Adam",
        "device"     : device,
    })
    run.set_tag("dataset", "MNIST")
    run.set_tag("framework", "PyTorch")

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optim_.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            optim_.step()
            total_loss += loss.item() * len(X)

        train_loss = total_loss / len(train_ds)

        # ── Evaluate ──
        model.eval()
        test_loss, correct = 0.0, 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                test_loss += loss_fn(logits, y).item() * len(X)
                correct   += (logits.argmax(1) == y).sum().item()

        test_loss /= len(test_ds)
        test_acc   = correct / len(test_ds) * 100

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(f"Epoch {epoch}/{EPOCHS}  train_loss={train_loss:.4f}  test_loss={test_loss:.4f}  acc={test_acc:.2f}%")

        # 📊 Log metrics to Discord each epoch
        run.log_metrics({
            "Train Loss" : round(train_loss, 4),
            "Test Loss"  : round(test_loss,  4),
            "Test Acc %" : round(test_acc,   2),
        }, step=epoch)

    # ── Upload loss curve as artifact ────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(history["train_loss"], label="Train Loss", marker="o")
    ax1.plot(history["test_loss"],  label="Test Loss",  marker="s")
    ax1.set_title("Loss Curve")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(history["test_acc"], color="green", marker="^")
    ax2.set_title("Test Accuracy (%)")
    ax2.set_xlabel("Epoch")

    plt.tight_layout()
    plt.savefig("/tmp/loss_curve.png", dpi=150)
    plt.show()

    run.log_artifact("/tmp/loss_curve.png")   # 📁 uploaded to Discord

    # Save model weights and upload
    torch.save(model.state_dict(), "/tmp/mlp_mnist.pt")
    run.log_artifact("/tmp/mlp_mnist.pt")     # 📁 weights uploaded too

print("\n✅ Training complete! Check your Discord channel.")
