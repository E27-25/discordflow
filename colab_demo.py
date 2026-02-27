# ============================================================
#  DiscordFlow v0.3.0 × Google Colab Demo
#  Demonstrates BOTH normal channel AND forum channel modes
#  + system metrics + figure logging + session backup
# ============================================================
#  Cell 1 — Install
# !pip install "discordflow[system]" -q

# ============================================================
#  Cell 2 — Config & Setup
# ============================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

from discordflow import DiscordFlow

# ─── Fill in your webhook URLs ───────────────────────────────
NORMAL_WEBHOOK = "YOUR_NORMAL_CHANNEL_WEBHOOK_URL"
FORUM_WEBHOOK  = "YOUR_FORUM_CHANNEL_WEBHOOK_URL"

# Choose one mode — comment out the other
# MODE = "normal"   # ← posts embeds directly into the channel
MODE = "forum"      # ← each run gets its own forum thread

WEBHOOK = NORMAL_WEBHOOK if MODE == "normal" else FORUM_WEBHOOK

dflow = DiscordFlow(
    webhook_url     = WEBHOOK,
    experiment_name = "MNIST_MLP_v2",
    username        = "TrainBot 🏋️",
    avatar_url      = "https://i.imgur.com/AfFp7pu.png",
    async_logging   = True,   # non-blocking — never pauses training
)

# ============================================================
#  Cell 3 — Model & Data
# ============================================================
class MLP(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, 64),      nn.ReLU(),
            nn.Linear(64, 10),
        )
    def forward(self, x): return self.net(x)

BATCH_SIZE  = 256
HIDDEN      = 128
LR          = 1e-3
EPOCHS      = 5

transform    = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_ds     = datasets.MNIST("/tmp/data", train=True,  download=True, transform=transform)
test_ds      = datasets.MNIST("/tmp/data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

device  = "cuda" if torch.cuda.is_available() else "cpu"
model   = MLP(HIDDEN).to(device)
optim_  = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

# ============================================================
#  Cell 4 — Training Loop
# ============================================================
history = {"train_loss": [], "test_loss": [], "test_acc": []}
PARAMS  = {"hidden": HIDDEN, "lr": LR, "batch": BATCH_SIZE, "epochs": EPOCHS, "device": device}
TAGS    = {"dataset": "MNIST", "framework": "PyTorch", "mode": MODE}

# ── Which system metrics to log each epoch ──────────────────
# Remove any you don't need. GPU requires `pip install discordflow[system,gpu]`
SYS_METRICS = ["cpu", "ram"]           # safe for all machines
# SYS_METRICS = ["cpu", "ram", "gpu"] # uncomment if on Colab GPU / NVIDIA

# ── Pick the right context manager based on MODE ─────────────
if MODE == "forum":
    run_ctx = dflow.start_forum_run("mlp_baseline", description="MLP on MNIST, 5 epochs")
else:
    run_ctx = dflow.start_run("mlp_baseline")

with run_ctx as run:
    run.log_params(PARAMS)
    for k, v in TAGS.items():
        run.set_tag(k, v)

    for epoch in range(1, EPOCHS + 1):
        # Train
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

        # Evaluate
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

        print(f"Epoch {epoch}/{EPOCHS}  loss={train_loss:.4f}  val={test_loss:.4f}  acc={test_acc:.2f}%")

        # 📊 Log metrics (with configurable hardware stats)
        run.log_metrics(
            {
                "Train Loss": round(train_loss, 4),
                "Test Loss":  round(test_loss,  4),
                "Test Acc %": round(test_acc,   2),
            },
            step=epoch,
            system_metrics=SYS_METRICS,   # ← adjust this list freely
        )

    # 📈 Upload final loss curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(history["train_loss"], label="Train", marker="o")
    ax1.plot(history["test_loss"],  label="Test",  marker="s")
    ax1.set_title("Loss Curve"); ax1.legend()
    ax2.plot(history["test_acc"], color="green", marker="^")
    ax2.set_title("Test Accuracy (%)")
    plt.tight_layout()
    plt.savefig("/tmp/loss_curve.png", dpi=150)
    plt.show()

    run.log_figure(fig, title="Loss & Accuracy Curves")    # send to Discord
    run.log_text(
        "\n".join([f"{i+1},{tl:.4f},{vl:.4f}" for i, (tl, vl) in
                   enumerate(zip(history["train_loss"], history["test_loss"]))]),
        filename="loss_history.csv",
    )

    # Save model weights
    torch.save(model.state_dict(), "/tmp/mlp_mnist.pt")
    run.log_artifact("/tmp/mlp_mnist.pt")

# 💾 Save forum thread IDs (important for Colab restart recovery!)
if MODE == "forum":
    dflow.save()

# Flush async queue before program exit
dflow.finish()

# ============================================================
#  Cell 5 — Colab Session Backup (forum mode only)
# ============================================================
# Uncomment to download a ZIP backup of your thread IDs to your PC:
# from discordflow.colab_utils import export_session
# export_session(dflow)

# On a FRESH Colab runtime, restore with:
# from discordflow.colab_utils import import_session
# import_session(dflow)

print("\n✅ Training complete! Check your Discord channel/thread.")
