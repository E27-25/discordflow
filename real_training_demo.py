#!/usr/bin/env python3
"""
DiscordFlow — Real Training Demo
Author : Watin Promfiy
Version: 0.3.3

Trains a real MLP classifier on the Wine dataset (sklearn).
Logs params, per-epoch metrics, system hardware, a loss curve
figure, and a CSV artifact — all to Discord.

Setup
-----
1. pip install "discordflow[system]==0.3.3" scikit-learn matplotlib
2. Fill in WEBHOOK_URL below
3. Set CHANNEL_MODE to "normal" or "forum" to match your webhook channel type
4. Run: python real_training_demo.py (or paste into a Colab cell)
"""

# ─── CONFIG — edit these two lines ──────────────────────────────────────────
WEBHOOK_URL  = "YOUR_WEBHOOK_URL_HERE"
CHANNEL_MODE = "normal"   # "normal"  → text/announcement channel
                           # "forum"   → forum channel (each run = its own thread)
DRY_RUN      = False       # True = print to stdout only, no real Discord calls
# ─────────────────────────────────────────────────────────────────────────────

import time
import math

import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from discordflow import DiscordFlow

# ── 1. Load & split data ─────────────────────────────────────────────────────
data   = load_wine()
X, y   = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── 2. Hyperparameters ───────────────────────────────────────────────────────
PARAMS = {
    "model"        : "MLPClassifier",
    "dataset"      : "Wine (sklearn)",
    "hidden_layers": "(128, 64)",
    "activation"   : "relu",
    "optimizer"    : "adam",
    "learning_rate": 1e-3,
    "max_iter"     : 1,          # we train 1 epoch at a time manually
    "batch_size"   : 32,
    "alpha"        : 1e-4,       # L2 regularisation
    "random_state" : 42,
    "author"       : "Watin Promfiy",
}
EPOCHS     = 20
LOG_EVERY  = 5    # post to Discord every N epochs (keeps message count low)

# ── 3. Build MLPClassifier (warm_start lets us train incrementally) ───────────
clf = MLPClassifier(
    hidden_layer_sizes = (128, 64),
    activation         = "relu",
    solver             = "adam",
    learning_rate_init = PARAMS["learning_rate"],
    max_iter           = 1,
    warm_start         = True,    # ← key: keeps weights between .fit() calls
    alpha              = PARAMS["alpha"],
    batch_size         = PARAMS["batch_size"],
    random_state       = PARAMS["random_state"],
    n_iter_no_change   = 9999,   # disable early stopping for demo
)

# ── 4. Init DiscordFlow ───────────────────────────────────────────────────────
dflow = DiscordFlow(
    webhook_url     = WEBHOOK_URL,
    experiment_name = "WP_WineClassifier",
    username        = "TrainBot | Watin Promfiy",
    async_logging   = False,   # sync — easier to follow for a demo
    dry_run         = DRY_RUN,
)

# ── 5. Train and log ──────────────────────────────────────────────────────────
history = {"epoch": [], "train_loss": [], "val_loss": [], "val_acc": []}

def run_training(run):
    """Core training loop — called inside the with-block."""
    run.log_params(PARAMS)
    run.set_tag("framework", "scikit-learn")
    run.set_tag("author",    "Watin Promfiy")

    print(f"\n{'─'*55}")
    print(f"  Training MLPClassifier on Wine dataset")
    print(f"  Epochs: {EPOCHS}  |  Log every: {LOG_EVERY} epochs")
    print(f"{'─'*55}\n")

    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        # ── One epoch of training ────────────────────────────────────────
        clf.fit(X_train, y_train)

        # ── Compute metrics ──────────────────────────────────────────────
        prob_train = clf.predict_proba(X_train)
        prob_val   = clf.predict_proba(X_test)
        pred_val   = clf.predict(X_test)

        t_loss = round(log_loss(y_train, prob_train), 5)
        v_loss = round(log_loss(y_test,  prob_val),   5)
        v_acc  = round(accuracy_score(y_test, pred_val), 4)

        history["epoch"].append(epoch)
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)

        elapsed = time.time() - t0
        print(
            f"  Epoch {epoch:>2}/{EPOCHS}  "
            f"train_loss={t_loss:.5f}  val_loss={v_loss:.5f}  "
            f"val_acc={v_acc:.4f}  ({elapsed:.1f}s)"
        )

        # ── Post to Discord every LOG_EVERY epochs ───────────────────────
        if epoch % LOG_EVERY == 0 or epoch == EPOCHS:
            run.log_metrics(
                {
                    "Train Loss" : t_loss,
                    "Val Loss"   : v_loss,
                    "Val Acc"    : f"{v_acc*100:.1f}%",
                },
                step           = epoch,
                system_metrics = ["cpu", "ram", "gpu"],
            )

    # ── Final figure: Loss + Accuracy curves ─────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Wine Classifier — Watin Promfiy", fontsize=13)

    ax1.plot(history["epoch"], history["train_loss"],
             "o-", color="#3498DB", label="Train Loss", markersize=4)
    ax1.plot(history["epoch"], history["val_loss"],
             "s--", color="#E74C3C", label="Val Loss", markersize=4)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Log Loss")
    ax1.set_title("Loss Curves"); ax1.legend()

    ax2.plot(history["epoch"], [a * 100 for a in history["val_acc"]],
             "o-", color="#2ECC71", markersize=4)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Validation Accuracy")

    plt.tight_layout()
    run.log_figure(fig, title="Wine Classifier — Loss & Accuracy (Watin Promfiy)")
    plt.close(fig)

    # ── Final CSV artifact ────────────────────────────────────────────────
    csv_lines = ["epoch,train_loss,val_loss,val_acc"]
    for e, tl, vl, va in zip(
        history["epoch"], history["train_loss"],
        history["val_loss"], history["val_acc"]
    ):
        csv_lines.append(f"{e},{tl},{vl},{va}")
    run.log_text("\n".join(csv_lines), filename="training_history.csv")

    print(f"\n  ✅  Final  val_acc={v_acc*100:.1f}%  val_loss={v_loss:.5f}")

# ── 6. Start run ──────────────────────────────────────────────────────────────
if CHANNEL_MODE == "forum":
    with dflow.start_forum_run(
        "wine-mlp-v1",
        description="MLP on Wine dataset — Watin Promfiy"
    ) as run:
        run_training(run)
    dflow.save()   # persist thread ID for next session
else:
    with dflow.start_run("wine-mlp-v1") as run:
        run_training(run)

dflow.finish()

print("\n  Done! Check your Discord channel for the results.")
