#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║      DiscordFlow v0.3.5 — Complete Feature Demo                 ║
║      Real Training: MLPClassifier on Wine + LDA on Iris         ║
║      Author : Watin Promfiy                                     ║
╚══════════════════════════════════════════════════════════════════╝

Install
-------
!pip install "discordflow[system]==0.3.5" scikit-learn matplotlib -q

Config
------
Set WEBHOOK_URL and CHANNEL_MODE below, then run.
"""

# ─── CONFIG ──────────────────────────────────────────────────────────────────
WEBHOOK_URL  = "YOUR_WEBHOOK_URL_HERE"

# "normal" → text channel webhook   → uses start_run()
# "forum"  → forum channel webhook  → uses start_forum_run() (each run = own thread)
CHANNEL_MODE = "forum"

DRY_RUN = False  # True = print to stdout, no real Discord calls

# Hardware metrics logged every epoch (remove any you don't have)
SYS_METRICS = ["cpu", "ram", "gpu"]
# SYS_METRICS = []   # disable HW logging
# ─────────────────────────────────────────────────────────────────────────────

# ── Stdlib ────────────────────────────────────────────────────────────────────
import time
import json
import tempfile
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)   # suppress sklearn ConvergenceWarning

# ── ML ────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.datasets import load_wine, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, log_loss

# ── Plotting ─────────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("⚠ matplotlib not installed — figure tests skipped")

# ── DiscordFlow ───────────────────────────────────────────────────────────────
from discordflow import DiscordFlow, ActiveRun, ForumActiveRun
from discordflow.exceptions import (
    DiscordFlowError, WebhookError, ArtifactTooLargeError, RunNotActiveError,
)
from discordflow.utils import (
    collect_system_metrics, human_size, human_duration,
    ascii_progress, format_kv_table, truncate, VALID_SYSTEM_METRICS,
)

print("=" * 65)
print("  DiscordFlow v0.3.5 — Complete Feature Demo")
print(f"  Author  : Watin Promfiy")
print(f"  Mode    : {'DRY RUN' if DRY_RUN else '🔴 LIVE → Discord'}")
print(f"  Channel : {CHANNEL_MODE.upper()}")
print("=" * 65, "\n")

# =============================================================================
# SECTION 1 — Utils
# =============================================================================
print("─" * 60)
print("  [1/8] Utility helpers")
print("─" * 60)
print("  human_size(1536)       :", human_size(1536))
print("  human_size(27_000_000) :", human_size(27_000_000))
print("  human_duration(65)     :", human_duration(65))
print("  human_duration(3723)   :", human_duration(3723))
print("  ascii_progress(7, 10)  :", ascii_progress(7, 10))
print("  truncate('x'*2000, 10) :", truncate("x" * 2000, 10))
print("  VALID_SYSTEM_METRICS   :", sorted(VALID_SYSTEM_METRICS))
print()

# =============================================================================
# SECTION 2 — System Metrics
# =============================================================================
print("─" * 60)
print("  [2/8] System metrics")
print("─" * 60)
hw = collect_system_metrics(["cpu", "ram", "gpu", "disk", "network"])
for k, v in hw.items():
    print(f"  {k}: {v}")
print()

# =============================================================================
# Helper: prepare a sklearn dataset + scaler
# =============================================================================
def prepare_wine():
    d = load_wine()
    X_tr, X_te, y_tr, y_te = train_test_split(
        d.data, d.target, test_size=0.2, random_state=42, stratify=d.target
    )
    sc = StandardScaler()
    return sc.fit_transform(X_tr), sc.transform(X_te), y_tr, y_te, d.target_names

def prepare_iris():
    d = load_iris()
    X_tr, X_te, y_tr, y_te = train_test_split(
        d.data, d.target, test_size=0.2, random_state=7, stratify=d.target
    )
    sc = StandardScaler()
    return sc.fit_transform(X_tr), sc.transform(X_te), y_tr, y_te, d.target_names

# =============================================================================
# SECTION 3 — Normal / Forum Channel — real MLP training on Wine
# =============================================================================
print("─" * 60)
print("  [3/8] Primary run — MLP on Wine (all logging features)")
print("─" * 60)

X_tr, X_te, y_tr, y_te, cls_names = prepare_wine()

EPOCHS   = 20
LOG_EVERY = 5   # post metric embed every N epochs → keeps Discord tidy

clf_wine = MLPClassifier(
    hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
    learning_rate_init=1e-3, max_iter=1, warm_start=True,
    alpha=1e-4, random_state=42, n_iter_no_change=9999,
)

history_wine = {"epoch": [], "train_loss": [], "val_loss": [], "val_acc": []}

dflow = DiscordFlow(
    webhook_url     = WEBHOOK_URL,
    experiment_name = "WP_WineClassifier",
    username        = "TrainBot | Watin Promfiy",
    state_file      = "discordflow_state.json",   # saved in current dir (visible in Colab)
    async_logging   = False,
    dry_run         = DRY_RUN,
)

def train_wine(run):
    run.log_param("author", "Watin Promfiy")
    run.log_params({
        "dataset"       : "Wine (sklearn)",
        "model"         : "MLPClassifier",
        "hidden_layers" : "(128, 64)",
        "activation"    : "relu",
        "optimizer"     : "adam",
        "lr"            : 1e-3,
        "alpha_l2"      : 1e-4,
        "batch_size"    : 32,
        "epochs"        : EPOCHS,
        "log_every"     : LOG_EVERY,
    })
    run.set_tag("framework", "scikit-learn")
    run.set_tag("dataset",   "Wine (3 classes, 13 features)")
    run.set_tag("author",    "Watin Promfiy")

    print(f"\n  Training {EPOCHS} epochs …\n")
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        clf_wine.fit(X_tr, y_tr)

        p_tr = clf_wine.predict_proba(X_tr)
        p_te = clf_wine.predict_proba(X_te)
        t_loss = round(log_loss(y_tr, p_tr), 5)
        v_loss = round(log_loss(y_te, p_te), 5)
        v_acc  = round(accuracy_score(y_te, clf_wine.predict(X_te)), 4)

        history_wine["epoch"].append(epoch)
        history_wine["train_loss"].append(t_loss)
        history_wine["val_loss"].append(v_loss)
        history_wine["val_acc"].append(v_acc)

        bar = ascii_progress(epoch, EPOCHS)
        print(f"  Epoch {epoch:>2}/{EPOCHS}  {bar}  "
              f"train_loss={t_loss:.5f}  val_acc={v_acc*100:.1f}%")

        if epoch % LOG_EVERY == 0 or epoch == EPOCHS:
            run.log_metrics(
                {"Train Loss": t_loss, "Val Loss": v_loss,
                 "Val Acc": f"{v_acc*100:.1f}%"},
                step=epoch,
                system_metrics=SYS_METRICS if SYS_METRICS else None,
            )

    elapsed = time.time() - t0
    print(f"\n  ✅  Done in {human_duration(elapsed)}"
          f"  |  Final val_acc={v_acc*100:.1f}%\n")

    # ── log_figure ──────────────────────────────────────────────────────────
    if HAS_MPL:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle("MLP on Wine — Watin Promfiy", fontsize=13)

        ax1.plot(history_wine["epoch"], history_wine["train_loss"],
                 "o-", color="#3498DB", label="Train", markersize=4)
        ax1.plot(history_wine["epoch"], history_wine["val_loss"],
                 "s--", color="#E74C3C", label="Val", markersize=4)
        ax1.set(xlabel="Epoch", ylabel="Log Loss", title="Loss Curves")
        ax1.legend()

        ax2.plot(history_wine["epoch"],
                 [a * 100 for a in history_wine["val_acc"]],
                 "o-", color="#2ECC71", markersize=4)
        ax2.set(xlabel="Epoch", ylabel="Accuracy (%)", title="Val Accuracy")

        plt.tight_layout()
        run.log_figure(fig, title="Wine Classifier — Loss & Accuracy (Watin Promfiy)")
        plt.close(fig)
        print("  📊 Figure logged")

    # ── log_text (CSV) ───────────────────────────────────────────────────────
    csv = "epoch,train_loss,val_loss,val_acc\n" + "\n".join(
        f"{e},{tl},{vl},{va}"
        for e, tl, vl, va in zip(
            history_wine["epoch"], history_wine["train_loss"],
            history_wine["val_loss"], history_wine["val_acc"]
        )
    )
    run.log_text(csv, filename="wine_history.csv")
    print("  📄 CSV logged")

    # ── log_artifact (temp JSON) ─────────────────────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        json.dump({
            "best_val_acc": max(history_wine["val_acc"]),
            "final_val_acc": history_wine["val_acc"][-1],
            "author": "Watin Promfiy",
        }, f, indent=2)
        tmp_json = f.name
    run.log_artifact(tmp_json)
    os.unlink(tmp_json)
    print("  📁 JSON artifact logged")

# Launch with correct channel mode
if CHANNEL_MODE == "forum":
    with dflow.start_forum_run(
        "wine-mlp-v1",
        description="MLP on Wine — Watin Promfiy"
    ) as run:
        train_wine(run)
    dflow.save()   # ← saves thread ID so run can be resumed after Colab restart
else:
    with dflow.start_run("wine-mlp-v1") as run:
        train_wine(run)

dflow.finish()
print()

# =============================================================================
# SECTION 4 — Forum Thread Resumption / Second Run (LDA on Iris)
# =============================================================================
print("─" * 60)
print("  [4/8] Second run — LDA on Iris (forum resume / second channel)")
print("─" * 60)

X_tr2, X_te2, y_tr2, y_te2, _ = prepare_iris()

clf_lda = LinearDiscriminantAnalysis()
clf_lda.fit(X_tr2, y_tr2)

lda_acc  = round(accuracy_score(y_te2, clf_lda.predict(X_te2)), 4)
lda_prob = clf_lda.predict_proba(X_te2)
lda_loss = round(log_loss(y_te2, lda_prob), 5)

dflow2 = DiscordFlow(
    webhook_url     = WEBHOOK_URL,
    experiment_name = "WP_IrisLDA",
    username        = "TrainBot | Watin Promfiy",
    state_file      = "discordflow_state.json",   # same state file — merges thread IDs
    async_logging   = False,
    dry_run         = DRY_RUN,
)

def train_lda(run):
    run.log_param("author", "Watin Promfiy")
    run.log_params({
        "dataset"   : "Iris (sklearn)",
        "model"     : "LinearDiscriminantAnalysis",
        "solver"    : "svd",
        "n_classes" : 3,
    })
    run.set_tag("author",   "Watin Promfiy")
    run.set_tag("dataset",  "Iris (3 classes, 4 features)")
    run.set_tag("framework","scikit-learn")

    # LDA is a single-pass algorithm — one result, logged as step=1
    run.log_metrics(
        {"Val Loss": lda_loss, "Val Acc": f"{lda_acc*100:.1f}%"},
        step=1,
        system_metrics=SYS_METRICS if SYS_METRICS else None,
    )

    if HAS_MPL:
        # Plot 2D LDA projection
        lda_2d = LinearDiscriminantAnalysis(n_components=2)
        X_all  = np.vstack([X_tr2, X_te2])
        y_all  = np.concatenate([y_tr2, y_te2])
        X_proj = lda_2d.fit_transform(X_all, y_all)

        fig, ax = plt.subplots(figsize=(7, 5))
        colors  = ["#3498DB", "#E74C3C", "#2ECC71"]
        for cls, col in enumerate(colors):
            idx = y_all == cls
            ax.scatter(X_proj[idx, 0], X_proj[idx, 1],
                       c=col, label=f"Class {cls}", alpha=0.7, edgecolors="k", s=50)
        ax.set(title="LDA Projection — Iris (Watin Promfiy)",
               xlabel="LD1", ylabel="LD2")
        ax.legend()
        plt.tight_layout()
        run.log_figure(fig, title="Iris LDA 2D Projection — Watin Promfiy")
        plt.close(fig)
        print("  📊 LDA projection figure logged")

    run.log_text(
        f"model,val_loss,val_acc\nLDA,{lda_loss},{lda_acc}",
        filename="iris_lda_result.csv",
    )
    print(f"  ✅  LDA val_acc={lda_acc*100:.1f}%  val_loss={lda_loss}")

if CHANNEL_MODE == "forum":
    with dflow2.start_forum_run(
        "iris-lda-v1",
        description="LDA on Iris — Watin Promfiy"
    ) as run:
        train_lda(run)
    dflow2.save()   # ← merges iris thread ID into same state file
else:
    with dflow2.start_run("iris-lda-v1") as run:
        train_lda(run)

dflow2.finish()
print()

# =============================================================================
# SECTION 5 — State Persistence & resume_run
# =============================================================================
print("─" * 60)
print("  [5/8] State persistence & resume_run()")
print("─" * 60)

fresh = DiscordFlow(
    webhook_url     = WEBHOOK_URL,
    experiment_name = "WP_StateTest",
    state_file      = "discordflow_state.json",
    dry_run         = DRY_RUN,
    async_logging   = False,
)
dflow.save("discordflow_state.json")
fresh2 = DiscordFlow(
    webhook_url     = WEBHOOK_URL,
    experiment_name = "WP_StateTest",
    state_file      = "discordflow_state.json",
    dry_run         = DRY_RUN,
    async_logging   = False,
)
print("  Auto-loaded state:", fresh2._run_state)

fresh2.resume_run("manual_run", "1234567890000001")
fresh2.save()   # saves to discordflow_state.json in current dir
print("  After resume_run:", json.load(open("discordflow_state.json")))
fresh2.finish()
print()

# =============================================================================
# SECTION 6 — Error Capture
# =============================================================================
print("─" * 60)
print("  [6/8] Error capture — exception inside run")
print("─" * 60)

err_logger = DiscordFlow(
    webhook_url     = WEBHOOK_URL,
    experiment_name = "WP_ErrorTest",
    username        = "TrainBot | Watin Promfiy",
    async_logging   = False,
    dry_run         = DRY_RUN,
)

if CHANNEL_MODE == "forum":
    ctx = err_logger.start_forum_run("crash-run", "Error test")
else:
    ctx = err_logger.start_run("crash-run")

try:
    with ctx as run:
        run.log_params({"lr": 1e-3, "author": "Watin Promfiy"})
        run.log_metrics({"loss": 0.5}, step=1)
        raise ValueError("Simulated NaN loss — training diverged!")
except ValueError:
    pass

err_logger.finish()
print("  ✅  Crash captured and posted to Discord as ❌ FAILED\n")

# =============================================================================
# SECTION 7 — Exceptions
# =============================================================================
print("─" * 60)
print("  [7/8] Exceptions")
print("─" * 60)

try:
    raise ArtifactTooLargeError("/tmp/huge.bin", 30 * 1024 * 1024)
except ArtifactTooLargeError as e:
    print("  ArtifactTooLargeError:", e)

try:
    raise RunNotActiveError()
except RunNotActiveError as e:
    print("  RunNotActiveError    :", e)

try:
    raise WebhookError("404 Not Found — check your webhook URL")
except WebhookError as e:
    print("  WebhookError         :", e)
print()

# =============================================================================
# SECTION 8 — All system_metrics keys via dry-run
# =============================================================================
print("─" * 60)
print("  [8/8] All system_metrics smoke test")
print("─" * 60)

meta = DiscordFlow(
    webhook_url     = WEBHOOK_URL,
    experiment_name = "WP_MetricsCheck",
    async_logging   = False,
    dry_run         = True,   # always dry for this check
)
if CHANNEL_MODE == "forum":
    ctx = meta.start_forum_run("sys-metrics-check", "HW check")
else:
    ctx = meta.start_run("sys-metrics-check")

with ctx as run:
    run.log_metrics(
        {"dummy": 0.0}, step=1,
        system_metrics=["cpu", "ram", "gpu", "disk", "network"],
    )
meta.finish()
print()

# =============================================================================
# SUMMARY
# =============================================================================
print("=" * 65)
print("  ✅  All features verified with REAL training!")
print()
print("  Covered:")
print("   [x] Real MLP training  — Wine dataset (sklearn, 20 epochs)")
print("   [x] Real LDA training  — Iris dataset (single pass)")
print("   [x] start_run / start_forum_run — Normal & Forum modes")
print("   [x] log_param / log_params      — Hyperparameters")
print("   [x] log_metric / log_metrics    — Per-epoch real metrics")
print("   [x] set_tag                     — Experiment labels")
print("   [x] log_artifact                — JSON file upload")
print("   [x] log_text                    — CSV string upload")
print("   [x] log_figure                  — Loss curve + LDA plot")
print("   [x] system_metrics=             — cpu/ram/gpu/disk/network")
print("   [x] save() / resume_run()       — State persistence")
print("   [x] State auto-load             — state_file on __init__")
print("   [x] Error capture               — ❌ FAILED embed")
print("   [x] ArtifactTooLargeError / RunNotActiveError / WebhookError")
print("   [x] async_logging=False / dry_run / finish()")
print("   [x] Utils: human_size, human_duration, ascii_progress, truncate")
print()
print(f"  Author: Watin Promfiy  |  DiscordFlow v0.3.5")
print("=" * 65)
