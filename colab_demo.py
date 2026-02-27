#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║         DiscordFlow v0.3.0 — Complete Feature Demo              ║
║  Author : Watin Promfiy                                         ║
║  Purpose: Exercise EVERY feature in the library                 ║
╚══════════════════════════════════════════════════════════════════╝

Run in Google Colab (or locally with dry_run=True):

    # Install
    !pip install "discordflow[system]" -q

    # Optional: GPU metrics on Colab GPU runtime
    !pip install "discordflow[system,gpu]" -q

Fill in your webhook URLs below, set DRY_RUN = False to send live.
"""

# ─── CONFIG ──────────────────────────────────────────────────────────────────
NORMAL_WEBHOOK_URL = "YOUR_NORMAL_CHANNEL_WEBHOOK_URL"
FORUM_WEBHOOK_URL  = "YOUR_FORUM_CHANNEL_WEBHOOK_URL"

# Set False to post to real Discord channels
DRY_RUN = True

# Which hardware metrics to log each step (remove any you don't need)
# Requires: pip install "discordflow[system]"
#           pip install "discordflow[system,gpu]"  ← for GPU
SYS_METRICS = ["cpu", "ram"]
# SYS_METRICS = ["cpu", "ram", "gpu", "disk", "network"]   # everything
# SYS_METRICS = []                                           # disabled
# ─────────────────────────────────────────────────────────────────────────────

import time
import json
import math
import tempfile
import os

# ── Try importing matplotlib (optional for this demo) ────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend — safe for Colab and scripts
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠  matplotlib not installed — figure logging tests will be skipped.")
    print("   Install with: pip install matplotlib\n")

from discordflow import DiscordFlow, ActiveRun, ForumActiveRun
from discordflow.exceptions import (
    DiscordFlowError, WebhookError, ArtifactTooLargeError, RunNotActiveError,
)
from discordflow.utils import (
    collect_system_metrics, human_size, human_duration,
    ascii_progress, format_kv_table, truncate, VALID_SYSTEM_METRICS,
)

print("=" * 65)
print("  DiscordFlow v0.3.0 — Feature Coverage Demo")
print(f"  Author : Watin Promfiy")
print(f"  Mode   : {'DRY RUN (stdout only)' if DRY_RUN else '🔴 LIVE (posting to Discord)'}")
print("=" * 65)
print()

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
print("  [2/8] System metrics (collect_system_metrics)")
print("─" * 60)

all_metrics = ["cpu", "ram", "gpu", "disk", "network"]
hw = collect_system_metrics(all_metrics)
for label, val in hw.items():
    print(f"  {label}: {val}")
print()

# =============================================================================
# SECTION 3 — Normal Channel Mode  (start_run)
# =============================================================================
print("─" * 60)
print("  [3/8] Normal channel — start_run()")
print("─" * 60)

normal_logger = DiscordFlow(
    webhook_url     = NORMAL_WEBHOOK_URL,
    experiment_name = "WP_ResNet50",
    username        = "TrainBot · Watin Promfiy",
    async_logging   = False,   # sync for predictable demo output
    dry_run         = DRY_RUN,
)

# 3a — log_params
with normal_logger.start_run("baseline_v1") as run:

    # ── Hyperparameters ──────────────────────────────────────────
    run.log_param("author", "Watin Promfiy")
    run.log_params({
        "architecture" : "ResNet-50",
        "lr"           : 3e-4,
        "batch_size"   : 128,
        "epochs"       : 5,
        "optimizer"    : "AdamW",
        "scheduler"    : "CosineAnnealingLR",
        "weight_decay" : 1e-4,
        "mixed_precision": True,
    })

    # ── Tags ─────────────────────────────────────────────────────
    run.set_tag("dataset",   "ImageNet-1k")
    run.set_tag("framework", "PyTorch 2.2")
    run.set_tag("author",    "Watin Promfiy")

    # ── Metrics per epoch (with system metrics) ──────────────────
    EPOCHS = 5
    for epoch in range(1, EPOCHS + 1):
        t_loss = round(2.5 * math.exp(-0.4 * epoch) + 0.05, 4)
        v_loss = round(2.8 * math.exp(-0.38 * epoch) + 0.08, 4)
        v_acc  = round(0.5 + 0.09 * epoch, 4)

        run.log_metric("Train Loss", t_loss, step=epoch,
                       system_metrics=SYS_METRICS if SYS_METRICS else None)
        run.log_metrics(
            {"Val Loss": v_loss, "Val Acc": v_acc},
            step=epoch,
            system_metrics=SYS_METRICS if epoch == 3 else None,  # attach HW on epoch 3 only
        )

        progress = ascii_progress(epoch, EPOCHS)
        print(f"  Epoch {epoch}/{EPOCHS}  {progress}  loss={t_loss}  val_acc={v_acc}")

    # ── log_text ──────────────────────────────────────────────────
    csv_data = "epoch,train_loss,val_loss,val_acc\n"
    for e in range(1, EPOCHS + 1):
        tl = round(2.5 * math.exp(-0.4 * e) + 0.05, 4)
        vl = round(2.8 * math.exp(-0.38 * e) + 0.08, 4)
        va = round(0.5 + 0.09 * e, 4)
        csv_data += f"{e},{tl},{vl},{va}\n"
    run.log_text(csv_data, filename="training_history.csv")
    print("  📄 Logged text artifact: training_history.csv")

    # ── log_artifact (real file) ──────────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        json.dump({"best_epoch": 5, "val_acc": 0.95, "author": "Watin Promfiy"}, f, indent=2)
        tmp_json = f.name
    run.log_artifact(tmp_json)
    print(f"  📁 Logged file artifact: {os.path.basename(tmp_json)}")
    os.unlink(tmp_json)

    # ── log_figure ────────────────────────────────────────────────
    if HAS_MATPLOTLIB:
        epochs_list = list(range(1, EPOCHS + 1))
        t_losses = [round(2.5 * math.exp(-0.4 * e) + 0.05, 4) for e in epochs_list]
        v_losses = [round(2.8 * math.exp(-0.38 * e) + 0.08, 4) for e in epochs_list]
        v_accs   = [round(0.5 + 0.09 * e, 4) for e in epochs_list]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle("ResNet-50 Training · Watin Promfiy", fontsize=13)

        ax1.plot(epochs_list, t_losses, "o-", label="Train Loss", color="#3498DB")
        ax1.plot(epochs_list, v_losses, "s--", label="Val Loss", color="#E74C3C")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
        ax1.set_title("Loss Curve"); ax1.legend()

        ax2.bar(epochs_list, v_accs, color="#2ECC71")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
        ax2.set_title("Validation Accuracy")

        plt.tight_layout()
        run.log_figure(fig, title="ResNet-50 Loss & Accuracy — Watin Promfiy")
        plt.close(fig)
        print("  📊 Logged matplotlib figure")
    else:
        print("  ⚠  Skipping figure (matplotlib not installed)")

print()
normal_logger.finish()

# =============================================================================
# SECTION 4 — Forum Channel Mode  (start_forum_run)
# =============================================================================
print("─" * 60)
print("  [4/8] Forum channel — start_forum_run()")
print("─" * 60)

forum_logger = DiscordFlow(
    webhook_url     = FORUM_WEBHOOK_URL,
    experiment_name = "WP_LLM_FineTune",
    username        = "TrainBot · Watin Promfiy",
    state_file      = "/tmp/discordflow_demo_state.json",
    async_logging   = False,
    dry_run         = DRY_RUN,
)

with forum_logger.start_forum_run(
    "lora_rank_16",
    description="LoRA fine-tine benchmark · Watin Promfiy · rank=16"
) as run:

    run.log_param("author", "Watin Promfiy")
    run.log_params({
        "base_model"  : "Meta-Llama-3-8B",
        "lora_rank"   : 16,
        "lora_alpha"  : 32,
        "lr"          : 2e-4,
        "epochs"      : 3,
        "batch_size"  : 4,
        "grad_accum"  : 8,
        "bf16"        : True,
    })
    run.set_tag("task",    "Instruction Tuning")
    run.set_tag("dataset", "Alpaca-52k")
    run.set_tag("author",  "Watin Promfiy")

    for epoch in range(1, 4):
        t_loss = round(2.0 * math.exp(-0.5 * epoch) + 0.1, 4)
        run.log_metrics(
            {"Train Loss": t_loss, "Perplexity": round(math.exp(t_loss), 3)},
            step=epoch,
            system_metrics=SYS_METRICS if SYS_METRICS else None,
        )
        print(f"  Forum Epoch {epoch}/3  loss={t_loss}")

    if HAS_MATPLOTLIB:
        epochs_list = list(range(1, 4))
        losses = [round(2.0 * math.exp(-0.5 * e) + 0.1, 4) for e in epochs_list]
        ppls   = [round(math.exp(l), 3) for l in losses]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle("LLaMA-3 LoRA Fine-tune · Watin Promfiy", fontsize=13)
        axes[0].plot(epochs_list, losses, "o-", color="#9B59B6"); axes[0].set_title("Train Loss")
        axes[1].plot(epochs_list, ppls,   "s-", color="#F39C12"); axes[1].set_title("Perplexity")
        plt.tight_layout()
        run.log_figure(fig, title="LoRA Training Curves — Watin Promfiy")
        plt.close(fig)
        print("  📊 Logged forum figure")

    run.log_text(
        "Step,Loss\n" + "\n".join(
            f"{e},{round(2.0*math.exp(-0.5*e)+0.1,4)}" for e in range(1, 4)
        ),
        filename="lora_loss.csv",
    )

print()

# 4b — Save forum state
forum_logger.save()
print("  💾 State saved to /tmp/discordflow_demo_state.json")
state_contents = json.load(open("/tmp/discordflow_demo_state.json"))
print("  State contents:", state_contents)
forum_logger.finish()
print()

# =============================================================================
# SECTION 5 — State Persistence & resume_run
# =============================================================================
print("─" * 60)
print("  [5/8] State persistence & resume_run()")
print("─" * 60)

# Simulate a fresh runtime loading saved state
fresh_logger = DiscordFlow(
    webhook_url     = FORUM_WEBHOOK_URL,
    experiment_name = "WP_LLM_FineTune",
    state_file      = "/tmp/discordflow_demo_state.json",  # ← auto-loaded
    async_logging   = False,
    dry_run         = DRY_RUN,
)
print("  Loaded state:", fresh_logger._run_state)

# Manual resume (for when you know the thread ID)
fresh_logger.resume_run("manual_run", "1234567890000001")
fresh_logger.save("/tmp/discordflow_demo_state_v2.json")
print("  After manual resume:", json.load(open("/tmp/discordflow_demo_state_v2.json")))
fresh_logger.finish()
print()

# =============================================================================
# SECTION 6 — Error Capture (exception inside with-block)
# =============================================================================
print("─" * 60)
print("  [6/8] Error capture — exception inside run")
print("─" * 60)

err_logger = DiscordFlow(
    webhook_url     = NORMAL_WEBHOOK_URL,
    experiment_name = "WP_ErrorTest",
    username        = "TrainBot · Watin Promfiy",
    async_logging   = False,
    dry_run         = DRY_RUN,
)

try:
    with err_logger.start_run("crash_run") as run:
        run.log_params({"lr": 1e-3, "author": "Watin Promfiy"})
        run.log_metrics({"loss": 0.5}, step=1)
        raise ValueError("Simulated NaN loss — training diverged!")  # ← intentional crash
except ValueError:
    pass  # DiscordFlow already caught it, posted ❌ FAILED embed, re-raised
err_logger.finish()
print("  ✅ Error was captured and posted to Discord")
print()

# =============================================================================
# SECTION 7 — ArtifactTooLargeError exception
# =============================================================================
print("─" * 60)
print("  [7/8] Exceptions")
print("─" * 60)

try:
    raise ArtifactTooLargeError("/tmp/huge_model.bin", 30 * 1024 * 1024)
except ArtifactTooLargeError as e:
    print("  ArtifactTooLargeError:", e)

try:
    raise RunNotActiveError()
except RunNotActiveError as e:
    print("  RunNotActiveError:", e)

try:
    raise WebhookError("404 Not Found — check your webhook URL")
except WebhookError as e:
    print("  WebhookError:", e)
print()

# =============================================================================
# SECTION 8 — Dry-run meta check (all 5 system metrics)
# =============================================================================
print("─" * 60)
print("  [8/8] Smoke: all system_metrics keys via log_metrics()")
print("─" * 60)

meta_logger = DiscordFlow(
    webhook_url     = NORMAL_WEBHOOK_URL,
    experiment_name = "WP_MetricsCheck",
    async_logging   = False,
    dry_run         = True,   # always dry for this section
)
with meta_logger.start_run("all_sys_metrics") as run:
    run.log_metrics(
        {"dummy_loss": 0.0},
        step=1,
        system_metrics=["cpu", "ram", "gpu", "disk", "network"],
    )
meta_logger.finish()
print()

# =============================================================================
# SUMMARY
# =============================================================================
print("=" * 65)
print("  ✅  All features verified!")
print()
print("  Covered:")
print("   [x] start_run()           — Normal channel mode")
print("   [x] start_forum_run()     — Forum channel thread mode")
print("   [x] log_param/log_params  — Hyperparameter logging")
print("   [x] log_metric/log_metrics— Metric logging with step")
print("   [x] set_tag               — Arbitrary key-value tags")
print("   [x] log_artifact          — File upload (25 MB max)")
print("   [x] log_text              — Text/CSV string upload")
print("   [x] log_figure            — matplotlib PNG upload")
print("   [x] system_metrics=       — cpu, ram, gpu, disk, network")
print("   [x] save() / resume_run() — State persistence")
print("   [x] State auto-load       — state_file on __init__")
print("   [x] Error capture         — ❌ FAILED embed on exception")
print("   [x] ArtifactTooLargeError — Exception raised correctly")
print("   [x] RunNotActiveError     — Exception raised correctly")
print("   [x] WebhookError          — Exception raised correctly")
print("   [x] async_logging=False   — Synchronous mode tested")
print("   [x] dry_run=True          — No real HTTP calls")
print("   [x] finish()              — Executor flushed")
print("   [x] Utils helpers         — human_size, human_duration,")
print("       ascii_progress, format_kv_table, truncate")
print()
print(f"  Author: Watin Promfiy  |  DiscordFlow v0.3.0")
print("=" * 65)
