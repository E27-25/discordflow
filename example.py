#!/usr/bin/env python3
"""
example.py — DiscordFlow demo script (dry_run mode)
====================================================
Run this locally to see exactly what DiscordFlow would post to Discord,
without needing a real webhook URL.

    python example.py

To use a real webhook, set the DISCORD_WEBHOOK_URL environment variable
and change dry_run=False below.
"""

import os
import math
from discordflow import DiscordFlow

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/FAKE/URL")

# dry_run=True → prints to stdout, makes zero HTTP calls
dflow = DiscordFlow(
    WEBHOOK_URL,
    experiment_name="MoE_Router_Training_v2",
    dry_run=True,
)

# -------------------------------------------------------------------
# Example 1: Simple log_param / log_metrics (no run context)
# -------------------------------------------------------------------
print("\n=== Example 1: Standalone logging ===\n")

dflow.log_param("model_type", "MixtureOfExperts")
dflow.log_param("num_experts", 8)
dflow.log_param("routing_strategy", "top-k")

dflow.log_params({
    "learning_rate": 3e-4,
    "batch_size": 128,
    "warmup_steps": 500,
    "weight_decay": 0.01,
})

epochs = 5
for epoch in range(1, epochs + 1):
    loss = 1.0 / epoch
    expert_balance = 0.8 + (0.02 * epoch)
    perplexity = math.exp(loss)

    dflow.log_metrics({
        "Train Loss": round(loss, 4),
        "Load Balance": round(expert_balance, 4),
        "Perplexity": round(perplexity, 4),
    }, step=epoch)

dflow.set_tag("author", "e27")
dflow.set_tag("dataset", "openwebtext")

# Simulate artifact logging (no file needed in dry_run)
# dflow.log_artifact("router_weights.pt")
# dflow.log_artifact("loss_curve.png")

dflow.log_text("epoch,loss,perplexity\n1,1.0,2.718\n2,0.5,1.649\n3,0.33,1.391", filename="metrics.csv")

# -------------------------------------------------------------------
# Example 2: Context manager (ActiveRun) — auto summary on exit
# -------------------------------------------------------------------
print("\n\n=== Example 2: Context manager (start_run) ===\n")

dflow2 = DiscordFlow(
    WEBHOOK_URL,
    experiment_name="GPT2_Fine_Tune",
    dry_run=True,
)

with dflow2.start_run("lora_rank_16") as run:
    run.log_params({
        "base_model": "gpt2-medium",
        "lora_rank": 16,
        "lora_alpha": 32,
        "epochs": 3,
        "lr": 2e-4,
    })

    run.set_tag("framework", "HuggingFace")
    run.set_tag("task", "text-classification")

    for epoch in range(1, 4):
        train_loss = 2.5 / epoch
        val_loss   = 2.7 / epoch
        run.log_metrics({
            "Train Loss": round(train_loss, 4),
            "Val Loss":   round(val_loss,   4),
        }, step=epoch)

# Summary embed is automatically posted when the `with` block exits ↑

print("\n✅  Demo complete. Set DISCORD_WEBHOOK_URL and use dry_run=False to post to Discord.")
