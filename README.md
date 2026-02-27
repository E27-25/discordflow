# DiscordFlow 🚀

> **The MLflow you already have open on your phone.**
> Log ML training metrics, parameters, and artifacts directly to a Discord channel via webhooks — no server required.

![PyPI](https://img.shields.io/pypi/v/discordflow)
![Python Versions](https://img.shields.io/pypi/pyversions/discordflow)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

---

## ✨ Features

| Feature | Description |
|---|---|
| 📈 **Metric Logging** | Post metrics per step/epoch with `log_metrics()` |
| ⚙️ **Param Logging** | Log hyperparameters with `log_param()` / `log_params()` |
| 🏷️ **Tags** | Attach arbitrary key-value tags to runs |
| 📁 **Artifact Upload** | Upload files (models, plots, CSVs) up to 25 MB |
| 📄 **Text Artifacts** | Upload text snippets as `.txt` file attachments |
| 📈 **Figure Upload** | Send `matplotlib` figures directly as PNG attachments |
| 💬 **Normal Channel** | `start_run()` — logs embeds directly into the channel |
| 📋 **Forum Channel** | `start_forum_run()` — each run gets its own dedicated forum thread |
| 💾 **State Persistence** | `save()` — persist thread IDs to JSON for Colab restart recovery |
| 🖥️ **System Metrics** | Per-call hardware stats: CPU, RAM, GPU, Disk, Network |
| ⚡ **Async Logging** | Background thread executor — never blocks your training loop |
| ❌ **Error Capture** | Exceptions inside a run block are caught and posted to Discord |
| 🖥️ **Dry-Run Mode** | `dry_run=True` prints to stdout — no real webhook calls |

---

## 📦 Installation

```bash
pip install discordflow
```

With system hardware metrics (CPU, RAM, Disk, Network):
```bash
pip install "discordflow[system]"
```

With NVIDIA GPU metrics:
```bash
pip install "discordflow[system,gpu]"
```

**Requirements:** Python ≥ 3.8, `requests`

---

## ⚡ Quickstart

### 1. Get a Discord Webhook URL

In your Discord server: **Server Settings → Integrations → Webhooks → New Webhook → Copy URL**

### 2. Normal Channel Mode — `start_run()`

Use this when your webhook points to a **regular text/announcement channel**.
Embeds are posted directly into the channel.

```python
from discordflow import DiscordFlow

dflow = DiscordFlow(WEBHOOK_URL, experiment_name="ResNet_Training")

with dflow.start_run("baseline") as run:
    run.log_params({"lr": 3e-4, "batch_size": 128, "epochs": 10})
    run.set_tag("dataset", "ImageNet")

    for epoch in range(1, 11):
        run.log_metrics(
            {"Train Loss": 1.0 / epoch, "Val Acc": 0.7 + 0.02 * epoch},
            step=epoch,
            system_metrics=["cpu", "ram"],  # ← hardware stats attached to each update
        )

    run.log_artifact("best_model.pt")
    run.log_figure(fig, title="Loss Curve")

dflow.finish()  # flush async queue
```

### 3. Forum Channel Mode — `start_forum_run()`

Use this when your webhook points to a **Forum channel**.
Each run automatically gets its own Discord thread. Perfect for tracking many experiments.

```python
dflow = DiscordFlow(FORUM_WEBHOOK_URL, experiment_name="LLM_FineTune")

with dflow.start_forum_run("lora_r16", description="LoRA sweep rank=16") as run:
    run.log_params({"lora_rank": 16, "lr": 2e-4, "epochs": 3})
    run.set_tag("framework", "HuggingFace")

    for epoch in range(1, 4):
        run.log_metrics(
            {"Train Loss": 2.5 / epoch, "Val Loss": 2.7 / epoch},
            step=epoch,
            system_metrics=["cpu", "ram", "gpu"],  # ← GPU stats for NVIDIA Colab
        )
        run.log_figure(fig, title=f"Epoch {epoch} Loss")

# ✅ Summary posted in the thread automatically

dflow.save()   # ← IMPORTANT: persist thread IDs for restart recovery
dflow.finish()
```

---

## 💾 Colab Restart Recovery

If your Colab runtime crashes or disconnects, restore your session:

```python
# In a fresh Colab runtime — restore your thread IDs
dflow = DiscordFlow(FORUM_WEBHOOK_URL, "LLM_FineTune")

# Option A — automatic restore (if state file was previously saved)
# State is loaded automatically from .discordflow_state.json on startup

# Option B — manual override
dflow.resume_run("lora_r16", thread_id="1234567890123456789")
dflow.save()

# Option C — ZIP backup/restore (downloads to your PC, re-uploads to Colab)
from discordflow.colab_utils import export_session, import_session

export_session(dflow)   # downloads discordflow_backup.zip to your machine
import_session(dflow)   # on fresh runtime: upload the zip to restore threads
```

---

## 🎨 Custom Bot Identity

```python
dflow = DiscordFlow(
    webhook_url     = WEBHOOK_URL,
    experiment_name = "ResNet_Training",
    username        = "TrainBot 🏋️",                      # custom bot name
    avatar_url      = "https://i.imgur.com/AfFp7pu.png",  # any public image URL
)
```

---

## 🖥️ System Metrics Reference

Pass any combination to `system_metrics=` on `log_metrics()`:

| Key | What's logged | Requires |
|---|---|---|
| `"cpu"` | Usage % + clock speed | `discordflow[system]` |
| `"ram"` | Usage % + GB used/total | `discordflow[system]` |
| `"gpu"` | Util % + VRAM used/total per GPU | `discordflow[system,gpu]` |
| `"disk"` | Usage % + GB used/total | `discordflow[system]` |
| `"network"` | Total MB sent/received | `discordflow[system]` |

```python
# Mix and match freely per call
run.log_metrics({"loss": 0.4}, step=1, system_metrics=["cpu", "ram"])
run.log_metrics({"val_loss": 0.5}, step=1, system_metrics=["cpu", "ram", "gpu", "disk", "network"])
```

---

## 📚 API Reference

### `DiscordFlow(webhook_url, experiment_name, ...)`

| Parameter | Default | Description |
|---|---|---|
| `webhook_url` | required | Discord webhook URL |
| `experiment_name` | `"Default Experiment"` | Shown in every embed |
| `state_file` | `".discordflow_state.json"` | JSON file for thread ID persistence |
| `async_logging` | `True` | Background thread for non-blocking sends |
| `dry_run` | `False` | Print to stdout instead of calling webhook |
| `username` | `"DiscordFlow 🤖"` | Bot display name in Discord |
| `avatar_url` | `None` | Bot profile picture URL |

### Channel Methods

| Method | Channel type | Description |
|---|---|---|
| `start_run(run_name)` | Normal | Start a run, post embeds to channel |
| `start_forum_run(run_name, description)` | Forum | Create/resume a forum thread for this run |
| `resume_run(run_name, thread_id)` | Forum | Manually re-link a run to an existing thread |
| `save(filepath)` | Both | Save `{run_name: thread_id}` state to JSON |
| `finish()` | Both | Flush async queue and shut down executor |

### Logging Methods (available on both `ActiveRun` and `ForumActiveRun`)

```python
run.log_param("lr", 3e-4)
run.log_params({"lr": 3e-4, "batch": 128})
run.log_metric("loss", 0.42, step=5)
run.log_metrics({"loss": 0.42, "acc": 0.91}, step=5, system_metrics=["cpu", "ram"])
run.set_tag("author", "e27")
run.log_artifact("checkpoint.pt")
run.log_text("epoch,loss\n1,1.0", filename="metrics.csv")
run.log_figure(fig, title="Loss Curve")
```

---

## 🧪 Local Testing (Dry Run)

```python
dflow = DiscordFlow("ANY_URL", dry_run=True)
with dflow.start_run("test") as run:
    run.log_metrics({"loss": 0.42}, step=1)
```

---

## 🤝 Contributing

1. Fork the repo
2. Create your feature branch: `git checkout -b feat/my-feature`
3. Commit your changes: `git commit -m "feat: add my feature"`
4. Push: `git push origin feat/my-feature`
5. Open a Pull Request

---

## 📄 License

MIT © DiscordFlow Contributors