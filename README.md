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
| ▶️ **Run Management** | Context-manager `start_run()` with auto summary embed on exit |
| 🖥️ **Dry-Run Mode** | `dry_run=True` prints to stdout — no real webhook calls |
| ❌ **Error Capture** | Exceptions inside a `start_run()` block are caught and posted |

---

## 📦 Installation

```bash
pip install discordflow
```

**Requirements:** Python ≥ 3.8, `requests`

---

## ⚡ Quickstart

### 1. Get a Discord Webhook URL

In your Discord server: **Server Settings → Integrations → Webhooks → New Webhook → Copy URL**

### 2. Drop it into your training loop

```python
from discordflow import DiscordFlow

WEBHOOK_URL = "YOUR_DISCORD_WEBHOOK_URL"
dflow = DiscordFlow(WEBHOOK_URL, experiment_name="MoE_Router_Training")

# Log hyperparameters
dflow.log_params({
    "experts": 8,
    "routing_strategy": "top-k",
    "learning_rate": 3e-4,
})

# Training loop
for epoch in range(1, 6):
    loss = 1.0 / epoch
    dflow.log_metrics({
        "Train Loss": round(loss, 4),
        "Load Balance": round(0.8 + 0.02 * epoch, 4),
    }, step=epoch)

# Upload an artifact (max 25 MB)
# dflow.log_artifact("router_weights.pt")
# dflow.log_artifact("loss_curve.png")
```

### 3. Context-manager pattern (recommended)

Use `start_run()` to get an automatic run-summary embed when the block exits — including elapsed time, all params, and final metrics. If your code crashes, the traceback is posted too.

```python
with dflow.start_run("lora_rank_16") as run:
    run.log_params({"lr": 2e-4, "lora_rank": 16, "epochs": 3})
    run.set_tag("framework", "HuggingFace")

    for epoch in range(1, 4):
        run.log_metrics({
            "Train Loss": round(2.5 / epoch, 4),
            "Val Loss":   round(2.7 / epoch, 4),
        }, step=epoch)

# ✅ Run Complete embed is auto-posted here
```

---

## 🎨 Custom Bot Identity

Give your DiscordFlow bot a custom name and profile picture so it blends into your server:

```python
dflow = DiscordFlow(
    webhook_url    = "YOUR_WEBHOOK_URL",
    experiment_name= "ResNet_Training",
    username       = "TrainBot 🏋️",          # Bot name shown in Discord
    avatar_url     = "https://i.imgur.com/YOUR_IMAGE.png",  # Bot profile picture URL
)
```

| Parameter | Type | Description |
|---|---|---|
| `username` | `str` | Display name shown on every Discord message (default: `"DiscordFlow 🤖"`) |
| `avatar_url` | `str` | Public URL to any image for the bot's avatar (JPEG, PNG, GIF) |

> **Tip:** Use any publicly accessible image URL — Discord's CDN, Imgur, GitHub raw links, etc.

---

## 🧪 Local Testing (Dry Run)

No Discord server? No problem. Use `dry_run=True` to print all messages to stdout instead of calling the webhook:

```python
dflow = DiscordFlow("ANY_URL", experiment_name="test", dry_run=True)
dflow.log_metrics({"loss": 0.42}, step=1)
```

Run the bundled demo:

```bash
python example.py
```

---

## 📚 API Reference

### `DiscordFlow(webhook_url, experiment_name, dry_run, username, avatar_url)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `webhook_url` | `str` | required | Discord webhook URL |
| `experiment_name` | `str` | `"Default Experiment"` | Shown in every embed |
| `dry_run` | `bool` | `False` | Print to stdout instead of calling webhook |
| `username` | `str` | `"DiscordFlow 🤖"` | Bot username shown in Discord |
| `avatar_url` | `str` | `None` | Custom bot avatar URL |

---

### Logging Methods

```python
# Single param
dflow.log_param("learning_rate", 3e-4)

# Multiple params in one embed
dflow.log_params({"lr": 3e-4, "batch_size": 128, "epochs": 10})

# Single metric
dflow.log_metric("loss", 0.42, step=5)

# Multiple metrics in one embed
dflow.log_metrics({"loss": 0.42, "acc": 0.91}, step=5)

# Arbitrary tags (purple embed)
dflow.set_tag("author", "e27")
dflow.set_tag("dataset", "openwebtext")

# Upload a file artifact (max 25 MB)
dflow.log_artifact("checkpoint.pt")
dflow.log_artifact("confusion_matrix.png")

# Upload a text snippet as a file
dflow.log_text("epoch,loss\n1,1.0\n2,0.5", filename="metrics.csv")
```

---

### Run Management

```python
# Start a named run (context manager — recommended)
with dflow.start_run("sweep_01") as run:
    run.log_params({...})
    run.log_metrics({...}, step=epoch)
    run.set_tag("status", "grid_search")
    run.log_artifact("model.pt")
# ← Auto-posts run summary embed on exit

# Or explicitly end a run
run = dflow.start_run("manual_run")
# ... do stuff ...
dflow.end_run(status="FINISHED")
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