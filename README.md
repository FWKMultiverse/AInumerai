# AInumerai Ultra — Advanced Numerai Prediction System

A production-grade machine learning system built for Numerai tournament predictions. Developed and iterated through 21 major versions with a focus on model quality, resource efficiency, and self-improving architecture.

---

## Overview

This system combines multiple ML model types into a unified ensemble pipeline that continuously learns from new data. The architecture is designed to run on consumer hardware while maintaining prediction quality competitive with professional quant systems.

The pipeline handles the full workflow — from raw Numerai data ingestion to weekly submission — with automated retraining, backtesting, and performance tracking.

---

## Architecture

The system is structured as a multi-stage training pipeline:

**Step 1 — Feature Engineering and Selection**
Automated discovery of causal features using statistical analysis. Includes VAE-based dimensionality reduction and feature importance tracking across eras.

**Step 2 — Base Model Training (CPU)**
LightGBM and CatBoost ensemble with 60-round hyperparameter tuning via Optuna. Models are trained with adaptive regularization to prevent overfitting on Numerai's obfuscated features.

**Step 3 — Deep Learning Models (GPU)**
Transformer-based sequence models and VAE encoders trained on GPU. Uses mixed precision training with gradient clipping and cosine annealing learning rate schedule.

**Step 4 — Graph Neural Network (GPU)**
GATv2-based GNN that models feature relationships as a graph using correlation-derived adjacency matrices. Trained with dynamic batch sizing based on available VRAM to prevent OOM errors.

**Step 5 — Ensemble and Post-processing**
Adaptive ensemble weighting based on per-era validation performance. MMC-optimized blending with two independent correlation methods.

---

## Key Systems

**AutoML** — Automated hyperparameter search across LightGBM, CatBoost, and neural architectures. Tracks experiment history and selects best configurations based on Numerai-specific metrics.

**Self-Awareness and Error Recovery** — Runtime diagnosis system that detects training instability, NaN gradients, and resource exhaustion. Automatically adjusts batch sizes, falls back to CPU, and recovers from partial failures without restarting.

**Backtesting Framework** — Era-by-era cross-validation that mirrors Numerai's evaluation structure. Submissions are blocked automatically if backtest performance falls below threshold.

**Resource Management** — Designed for 16GB RAM and RTX 3060 12GB VRAM. Includes SSD disk caching for large datasets, continuous RAM monitoring, and intelligent GPU memory allocation.

**Experiment Tracking** — All training runs logged with hyperparameters, per-era metrics, and model states. Performance dashboard for trend analysis across versions.

---

## Model Stack

| Model | Type | Training |
|---|---|---|
| LightGBM | Gradient Boosting | CPU, 60-round Optuna tuning |
| CatBoost | Gradient Boosting | CPU, 60-round Optuna tuning |
| GATv2 GNN | Graph Neural Network | GPU, dynamic batch sizing |
| Transformer | Sequence Model | GPU, mixed precision |
| VAE | Variational Autoencoder | GPU, feature compression |

---

## Technical Stack

- Python 3.10+
- PyTorch with CUDA support
- PyTorch Geometric (GATv2Conv, GINConv)
- LightGBM, CatBoost, XGBoost
- Optuna for hyperparameter optimization
- Hugging Face Transformers
- PEFT / LoRA for efficient fine-tuning
- Accelerate for multi-device training

---

## Hardware Target

Developed and optimized for:
- CPU: AMD Ryzen 5 5500 (6 cores)
- GPU: NVIDIA RTX 3060 12GB VRAM
- RAM: 16GB with SSD overflow caching

The system adapts automatically to available resources and degrades gracefully on lower-spec hardware.

---

## Version History

21 major versions spanning architecture redesigns, training stability improvements, and resource optimization. Key milestones include introduction of GNN (v15), VAE feature compression (v17), adaptive regularization (v18), GPU-CPU hybrid training (v20), and 60-round hyperparameter tuning with clean architecture (v21).

---

## Status

Active development. Currently in final testing before live Numerai tournament submission.
