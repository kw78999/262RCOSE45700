#!/usr/bin/env bash
# Commands actually used for saved outputs; run from /root/ViT.
# Run directories must not exist. For another run, choose NEW names; do not delete caches.
python preflight/collect.py > preflight/logs/collect-console.log 2>&1
FURIOSA_TORCH_LOG=debug FURIOSA_TORCH_PROFILE=trace TORCH_LOGS=graph_breaks,recompiles timeout 90s python -u preflight/check_vit.py --substrate --profile --run-dir preflight/substrate-first > preflight/logs/substrate-first.log 2>&1
# Exit 0: API-only BF16 addition, not a model benchmark.
FURIOSA_TORCH_LOG=debug TORCH_LOGS=graph_breaks,recompiles timeout 90s python -u preflight/check_vit.py --substrate --run-dir preflight/substrate-reuse --cache-dir preflight/substrate-first/compiler-cache > preflight/logs/substrate-reuse.log 2>&1
# Exit 0: new process, same private disk cache, fresh process-local LRU.
FURIOSA_TORCH_LOG=debug TORCH_LOGS=graph_breaks,recompiles timeout 30s python -u preflight/check_vit.py --run-dir preflight/vit-attempt > preflight/logs/vit-attempt.log 2>&1
# Exit 1: ModuleNotFoundError: No module named 'torchvision'.
furiosa-smi status > preflight/logs/device-after.log
furiosa-smi ps >> preflight/logs/device-after.log
