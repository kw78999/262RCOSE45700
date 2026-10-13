# RNGD LLM Execution Mode Benchmark

A project exploring the **break-even point between PyTorch Eager and Compile** for LLM inference on FuriosaAI RNGD.

## Overview

Compilation can speed up repeated inference, but its initial overhead may outweigh the benefits for smaller workloads. This project benchmarks both execution modes and estimates when compilation becomes worthwhile.

## Experimental Setup

- **Hardware:** FuriosaAI RNGD NPU (single device)
- **Model:** Qwen3-1.7B (pending compatibility verification)
- **Framework:** PyTorch and FuriosaAI SDK
- **Workload:** Autoregressive text generation
- **Variables:** Batch size, input length, and output length

## Benchmark Metrics

- Prefill latency and per-token decode latency
- End-to-end latency and throughput
- Compilation time and peak memory usage
- Recompilation behavior across input shapes
- Output consistency between execution modes


