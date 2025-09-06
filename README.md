# FlashAttention from Scratch — Triton & CUDA


A from-scratch implementation of **FlashAttention-style fused attention** kernels in **Triton** and **CUDA**, focusing on:

- **IO-awareness**: Fuse QKᵀ, scaling, masking, softmax, and PV into a single kernel to minimize HBM traffic.
- **Online softmax** for numerical stability at long sequence lengths.
- **Autotuning tile sizes** and optional **async double buffering** (`cp.async`) for high performance.

This repo includes:
- **Custom kernels** in Triton and CUDA
- **Testing suite** for correctness, gradients, and determinism
- **Benchmarking scripts** for throughput, latency, and peak memory
- **Colab-compatible setup** for easy reproducibility  

---

## 🚀 Features

- **Triton & CUDA Kernels**: Two independent implementations for comparison & learning  
- **Numerical Parity**: Validated vs. PyTorch SDPA across 200+ test cases (different N, dtypes, masks)  
- **Benchmark Suite**: Throughput (tokens/sec), latency (ms), peak GPU memory  
- **Mini-Transformer Integration**: End-to-end training/inference tests with synthetic data  
- **Colab-Friendly**: Runs on free T4 GPUs; optional A100 scaling for 8K–16K context lengths  

---

## 🛠️ Installation

```bash
git clone https://github.com/your-repo/flash-attention-scratch.git
cd flash-attention-scratch
pip install -r requirements.txt
