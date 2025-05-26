# TMA-Adaptive FP8 Grouped GEMM

## Abstract
Current FP8 grouped GEMM implementations require padding each group to a fixed alignment (e.g., 128), incurring memory and computational overhead. This paper proposes TMA-Adaptive FP8 Grouped GEMM that eliminates padding operation by addressing two critical challenges: (1) the fundamental limitation of a single static TMA descriptor to accommodate varying residual row counts across different groups and (2) strict memory alignment requirements (16-byte global memory/128-byte shared memory) for irregular group boundaries. Our solution employs two key techniques: (a) a predefined TMA descriptor pool that dynamically adapts to residual group dimensions through runtime selection, (b) memory alignment optimization combining overfetch prefetching with constrained block_K sizes (64-element multiples). Experiments demonstrate 1.7% to 20.4% speed up with up to 23.8% memory reduction compared to padding operation plus state-of-the-art FP8 grouped GEMM, while maintaining full numerical equivalence for valid data. This study eliminates the padding compromise, enabling simultaneous achievement of memory efficiency and computational throughput in low-precision computation pipelines for dynamically routed architectures.

## Quick start

### Requirements

- Hopper architecture GPUs, `sm_90a` must be supported
- Python 3.8 or above
- CUDA 12.3 or above
  - **But we highly recommend 12.8 or above for the best performance**
- PyTorch 2.1 or above
- CUTLASS 3.6 or above (could be cloned by Git submodule)

### Development

```bash
# Submodule must be cloned
git clone --recursive {git repository}

# Make symbolic links for third-party (CUTLASS and CuTe) include directories
python setup.py develop

# Test JIT compilation
python tests/test_jit.py

# Test TMA-Adaptive FP8 Grouped GEMM
python tests/test_TMA_adaptive_groupm.py
```

### Test Example

The test script test_TMA_adaptive_groupm.py compares our method with padding operation + DeepGEMM grouped GEMM, reporting: Execution time, Acceleration ratio, Bandwidth of padding peration, Memory saving, and Numerical correctness

Sample output:
```bash
Testing grouped contiguous GEMM:
    Our time 0.785 ms , padding + grouped gemm time 0.904 ms
    > Performance: 1870 GB/s (padding bandwidth)| Accelaration 15.149 % | Memory saving 9.220 %
    > The result of our kernel is identical to deep gemm after removal padded data? True
```

### Installation

```bash
python setup.py install
```

## Original deepgemm
To get detailed information of original deepgeem, please refer to https://github.com/deepseek-ai/DeepGEMM
