# Benchmark: HIP versus Julia

We are comparing the performance differences between kernels writtein in C++ HIP versus Julia's AMDGPU.jl, both implementing an identical algoritm.

The aim is for the AMDGPU.jl version to reach speed parity with HIP.

## The algorithm

This particular kernel is a stripped-down version of what is essentially a direct Fourier Transform:

$$
y_{l,m} = \sum_{u,v} x_{u,v} \cdot \exp \left\{ 2 \pi i (u l + v m) \right\}$$

Where $u, v$ are coordinates in some 2D space, and $l, m$ are their Fourier counterparts. It is used as part of a radio astronomy algorithm known as Image Domain Gridding.

## Benchmarking

The code for both benchmarks is located at `mwe.cpp` and `mwe.jl` for the HIP and AMDGPU.jl versions, repsectively. The bencharks are computed on a Radeon W6800 Pro.

Neither version implments caching of global memory into the local memory of the workgroup. For simplicity of implementation, I am currently avoiding this.

Both versions are tested against a simple CPU version for correctness.

### HIP

```
> hipcc -o mwe mwe.cpp && AMD_SERIALIZE_KERNEL=3 AMD_LOG_LEVEL=2 ./mwe

Elapsed time per call: 25.3059 ms
```

### AMDGPU.jl

```
~/julia-1.9.0-alpha1/bin/julia --threads=auto --project=. mwe.jl

@roc OK
BenchmarkTools.Trial: 10 samples with 1 evaluation.
 Range (min … max):  216.361 ms … 226.832 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     216.491 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   217.557 ms ±   3.261 ms  ┊ GC (mean ± σ):  0.11% ± 0.33%

  █                                                              
  █▇▄▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▄ ▁
  216 ms           Histogram: frequency by time          227 ms <

 Memory estimate: 4.86 MiB, allocs estimate: 318079.
```

Additionally:

- the AMDGPU.jl version segfaults on clean-up
- both `cispi()` and `sincospi()` functions cause the kernel to silently fail

## Results

The HIP version is currently approximately 9x faster than the AMDGPU.jl version.
