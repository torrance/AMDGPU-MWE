# Benchmark: HIP versus Julia

We are comparing the performance differences between kernels writtein in C++ HIP versus Julia's AMDGPU.jl, both implementing an identical algoritm.

The aim is for the AMDGPU.jl version to reach speed parity with HIP.

## The algorithm

This particular kernel is a stripped-down version of what is essentially a direct Fourier Transform:

$$
y_{l,m} = \sum_{u,v} x_{u,v} \cdot \exp \left\lbrace 2 \pi i (u l + v m) \right\rbrace
$$

Where $u, v$ are coordinates in some 2D space, and $l, m$ are their Fourier counterparts. It is used as part of a radio astronomy algorithm known as Image Domain Gridding.

**Update:** In an effort to strip down the code to its bare essentials whilst still retaining the speed difference, the algorithm is now:

$$
y = \sum_{u,v} x_{u,v} \cdot \left\lbrace 2 \pi (5u + 4v) \right\rbrace
$$

If we can reach parity with this simpler version, we can revert back to the fuller algoirthm.

## Benchmarking

The code for both benchmarks is located at `mwe.cpp` and `mwe.jl` for the HIP and AMDGPU.jl versions, repsectively. The bencharks are computed on a Radeon W6800 Pro.

Neither version implments caching of global memory into the local memory of the workgroup. For simplicity of implementation, I am currently avoiding this.

Both versions are tested against a simple CPU version for correctness.

### HIP

```
> hipcc -o mwe mwe.cpp && AMD_SERIALIZE_KERNEL=3 AMD_LOG_LEVEL=2 ./mwe

Elapsed time per call: 4.053 ms
```

### AMDGPU.jl

```
~/julia-1.9.0-alpha1/bin/julia --threads=auto --project=. mwe.jl

@roc OK
BenchmarkTools.Trial: 10 samples with 1 evaluation.
 Range (min … max):  33.093 ms … 33.409 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     33.204 ms              ┊ GC (median):    0.00%
 Time  (mean ± σ):   33.209 ms ± 91.935 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

  █    ██     █      █ █      ██  █                         █  
  █▁▁▁▁██▁▁▁▁▁█▁▁▁▁▁▁█▁█▁▁▁▁▁▁██▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
  33.1 ms         Histogram: frequency by time        33.4 ms <

 Memory estimate: 767.36 KiB, allocs estimate: 48831.
```

Additionally:

- the AMDGPU.jl version segfaults on clean-up
- both `cispi()` and `sincospi()` functions cause the kernel to silently fail

## Results

The HIP version is currently approximately 8x faster than the AMDGPU.jl version.

## Changelog

* Fixed a a type issue where the `us` and `vs` vectors were becoming `Float64`. Slowdown reduced from ~9x to ~2x.

* Removed the exponential function and fixed `lpx` and `mpx`; the slowdown has increased to ~8x.
