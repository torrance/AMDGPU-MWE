// hipcc -o mwe mwe.cpp && AMD_SERIALIZE_KERNEL=3 AMD_LOG_LEVEL=2 ./mwe
#include <hip/hip_runtime.h>
#include <iostream>
#include <complex>
#include <chrono>
#include <cmath>
#include <random>

const int N = 1024 * 200;
const int Nsubgrid = 96 * 96;
const float pi = std::acosf(-1);

typedef struct Origin {
    float u0;
    float v0;
    float w0;
} Origin;

// CPU version used for testing
void cpukernel(
    std::array<std::complex<float>, Nsubgrid>& subgrid,
    const Origin origin,
    const std::array<float, N>& us,
    const std::array<float, N>& vs,
    const std::array<std::complex<float>, N>& data
) {

    for (int idx = 0; idx < Nsubgrid; ++idx) {
        int lpx = idx / 96;
        int mpx = idx - 96 * lpx;

        float l = lpx - 48;
        float m = mpx - 48;

        auto u = us.begin();
        auto v = vs.begin();
        auto datum = data.begin();

        while (datum != data.end()) {
            float phase = 2 * pi * (*u * l + *v * m);
            subgrid[idx] += *datum * std::complex<float>{std::cosf(phase), std::sinf(phase)};

            ++u; ++v; ++datum;
        }
    }
}

// GPU kernel
__global__
void kernel(std::complex<float>* subgrid, Origin origin, float* us, float* vs, std::complex<float>* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= Nsubgrid) {
        return;
    }

    int lpx = idx / 96;
    int mpx = idx - 96 * lpx;

    float l = lpx - 48;
    float m = mpx - 48;

    std::complex<float> cell{0};
    for (int i = 0; i < N; ++i) {
        float u = us[i];
        float v = vs[i];
        std::complex<float> datum = data[i];

        float phase = 2 * (u * l + v * m);
        float real, imag;
        sincospif(phase, &imag, &real);
        cell += datum * std::complex<float>{real, imag};
    }

    subgrid[idx] = cell;
}

int cld(int x, int y) {
    return (x + y - 1) / y;
}

int main(void) {
    Origin origin{1.5, 3.2, -0.4};

    // Create a bunch of random arrays
    std::default_random_engine gen;
    std::uniform_real_distribution<float> dist(-1, 1);

    std::array<float, N> us;
    for (auto iter=us.begin(); iter != us.end(); ++iter) {
        *iter = dist(gen);
    }

    std::array<float, N> vs;
    for (auto iter=vs.begin(); iter != vs.end(); ++iter) {
        *iter = dist(gen);
    }

    std::array<std::complex<float>, N> data;
    for (auto iter=data.begin(); iter != data.end(); ++iter) {
        *iter = std::complex<float>{dist(gen), dist(gen)};
    }

    std::array<std::complex<float>, Nsubgrid> expected;

    // Calculate expected version using simple CPU loop
    cpukernel(expected, origin, us, vs, data);

    // Transfer random arrays to GPU
    float *us_d;
    hipMalloc(&us_d, sizeof(us));
    hipMemcpy(us_d, us.data(), sizeof(us), hipMemcpyHostToDevice);

    float *vs_d;
    hipMalloc(&vs_d, sizeof(vs));
    hipMemcpy(vs_d, vs.data(), sizeof(vs), hipMemcpyHostToDevice);

    std::complex<float> *data_d;
    hipMalloc(&data_d, sizeof(data));
    hipMemcpy(data_d, data.data(), sizeof(data), hipMemcpyHostToDevice);

    std::complex<float> *subgrid_d;
    hipMalloc(&subgrid_d, sizeof(expected));
    hipMemcpy(subgrid_d, expected.data(), sizeof(expected), hipMemcpyHostToDevice);

    hipStreamSynchronize(0);

    int nthreads = 512;
    int nblocks = cld(Nsubgrid, nthreads);

    // Benchmark GPU kernel
    auto begin = std::chrono::high_resolution_clock::now();
    for (int n = 0; n < 10; ++n) {
        hipLaunchKernelGGL(kernel, dim3(nblocks), dim3(nthreads), 0, 0, subgrid_d, origin, us_d, vs_d, data_d);
        hipStreamSynchronize(0);
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
    std::cout << "Elapsed time per call: " << elapsed.count() / 1000. / 10. << " ms" << std::endl;

    std::array<std::complex<float>, Nsubgrid> subgrid;
    hipMemcpy(subgrid.data(), subgrid_d, sizeof(subgrid), hipMemcpyDeviceToHost);

    // Compare CPU and GPU versions for first few results
    for (int i = 0; i < 10; ++i) {
        std::cout << "CPU: " << expected[i] << " GPU: " << subgrid[i] << std::endl;
    }

    hipFree(us_d);
    hipFree(vs_d);
    hipFree(data_d);
    hipFree(subgrid_d);

    return 0;
}