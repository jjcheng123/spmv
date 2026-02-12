#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <iomanip>

// 错误检查宏
#define CHECK_CUDA(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << ", " \
                  << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
}

// ----------------------------------------------------------------------
// Kernel 1: L2 Bypass Store (Cache Streaming) - FP64 Version
// ----------------------------------------------------------------------
__global__ void kernel_store_bypass_fp64(double* data, double val, size_t n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // st.global.cs.f64: 64-bit Store Global Cache-Streaming
        // 约束: "l" (long/address), "d" (double register)
        asm volatile ("st.global.cs.f64 [%0], %1;" :: "l"(&data[idx]), "d"(val));
    }
}

// ----------------------------------------------------------------------
// Kernel 2: Atomic Add - FP64 Version
// ----------------------------------------------------------------------
__global__ void kernel_atomic_add_fp64(double* data, double val, size_t n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // H200 (Hopper) 硬件原生支持 double atomicAdd
        atomicAdd(&data[idx], val);
    }
}

// ----------------------------------------------------------------------
// Kernel 3: Standard Store - FP64 Version
// ----------------------------------------------------------------------
__global__ void kernel_store_standard_fp64(double* data, double val, size_t n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = val;
    }
}

// ----------------------------------------------------------------------
// 辅助函数：运行 Benchmark Step
// ----------------------------------------------------------------------
double run_benchmark_step(void(*kernel)(double*, double, size_t), 
                          double* d_data, double val, size_t n, int block_size) {
    
    int grid_size = (n + block_size - 1) / block_size;

    float elapsed_time = 0.0f;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 预热
    for(int i=0; i<5; ++i) {
        kernel<<<grid_size, block_size>>>(d_data, val, n);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // 记录时间
    // 数据量越大，迭代次数可以适当减少
    int iterations = (n < (1<<25)) ? 50 : 20;

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        kernel<<<grid_size, block_size>>>(d_data, val, n);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    double avg_ms = elapsed_time / iterations;
    
    // FP64 是 8 字节
    double total_bytes = (double)n * sizeof(double);
    double bandwidth = (total_bytes * 1e-9) / (avg_ms * 1e-3);
    
    return bandwidth;
}

int main() {
    // 范围: 2^20 到 2^29 (注意: double 占空间大一倍，2^29 double = 4GB, 2^30 double = 8GB)
    // 我们可以测到 2^30
    const int start_power = 20; 
    const int end_power = 30;
    
    const size_t max_n = 1ULL << end_power;
    const size_t max_bytes = max_n * sizeof(double); // 8GB

    int block_size = 256;
    double val = 1.0;

    std::cout << "Target: FP64 (Double Precision)" << std::endl;
    std::cout << "Allocating Max Memory: " << (max_bytes / (1024.0 * 1024.0)) << " MB..." << std::endl;
    
    double* d_data;
    CHECK_CUDA(cudaMalloc(&d_data, max_bytes));
    CHECK_CUDA(cudaMemset(d_data, 0, max_bytes));

    std::cout << "\n==========================================================================================" << std::endl;
    std::cout << "|   Elements   |   Size (MB)  | L2 Bypass BW (GB/s) | AtomicAdd BW (GB/s) | Std Store (GB/s)|" << std::endl;
    std::cout << "|--------------|--------------|---------------------|---------------------|-----------------|" << std::endl;

    for (int p = start_power; p <= end_power; ++p) {
        size_t current_n = 1ULL << p;
        double size_mb = (current_n * sizeof(double)) / (1024.0 * 1024.0);

        // 1. FP64 Bypass
        double bw_bypass = run_benchmark_step(kernel_store_bypass_fp64, d_data, val, current_n, block_size);

        // 2. FP64 AtomicAdd
        // 注意：FP64 原子操作通常比 FP32 慢，因为它占用更多的 ALU 资源或需要更复杂的锁机制
        double bw_atomic = run_benchmark_step(kernel_atomic_add_fp64, d_data, val, current_n, block_size);

        // 3. FP64 Standard Store
        double bw_std = run_benchmark_step(kernel_store_standard_fp64, d_data, val, current_n, block_size);

        std::cout << "| 1<<" << std::left << std::setw(8) << p 
                  << " | " << std::setw(12) << std::fixed << std::setprecision(2) << size_mb 
                  << " | " << std::setw(19) << bw_bypass
                  << " | " << std::setw(19) << bw_atomic 
                  << " | " << std::setw(15) << bw_std << " |";

        if (size_mb < 50.0) std::cout << " (L2)";
        else std::cout << " (HBM)";
        
        std::cout << std::endl;
    }
    std::cout << "==========================================================================================" << std::endl;

    CHECK_CUDA(cudaFree(d_data));
    return 0;
}