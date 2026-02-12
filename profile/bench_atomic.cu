#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <iomanip>
#include <string>

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
// Kernel 1: L2 Bypass Store (Cache Streaming)
// ----------------------------------------------------------------------
__global__ void kernel_store_bypass(float* data, float val, size_t n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // st.global.cs.f32: 提示数据只流过 L2，不驻留
        asm volatile ("st.global.cs.f32 [%0], %1;" :: "l"(&data[idx]), "f"(val));
    }
}

// ----------------------------------------------------------------------
// Kernel 2: Atomic Add
// ----------------------------------------------------------------------
__global__ void kernel_atomic_add(float* data, float val, size_t n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(&data[idx], val);
    }
}

// ----------------------------------------------------------------------
// Kernel 3: Standard Store (用于对比，普通写入)
// ----------------------------------------------------------------------
__global__ void kernel_store_standard(float* data, float val, size_t n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = val;
    }
}

// ----------------------------------------------------------------------
// 辅助函数：运行 Benchmark
// 返回带宽 (GB/s)
// ----------------------------------------------------------------------
double run_benchmark_step(void(*kernel)(float*, float, size_t), 
                          float* d_data, float val, size_t n, int block_size) {
    
    // 计算 Grid Size
    // 注意：使用 size_t 防止溢出，虽然 2^30 在 int 范围内，但习惯要好
    int grid_size = (n + block_size - 1) / block_size;

    float elapsed_time = 0.0f;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 预热 (Warmup) - 关键！让 GPU 频率跑起来
    for(int i=0; i<5; ++i) {
        kernel<<<grid_size, block_size>>>(d_data, val, n);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // 记录时间
    // 对于小数据量，迭代次数要多一点，减少计时误差
    // 对于大数据量，迭代次数少一点，节省时间
    int iterations = (n < (1<<25)) ? 100 : 20;

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
    double total_bytes = (double)n * sizeof(float);
    // GB/s = (Bytes / 1e9) / (ms / 1e3)
    double bandwidth = (total_bytes * 1e-9) / (avg_ms * 1e-3);
    
    return bandwidth;
}

int main() {
    // 设置最大规模: 2^30 floats = 4GB
    // 设置最小规模: 2^20 floats = 4MB
    const int start_power = 20; 
    const int end_power = 30;
    const size_t max_n = 1ULL << end_power;
    const size_t max_bytes = max_n * sizeof(float);

    int block_size = 256;
    float val = 1.0f;

    std::cout << "Allocating Max Memory: " << (max_bytes / (1024.0 * 1024.0)) << " MB..." << std::endl;
    
    float* d_data;
    CHECK_CUDA(cudaMalloc(&d_data, max_bytes));
    CHECK_CUDA(cudaMemset(d_data, 0, max_bytes));

    std::cout << "\n==========================================================================================" << std::endl;
    std::cout << "|   Elements   |   Size (MB)  | L2 Bypass BW (GB/s) | AtomicAdd BW (GB/s) | Std Store (GB/s)|" << std::endl;
    std::cout << "|--------------|--------------|---------------------|---------------------|-----------------|" << std::endl;

    for (int p = start_power; p <= end_power; ++p) {
        size_t current_n = 1ULL << p;
        double size_mb = (current_n * sizeof(float)) / (1024.0 * 1024.0);

        // 1. 测试 L2 Bypass
        double bw_bypass = run_benchmark_step(kernel_store_bypass, d_data, val, current_n, block_size);

        // 2. 测试 Atomic Add
        // 为了防止 atomic overflow 导致某些不可预知的慢速（虽然 float 通常没事），这里在每次 atomic 前重置一下数据
        // (可选，这里为了纯粹测写性能，不重置其实也没事，atomicAdd 性能主要受限于竞争和队列)
        double bw_atomic = run_benchmark_step(kernel_atomic_add, d_data, val, current_n, block_size);

        // 3. (可选) 测试普通 Store 用于对比
        double bw_std = run_benchmark_step(kernel_store_standard, d_data, val, current_n, block_size);

        // 格式化输出
        std::cout << "| 1<<" << std::left << std::setw(8) << p 
                  << " | " << std::setw(12) << std::fixed << std::setprecision(2) << size_mb 
                  << " | " << std::setw(19) << bw_bypass
                  << " | " << std::setw(19) << bw_atomic 
                  << " | " << std::setw(15) << bw_std << " |";

        // 标记 L2 Cache 边界 (H200 L2 ~= 50MB)
        if (size_mb < 50.0) std::cout << " (L2)";
        else std::cout << " (HBM)";
        
        std::cout << std::endl;
    }
    std::cout << "==========================================================================================" << std::endl;

    CHECK_CUDA(cudaFree(d_data));
    return 0;
}