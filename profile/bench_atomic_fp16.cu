#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h> // 必须包含，用于 __half
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
// Kernel 1: L2 Bypass Store (Cache Streaming) - FP16
// ----------------------------------------------------------------------
__global__ void kernel_store_bypass(__half* data, __half val, size_t n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 修复：显式将 __half 转换为 unsigned short (16位整数)
        // 这样编译器就不会因为不知道用哪个类型转换操作符而报错了
        unsigned short val_raw = __half_as_ushort(val);
        
        // st.global.cs.b16 写入 16 位数据
        // "h" 约束对应 PTX 中的 .u16 寄存器
        asm volatile ("st.global.cs.b16 [%0], %1;" :: "l"(&data[idx]), "h"(val_raw));
    }
}
// ----------------------------------------------------------------------
// Kernel 2: Atomic Add - FP16
// 注意：FP16 atomicAdd 需要 Compute Capability >= 7.0
// ----------------------------------------------------------------------
__global__ void kernel_atomic_add(__half* data, __half val, size_t n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 现代 GPU 支持原生的 FP16 atomicAdd
#if __CUDA_ARCH__ >= 700
        atomicAdd(&data[idx], val);
#else
        // 旧架构可能不支持，这里仅仅为了编译通过，实际运行在旧卡上可能会报错或很慢
        // 实际 fallback 往往需要用 CAS 循环实现，此处省略
        data[idx] = val; 
#endif
    }
}

// ----------------------------------------------------------------------
// Kernel 3: Standard Store - FP16 (普通写入)
// ----------------------------------------------------------------------
__global__ void kernel_store_standard(__half* data, __half val, size_t n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = val;
    }
}

// ----------------------------------------------------------------------
// 辅助函数：运行 Benchmark
// ----------------------------------------------------------------------
double run_benchmark_step(void(*kernel)(__half*, __half, size_t), 
                          __half* d_data, __half val, size_t n, int block_size) {
    
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
    // FP16 计算量小，带宽压力大，同样根据 N 调整迭代次数
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
    // 注意：这里使用 sizeof(__half) = 2 bytes
    double total_bytes = (double)n * sizeof(__half);
    // GB/s
    double bandwidth = (total_bytes * 1e-9) / (avg_ms * 1e-3);
    
    return bandwidth;
}

int main() {
    // 编译检查
    int device_id = 0;
    cudaDeviceProp prop;
    cudaGetDevice(&device_id);
    cudaGetDeviceProperties(&prop, device_id);
    if (prop.major < 7) {
        std::cout << "Warning: This device (SM " << prop.major << "." << prop.minor 
                  << ") may not support native FP16 atomicAdd (Requires SM 7.0+)." << std::endl;
    }

    // 设置规模：因为 FP16 只有 2 字节，我们可以把元素数量翻倍，或者保持数量不变
    // 这里保持数量范围不变，但实际内存占用会减半
    const int start_power = 20; 
    const int end_power = 30; // 2^30 halves = 2GB
    const size_t max_n = 1ULL << end_power;
    const size_t max_bytes = max_n * sizeof(__half);

    int block_size = 256;
    // Host 端初始化 half，使用 __float2half
    __half val = __float2half(1.0f);

    std::cout << "Benchmarking FP16 (2 bytes/element)..." << std::endl;
    std::cout << "Allocating Max Memory: " << (max_bytes / (1024.0 * 1024.0)) << " MB..." << std::endl;
    
    __half* d_data;
    CHECK_CUDA(cudaMalloc(&d_data, max_bytes));
    CHECK_CUDA(cudaMemset(d_data, 0, max_bytes));

    std::cout << "\n==========================================================================================" << std::endl;
    std::cout << "|   Elements   |   Size (MB)  | L2 Bypass BW (GB/s) | AtomicAdd BW (GB/s) | Std Store (GB/s)|" << std::endl;
    std::cout << "|--------------|--------------|---------------------|---------------------|-----------------|" << std::endl;

    for (int p = start_power; p <= end_power; ++p) {
        size_t current_n = 1ULL << p;
        double size_mb = (current_n * sizeof(__half)) / (1024.0 * 1024.0);

        // 1. 测试 L2 Bypass
        double bw_bypass = run_benchmark_step(kernel_store_bypass, d_data, val, current_n, block_size);

        // 2. 测试 Atomic Add
        double bw_atomic = run_benchmark_step(kernel_atomic_add, d_data, val, current_n, block_size);

        // 3. 测试普通 Store
        double bw_std = run_benchmark_step(kernel_store_standard, d_data, val, current_n, block_size);

        std::cout << "| 1<<" << std::left << std::setw(8) << p 
                  << " | " << std::setw(12) << std::fixed << std::setprecision(2) << size_mb 
                  << " | " << std::setw(19) << bw_bypass
                  << " | " << std::setw(19) << bw_atomic 
                  << " | " << std::setw(15) << bw_std << " |";

        // 这里的 L2 大小标记是估算的，H100 L2 为 50MB
        if (size_mb < 50.0) std::cout << " (L2)";
        else std::cout << " (HBM)";
        
        std::cout << std::endl;
    }
    std::cout << "==========================================================================================" << std::endl;

    CHECK_CUDA(cudaFree(d_data));
    return 0;
}