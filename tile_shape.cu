#include <iostream>
#include <vector>
#include <cuda_runtime.h>
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

// =================================================================================
// 核心 Kernel：模拟 AtSpMV 中的 Tile 计算
// TILE_M * TILE_K = 32 (对应一个 Warp)
// =================================================================================
template <int TILE_M, int TILE_K>
__global__ void benchmark_tile_kernel(
    const float* __restrict__ values, // 模拟存储的非零元素值 (AtBSR val)
    const float* __restrict__ vector_x, // 稠密向量 X
    float* __restrict__ vector_y,       // 输出向量 Y
    int num_tiles                       // 总 Tile 数量
) {
    // 1. 计算全局 Warp ID 和 Lane ID
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_tid / 32;
    int lane_id = threadIdx.x % 32;

    if (warp_id >= num_tiles) return;

    // 2. 模拟计算 (Fused Multiply-Add)
    // 在真实 SpMV 中这里需要通过 col_idx 索引 X，这里为了单纯测计算和规约开销，简化为直接读取
    float val = values[global_tid];
    float x_val = vector_x[global_tid % 1024]; // 模拟读取 X
    float res = val * x_val;

    // 3. Warp 内 Shuffle 归约 (Row Reduction)
    // TILE_K 决定了 Shuffle 的深度。K=32 需要 5 次，K=1 需要 0 次。
    // 论文引用: ShuffleTimes = warpCount * log2(tile_k) 
    // #pragma unroll
    // for (int offset = TILE_K / 2; offset > 0; offset /= 2) {
    //     res += __shfl_down_sync(0xffffffff, res, offset);
    // }

    // 4. 原子写入 (Atomic Write)
    // 只有每行的第一个线程 (lane_id % TILE_K == 0) 负责写入
    // TILE_M 决定了写入次数。M=32 写 32 次，M=1 写 1 次。
    // 论文引用: AtomTimes = warpCount * tile_m 
    if (lane_id % TILE_K == 0) {
        // 计算当前线程负责写入哪一行
        // 为了模拟真实负载，我们将写入分散到 vector_y 的不同位置
        int row_offset = lane_id / TILE_K; 
        int target_row = warp_id * TILE_M + row_offset;
        
        // 关键性能点：Atomic Add
        atomicAdd(&vector_y[target_row], res);
    }
}

// =================================================================================
// 启动器与性能计时
// =================================================================================
template <int TILE_M, int TILE_K>
void run_test(float* d_val, float* d_x, float* d_y, int num_tiles, const char* label) {
    int block_size = 256; // 8 Warps per block
    int grid_size = (num_tiles * 32 + block_size - 1) / block_size;

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Warmup
    for(int i=0; i<10; ++i)
        benchmark_tile_kernel<TILE_M, TILE_K><<<grid_size, block_size>>>(d_val, d_x, d_y, num_tiles);

    CHECK_CUDA(cudaEventRecord(start));
    
    int iterations = 100;
    for(int i=0; i<iterations; ++i) {
        benchmark_tile_kernel<TILE_M, TILE_K><<<grid_size, block_size>>>(d_val, d_x, d_y, num_tiles);
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    
    float avg_ms = ms / iterations;
    // 每个 Warp 处理 32 个元素，做 32 次乘加
    double gflops = (double)num_tiles * 32 * 2 * 1e-9 / (avg_ms * 1e-3);

    std::cout << "| " << std::setw(10) << label 
              << " | " << std::setw(10) << std::fixed << std::setprecision(3) << avg_ms 
              << " | " << std::setw(10) << gflops 
              << " | " << "Shfl:" << int(log2(TILE_K)) << ", Atom:" << TILE_M
              << " |" << std::endl;

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

int main() {
    // 设置规模：100万个 Tile (即 3200万个非零元素)
    int num_tiles = 1000000;
    int total_nnz = num_tiles * 32;
    int vec_size = total_nnz; // 简化处理，假设 y 足够大

    float *d_val, *d_x, *d_y;
    CHECK_CUDA(cudaMalloc(&d_val, total_nnz * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_x, total_nnz * sizeof(float))); // 简化 X 读取
    CHECK_CUDA(cudaMalloc(&d_y, vec_size * sizeof(float)));

    // 初始化数据 (略过具体数值初始化，只关注性能)
    CHECK_CUDA(cudaMemset(d_val, 0, total_nnz * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_x, 0, total_nnz * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_y, 0, vec_size * sizeof(float)));

    std::cout << "Benchmarking different Tile Shapes (M x K = 32 elements)" << std::endl;
    std::cout << "Total Tiles: " << num_tiles << ", Total NNZ: " << total_nnz << std::endl;
    std::cout << "-------------------------------------------------------------" << std::endl;
    std::cout << "| Shape (MxK) |  Time(ms)  |   GFLOPs   | Overhead Factor    |" << std::endl;
    std::cout << "|-------------|------------|------------|--------------------|" << std::endl;

    // 依次测试 AtSpMV 中提到的 6 种候选形状 [cite: 783, 838]
    run_test<1, 32>(d_val, d_x, d_y, num_tiles, "1 x 32");
    run_test<2, 16>(d_val, d_x, d_y, num_tiles, "2 x 16");
    run_test<4, 8> (d_val, d_x, d_y, num_tiles, "4 x 8");
    run_test<8, 4> (d_val, d_x, d_y, num_tiles, "8 x 4");
    run_test<16, 2>(d_val, d_x, d_y, num_tiles, "16 x 2");
    run_test<32, 1>(d_val, d_x, d_y, num_tiles, "32 x 1");

    std::cout << "-------------------------------------------------------------" << std::endl;

    CHECK_CUDA(cudaFree(d_val));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));

    return 0;
}