#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    int device = 0;
    if (argc > 1) device = atoi(argv[1]);

    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("GPU Device %d: %s\n", device, prop.name);
    printf("  L2 Cache Size: %d bytes (%.1f MB)\n", 
           prop.l2CacheSize, prop.l2CacheSize / 1024.0 / 1024.0);
    printf("  Global Memory: %.1f GB\n", prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);
    printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);

    return 0;
}