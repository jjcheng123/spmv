device:
	nvcc -O3 -arch=sm_90a device.cu -o device

benchmark_fp16:
	nvcc -O3 -arch=sm_90a bench_atomic_fp16.cu -o benchmark_fp16

benchmark_fp32:
	nvcc -O3 -arch=sm_90a bench_atomic.cu -o benchmark

benchmark_fp64:
	nvcc -O3 -arch=sm_90a bench_atomic_fp64.cu -o benchmark_fp64