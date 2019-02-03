#pragma once

#include <cstdio>
#include <cuda_runtime.h>

#define ENABLE_GPU_CHECK

#ifdef ENABLE_GPU_CHECK
#define gpu_check(ans) { gpu_assert((ans), __FILE__, __LINE__); }
#else
#define gpu_check(ans) (ans)
#endif

inline void gpu_assert(cudaError_t code, const char *file, int line, bool abort = true) {
	
	if (code != cudaSuccess) {

		fprintf(stderr, "gpu_assert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);

	}

}

// prints out properties of CUDA capable GPUs
inline void print_dev_prop() {

	int ndev;
	cudaGetDeviceCount(&ndev);

	for (int i = 0; i < ndev; i++) {

		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);

		printf("[ Device %d ]:\n", i);
		printf("  Device name: %s\n", prop.name);
		printf("  Memory Clock Rate (MHz): %f\n", prop.memoryClockRate / 1.0e3);
		printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
		printf("  Peak Memory Bandwidth (GB/s): %f\n\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth / 8) / 1.0e6);

	}

}