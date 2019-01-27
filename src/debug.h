#pragma once

#include <cstdio>

#ifdef ENABLE_GPU_CHECK
#define gpu_check(ans) { gpu_assert((ans), __FILE__, __LINE__); }
#else
#define gpu_check(ans)
#endif

inline void gpu_assert(cudaError_t code, const char *file, int line, bool abort = true) {
	
	if (code != cudaSuccess) {

		fprintf(stderr, "gpu_assert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);

	}

}