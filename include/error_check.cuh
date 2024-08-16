#ifndef ERROR_CHECK_CUH
#define ERROR_CHECK_CUH

#include "model.cuh"
#include <iostream>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>

#define cudaErrCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      std::cout << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << "\n";
      if (abort) exit(code);
   }
}

#define cublasErrCheck(ans) { cublasAssert((ans), __FILE__, __LINE__); }
inline void cublasAssert(cublasStatus_t code, const char *file, int line, bool abort=true)
{
   if (code != CUBLAS_STATUS_SUCCESS) 
   {
      std::cout << "CUBLASassert: " << cublasGetStatusString(code) << " " << file << " " << line << "\n";
      if (abort) exit(code);
   }
}

#define curandErrCheck(ans) {gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(curandStatus_t code, const char *file, int line, bool abort=true)
{
	if (code != CURAND_STATUS_SUCCESS)
	{
		fprintf(stderr, "GPUassert: %d %s %d\n", code, file, line);
		if (abort) exit(code);
	}
}

#endif
