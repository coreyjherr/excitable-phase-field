/*
MIT License

Copyright (c) 2025 Corey Herr
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE
*/
#ifndef ERROR_CHECK_CUH
#define ERROR_CHECK_CUH

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <iostream>

#define cudaErrCheck(ans)                                                      \
  {                                                                            \
    gpuAssert((ans), __FILE__, __LINE__);                                      \
  }
inline void
gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
  if (code != cudaSuccess) {
    std::cout << "GPUassert: " << cudaGetErrorString(code) << " " << file << " "
              << line << "\n";
    if (abort)
      exit(code);
  }
}

#define cublasErrCheck(ans)                                                    \
  {                                                                            \
    cublasAssert((ans), __FILE__, __LINE__);                                   \
  }
inline void
cublasAssert(cublasStatus_t code, const char* file, int line, bool abort = true)
{
  if (code != CUBLAS_STATUS_SUCCESS) {
    std::cout << "CUBLASassert: " << cublasGetStatusString(code) << " " << file
              << " " << line << "\n";
    if (abort)
      exit(code);
  }
}

#define curandErrCheck(ans)                                                    \
  {                                                                            \
    gpuAssert((ans), __FILE__, __LINE__);                                      \
  }
inline void
gpuAssert(curandStatus_t code, const char* file, int line, bool abort = true)
{
  if (code != CURAND_STATUS_SUCCESS) {
    fprintf(stderr, "GPUassert: %d %s %d\n", code, file, line);
    if (abort)
      exit(code);
  }
}

#endif
