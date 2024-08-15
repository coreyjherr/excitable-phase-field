#ifndef ERROR_CHECK_CUH
#define ERROR_CHECK_CUH

#include <iostream>

#define cudaErrCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      std::cout << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << "\n";
      if (abort) exit(code);
   }
}

#endif