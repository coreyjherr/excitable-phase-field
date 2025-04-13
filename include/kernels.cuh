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
#ifndef KERNELS_CUH
#define KERNELS_CUH

#include "params.hpp"
// #include <__clang_cuda_runtime_wrapper.h>
#include <curand_kernel.h>

// #define USE_FLOAT

#ifdef USE_FLOAT
using Real2 = float2;
#else
using Real2 = double2;
#endif

__device__ __forceinline__ Real
laplacian(Real* phi, int i, int j, int k, Param p);
__device__ __forceinline__ Real
derivativeX(Real* phi, int i, int j, int k, Param p);
__device__ __forceinline__ Real
derivativeY(Real* phi, int i, int j, int k, Param p);
__device__ __forceinline__ Real
laplacianChem(Real* C, int i, int j, Param* p);
__device__ __forceinline__ Real
derivativeXChem(Real* C, int i, int j, Param* p);
__device__ __forceinline__ Real
derivativeYChem(Real* C, int i, int j, Param* p);

__global__ void
initializePhiKernel(Real* phi, Param* p);
__global__ void
updatePhiKernel(Real* phi,
                Real* A,
                Real* R,
                Param* p,
                Real* area,
                int2* grid_location,
                Real* collision_force,
                Real* phi_t,
                int t);
__global__ void
updatePhiKernelChemo(Real* phi,
                     Real* A,
                     Real* R,
                     Real* C,
                     Param* p,
                     Real* area,
                     int2* grid_location,
                     Real* collision_force,
                     Real* buffer,
                     int t);
__global__ void
updateCKernel(Real* C, Real* Cbuffer, Param* p);
__global__ void
updateAKernel(Real* phi,
              Real* A,
              Real* R,
              Param* p,
              Real* phi_t,
              int2* grid_location,
              curandState* state,
              Real* buffer);
__global__ void
updateRKernel(Real* phi,
              Real* A,
              Real* R,
              Param* p,
              Real* phi_t,
              int2* grid_location,
              curandState* state,
              Real* buffer);
__global__ void
updateRKernelChemo(Real* phi,
                   Real* A,
                   Real* R,
                   Real* C,
                   Param* p,
                   Real* phi_t,
                   int2* grid_location,
                   curandState* state,
                   Real* buffer);
__global__ void
setupCurandStates(curandState* state,
                  unsigned long long seed,
                  unsigned long long offset,
                  Param* p);
__global__ void
getYIndexMaskedKernel(Real* phi, Real* buffer, Param* p, Real cutoff);
__global__ void
getXIndexMaskedKernel(Real* phi, Real* buffer, Param* p, Real cutoff);
__global__ void
rollGridYKernel(Real* phi, Real* buffer, int2* d_grid, Param* p);
__global__ void
rollGridXKernel(Real* phi, Real* buffer, int2* d_grid, Param* p);
__global__ void
updateGridLocationKernel(int2* d_grid_minmax, int2* d_grid_location, Param* p);
__global__ void
updateCollisionForceKernel(Real* phi,
                           int2* grid_locations,
                           Param* p,
                           Real* collision_force);
__global__ void
updateBoundingIndex(int2* grid_cublas_index, Param* p);
__global__ void
createCOMXKernel(Real* phi, Real* buffer, Param* p);
__global__ void
createCOMYKernel(Real* phi, Real* buffer, Param* p);
__global__ void
createCOMAYKernel(Real* phi, Real* A, Real* buffer, Param* p);
__global__ void
createCOMAXKernel(Real* phi, Real* A, Real* buffer, Param* p);
__global__ void
multiplyKernel(Real* in1, Real* in2, Real* out, Param* p);
__global__ void
makeLabelImage(Real* phi, Real* buffer, int2* grid_location, Param* p);
__global__ void
copyToBacklogKernel(Real* d_outputs,
                    Real2* d_com,
                    Real2* d_comA,
                    Real* d_area,
                    Real* d_Aarea,
                    int2* d_grid_location,
                    int numCells,
                    int step,
                    int backlog_index);
__global__ void
copyToX(Real* tmp_com, Real2* d_com, int numCells);
__global__ void
copyToY(Real* tmp_com, Real2* d_com, int numCells);

#endif // KERNELS_CUH
