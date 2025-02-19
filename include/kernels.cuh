#ifndef KERNELS_CUH
#define KERNELS_CUH

#include "params.cuh"
// #include <__clang_cuda_runtime_wrapper.h>
#include <curand_kernel.h>

__device__ __forceinline__ Real laplacian(Real *phi, int i, int j, int k, Param p);
__device__ __forceinline__ Real derivativeX(Real *phi, int i, int j, int k, Param p);
__device__ __forceinline__ Real derivativeY(Real *phi, int i, int j, int k, Param p);
__device__ __forceinline__ Real laplacianChem(Real* C, int i, int j, Param* p);
__device__ __forceinline__ Real derivativeXChem(Real* C, int i, int j, Param* p);
__device__ __forceinline__ Real derivativeYChem(Real* C, int i, int j, Param* p);

__global__ void initializePhiKernel(Real* phi, Param* p);
__global__ void updatePhiKernel(Real* phi, Real* A, Real* R, Param *p, Real* area, int2* grid_location, Real* collision_force, Real* phi_t, int t);
__global__ void updatePhiKernelChemo(Real* phi, Real* A, Real* R, Real* C, Param *p, Real* area, int2* grid_location, Real* collision_force, Real* buffer, int t);
__global__ void updateCKernel(Real* C, Real* Cbuffer, Param* p);
__global__ void updateAKernel(Real* phi, Real* A, Real* R, Param* p, Real* phi_t, int2* grid_location, curandState* state, Real* buffer);
__global__ void updateRKernel(Real* phi, Real* A, Real* R, Param* p, Real* phi_t, int2* grid_location, curandState* state, Real* buffer);
__global__ void updateRKernelChemo(Real* phi, Real* A, Real* R, Real* C, Param* p, Real* phi_t, int2* grid_location, curandState* state, Real* buffer);
__global__ void setupCurandStates(curandState *state, unsigned long long seed, unsigned long long offset, Param* p);
__global__ void getYIndexMaskedKernel(Real* phi, Real* buffer, Param* p, Real cutoff);
__global__ void getXIndexMaskedKernel(Real* phi, Real* buffer, Param* p, Real cutoff);
__global__ void rollGridYKernel(Real* phi, Real* buffer, int2* d_grid, Param* p);
__global__ void rollGridXKernel(Real* phi, Real* buffer, int2* d_grid, Param* p);
__global__ void updateGridLocationKernel(int2* d_grid_minmax, int2* d_grid_location, Param* p);
__global__ void updateCollisionForceKernel(Real* phi, int2* grid_locations, Param* p, Real* collision_force);
__global__ void updateBoundingIndex(int2* grid_cublas_index, Param* p);
__global__ void createCOMXKernel(Real* phi, Real* buffer, Param* p);
__global__ void createCOMYKernel(Real* phi, Real* buffer, Param* p);
__global__ void createCOMAYKernel(Real* phi, Real* A, Real* buffer, Param* p);
__global__ void createCOMAXKernel(Real* phi, Real* A, Real* buffer, Param* p);
__global__ void multiplyKernel(Real* in1, Real* in2, Real* out, Param* p);
__global__ void makeLabelImage(Real* phi, Real* buffer, int2* grid_location, Param* p);

#endif // KERNELS_CUH
