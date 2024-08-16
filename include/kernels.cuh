#ifndef KERNELS_CUH
#define KERNELS_CUH

#include "params.cuh"
#include <curand_kernel.h>

__device__ __forceinline__ Real laplacian(Real *phi, int i, int j, int k, Param p);
__device__ __forceinline__ Real derivativeX(Real *phi, int i, int j, int k, Param p);
__device__ __forceinline__ Real derivativeY(Real *phi, int i, int j, int k, Param p);

__global__ void initializePhiKernel(Real* phi, Param* p);
__global__ void updatePhiKernel(Real* phi, Real* A, Real* R, Param *p, Real* area, Real* collision_force, Real* buffer);
__global__ void UpdateAKernel(Real* phi, Real* A, Real* R, Param* p, Real* phi_t, curandState* state, Real* buffer);
__global__ void updateRKernel(Real* phi, Real* A, Real* R, Param* p, Real* phi_t, curandState* state, Real* buffer);
__global__ void setupCurandStates(curandState *state, unsigned long long seed, Param* p);
__global__ void getYIndexMaskedKernel(Real* phi, Real* buffer, Param* p);
__global__ void getXIndexMaskedKernel(Real* phi, Real* buffer, Param* p);
__global__ void rollGridYKernel(Real* phi, Real* buffer, int* d_grid_y, Param* p);
__global__ void rollGridXKernel(Real* phi, Real* buffer, int* d_grid_x, Param* p);

#endif // KERNELS_CUH
