#ifndef MODEL_CUH
#define MODEL_CUH

#include <cublas_v2.h>
#include <curand_kernel.h>

#ifndef Real2
#define Real2 float2
#endif

struct CudaTypes {
  curandState *d_states, *d_Rstates;
  int2 *d_grid_location, *grid_location, *d_grid_min, *d_grid_max;
  Real2 *com, *d_com, *comA, *d_comA, *d_division_axes, *division_axes;
  cublasHandle_t handle;
  dim3 grid, block, Cblocks, Cthreads_per_block;
};

#endif  // MODEL_CUH
