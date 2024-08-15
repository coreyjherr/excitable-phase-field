#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas.h>

typedef double Real;

__global__ void UpdatePhiKernel(Real *phi, Real *a, Real *r, int numCells, Real dt, Real dx);

struct Param { 
    Real dt;
    Real dx;
    int N;
    // ...
};

class Model {
private:
    Real *phi, *A, *R;
    Real *d_phi, *d_A, *d_R;
    Param p;
    int numCells;
    cublasHandle_t handle;
public:
    Model(int numCells, Param p);
    ~Model();
    void UpdatePhi();
    void UpdateA();
    void UpdateR();
};

__global__ void UpdatePhiKernel();
__global__ void UpdateAKernel();
__global__ void UpdateRKernel();