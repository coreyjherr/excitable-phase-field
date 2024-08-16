#include "include/kernels.cuh"
#include "include/params.cuh"
#include <curand_kernel.h>

__global__ void initializePhiKernel(Real* phi, Param* p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < p->N && j < p->N && k < p->numCells) {
        int idx = k*p->N*p->N + j*p->N + i;
        Real x = (i-(Real)p->N/2)*p->dx;
        Real y = (j-(Real)p->N/2)*p->dx;
        phi[idx] = 0.5*(1-tanh(3*(sqrt(x*x + y*y) - p->r0)/sqrt(p->eps2)));
    }
} 

__global__ void getXIndexMaskedKernel(Real* phi, Real* buffer, Param* p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < p->N && j < p->N && k < p->numCells) {
        int idx = k*p->N*p->N + j*p->N + i;
        if (phi[idx] > 0.5)
            buffer[idx] = i;
        else
            buffer[idx] = 0.0/0.0;
    }
}

__global__ void getYIndexMaskedKernel(Real* phi, Real* buffer, Param* p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < p->N && j < p->N && k < p->numCells) {
        int idx = k*p->N*p->N + j*p->N + i;
        if (phi[idx] > 0.5)
            buffer[idx] = j;
        else
            buffer[idx] = 0.0/0.0;
    }
}

__global__ void rollGridXKernel(Real* phi, Real* buffer, int* d_grid_x, Param* p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    int N = p->N;

    if (i < N && j < N && k < p->numCells) {
        int idx = k*N*N + j*N + i;
        int x_distance = d_grid_x[k]%N;
        if (x_distance < N/8)
        {
            int shift = N/2 - x_distance;
            int new_i = (i + shift) % N;
            buffer[k*N*N + j*N + new_i] = phi[idx];
        }
        else if (x_distance > 7*N/8)
        {
            int shift = N/2 - x_distance;
            int new_i = (i + shift + N) % N;
            buffer[k*N*N + j*N + new_i] = phi[idx];
        }
        else
        {
            buffer[idx] = phi[idx];
        }
    }
}

__global__ void rollGridYKernel(Real* phi, Real* buffer, int* d_grid_y, Param* p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    int N = p->N;

    if (i < N && j < N && k < p->numCells) {
        int idx = k*N*N + j*N + i;
        int y_distance = d_grid_y[k]/N;
        if (y_distance < N/8)
        {
            int shift = N/2 - y_distance;
            int new_j = (j + shift) % N;
            buffer[k*N*N + new_j*N + i] = phi[idx];
        }
        else if (y_distance > 7*N/8)
        {
            int shift = N/2 - y_distance;
            int new_j = (j + shift + N) % N;
            buffer[k*N*N + new_j*N + i] = phi[idx];
        }
        else
        {
            buffer[idx] = phi[idx];
        }
    }
}

__device__ __forceinline__
Real laplacian(Real* phi, int i, int j, int k, Param* p) {
    int N = p->N;

    int i1 = (i + 1) % N;
    int i2 = (i - 1 + N) % N;
    int j1 = (j + 1) % N;
    int j2 = (j - 1 + N) % N;

    return (phi[k * N * N + j1 * N + i] +
            phi[k * N * N + j2 * N + i] +
            phi[k * N * N + j * N + i1] +
            phi[k * N * N + j * N + i2] -
            4 * phi[k * N * N + j * N + i]) /
           (p->dx * p->dx);
}

__device__ __forceinline__
Real derivativeX(Real* phi, int i, int j, int k, Param* p) {
    int N = p->N;
    int i1 = (i + 1) % N;
    int i2 = (i - 1 + N) % N;

    return (phi[k * N * N + j * N + i1] - phi[k * N * N + j * N + i2]) / (2 * p->dx);
}

__device__ __forceinline__
Real derivativeY(Real* phi, int i, int j, int k, Param* p) {
    int N = p->N;
    int j1 = (j + 1) % N;
    int j2 = (j - 1 + N) % N;

    return (phi[k * N * N + j1 * N + i] - phi[k * N * N + j2 * N + i]) / (2 * p->dx);
}

__global__ void updatePhiKernel(Real* phi, Real* A, Real* R, Param *p, Real* area, Real* collision_force, Real* buffer) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    int N = p->N;
    if (i < N && j < N && k < p->numCells) {
        int idx = k*N*N + j*N + i;
        buffer[idx] = 0;
        Real dphi = 0;
        
        Real phi_ = phi[idx];

        dphi += p->gamma * laplacian(phi, i, j, k, p);
        dphi -= (36 * p->gamma / p->eps2) * ( phi_ * (1 - phi_)*(1-phi_) - phi_*phi_ * (1 - phi_));

        Real phi_x = derivativeX(phi, i, j, k, p);
        Real phi_y = derivativeY(phi, i, j, k, p);

        dphi += sqrt(phi_x * phi_x + phi_y * phi_y) *
                (-p->B_s * (area[k]*p->dx*p->dx - p->S0) + p->alpha * A[idx] * phi_ -
                 p->m_repulsion * phi_*collision_force[idx]);

        buffer[idx] = dphi/p->xi;
    }

}

__global__ void UpdateAKernel(Real* phi, Real* A, Real* R, Param* p, Real* phi_t, curandState* state, Real* buffer) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    int N = p->N;

    if (i < N && j < N && k < p->numCells) {
        int idx = k*N*N + j*N + i;
        Real phi_ = phi[idx];
        Real A_ = A[idx];
        Real dA = 0;
        if (phi_ > 0.0025) {

            Real phi_x = derivativeX(phi, i, j, k, p);
            Real phi_y = derivativeY(phi, i, j, k, p);
            Real A_x = derivativeX(A, i, j, k, p);
            Real A_y = derivativeY(A, i, j, k, p);

            dA += p->D_A * laplacian(A, i, j, k, p);
            dA += p->D_A * (phi_x * A_x + phi_y * A_y)/phi_;

            dA += (p->A_t- A_)*(p->k_a*A_*A_/(p->K*p->K + A_*A_) + p->b*phi_) - (p->d1 + p->d2*R[idx])*A_;
            dA -= A_*phi_t[idx]/phi_;

            curandState localState = state[idx];
            Real noise = sqrt(p->sigma/p->dt)*curand_normal(&localState);
            state[idx] = localState;

            dA += noise;

            buffer[idx] = dA;
        }
        else {
            buffer[idx] = -A_/p->dt;
        }

    }
}

__global__ void updateRKernel(Real* phi, Real* A, Real* R, Param* p, Real* phi_t, curandState* state, Real* buffer) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    int N = p->N;

    if (i < N && j < N && k < p->numCells) {
        int idx = k*N*N + j*N + i;
        buffer[idx] = 0;
        Real phi_ = phi[idx];
        Real R_ = R[idx];
        Real dR = 0;
        if (phi_ > 0.0025) {

            Real phi_x = derivativeX(phi, i, j, k, p);
            Real phi_y = derivativeY(phi, i, j, k, p);
            Real R_x = derivativeX(R, i, j, k, p);
            Real R_y = derivativeY(R, i, j, k, p);

            dR += p->D_R * laplacian(R, i, j, k, p);
            dR += p->D_R * (phi_x * R_x + phi_y * R_y)/phi_;

            dR += (p->c2 * A[idx] - p->c1*R_)/p->tau;
            dR -= R_*phi_t[idx]/phi_;

            curandState localState = state[idx];
            Real noise = sqrt(p->sigma/p->dt)*curand_normal(&localState);
            state[idx] = localState;

            dR += noise;
            buffer[idx] = dR;

        }
        else {
            buffer[idx] = -R_/p->dt;
        }
    }
}

__global__ void setupCurandStates(curandState *state, unsigned long long seed, Param* p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < p->N && j < p->N && k < p->numCells) {
        int idx = k*p->N*p->N + j*p->N + i;
        curand_init(seed, idx, 0, &state[idx]);
    }
}

__global__ void addKernel(Real *a, Real *da, Param* p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    int N = p->N;

    if (i < N && j < N && k < p->numCells) {
        int idx = k*N*N + j*N + i;
        a[idx] = a[idx] + p->dt*da[idx];
    }

}
