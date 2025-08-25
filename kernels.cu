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
#include <curand_kernel.h>
#include <curand_uniform.h>

#include "include/kernels.cuh"
#include "include/params.hpp"

// Typedef for device function pointer
typedef Real (*BoundaryConditionFunc)(const int2&, int, Param*, int, Real);

__device__ Real volumeGrowthCurve(Real t, Param& p) {
  return p.S0 / 2 * exp(log(2.) * t / p.division_time);
}
__global__ void initializePhiKernel(Real* phi, Param* p) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i < N && j < N && k < p[k].numCellsInitial) {
    int idx = k * N * N + j * N + i;
    Real x = (i - (Real)N / 2) * p[k].dx;
    Real y = (j - (Real)N / 2) * p[k].dx;
    phi[idx] =
        0.5 * (1 - tanh(3 * (sqrt(x * x + y * y) - p[k].r0) / sqrt(p[k].eps2)));
  }
}

__global__ void updateBoundingIndex(int2* grid_cublas_index,
                                    Param* p,
                                    int numCells) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < numCells) {
    int2 minmax = grid_cublas_index[i];
    grid_cublas_index[i].x = (minmax.x - 1) % N;
    grid_cublas_index[i].y = (minmax.y - 1) / N;
  }
}

__global__ void getXIndexMaskedKernel(Real* phi,
                                      Real* buffer,
                                      Param* p,
                                      Real cutoff,
                                      int numCells) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i < N && j < N && k < numCells) {
    int idx = k * N * N + j * N + i;
    if (phi[idx] > cutoff)
      buffer[idx] = i;
    else
      buffer[idx] = 0.0 / 0.0;
  }
}

__global__ void getYIndexMaskedKernel(Real* phi,
                                      Real* buffer,
                                      Param* p,
                                      Real cutoff,
                                      int numCells) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i < N && j < N && k < numCells) {
    int idx = k * N * N + j * N + i;
    if (phi[idx] > cutoff)
      buffer[idx] = j;
    else
      buffer[idx] = 0.0 / 0.0;
  }
}

__global__ void rollGridXKernel(Real* phi,
                                Real* buffer,
                                int2* d_grid,
                                Param* p,
                                int numCells) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i < N && j < N && k < numCells) {
    int idx = k * N * N + j * N + i;
    // cublas uses 1 based indexing
    int x_distance = (d_grid[k].x - 1) % N;
    if (x_distance < N / 8) {
      int shift = N / 2 - N / 8;
      int new_i = (i + shift) % N;
      buffer[k * N * N + j * N + new_i] = phi[idx];
    } else if (x_distance > 7 * N / 8) {
      int shift = N / 2 - 7 * N / 8;
      int new_i = (i + shift + N) % N;
      buffer[k * N * N + j * N + new_i] = phi[idx];
    } else {
      buffer[idx] = phi[idx];
    }
  }
}

__global__ void rollGridYKernel(Real* phi,
                                Real* buffer,
                                int2* d_grid,
                                Param* p,
                                int numCells) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i < N && j < N && k < numCells) {
    int idx = k * N * N + j * N + i;
    int y_distance = (d_grid[k].y - 1) / N;
    if (y_distance < N / 8) {
      int shift = N / 2 - N / 8;
      int new_j = (j + shift) % N;
      buffer[k * N * N + new_j * N + i] = phi[idx];
    } else if (y_distance > 7 * N / 8) {
      int shift = N / 2 - 7 * N / 8;
      int new_j = (j + shift + N) % N;
      buffer[k * N * N + new_j * N + i] = phi[idx];
    } else {
      buffer[idx] = phi[idx];
    }
  }
}

__global__ void updateGridLocationKernel(int2* d_grid_minmax,
                                         int2* d_grid_location,
                                         Param* p,
                                         int numCells) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < numCells) {
    int2 minmax = d_grid_minmax[k];

    int x_distance = (minmax.x - 1) % N;
    int y_distance = (minmax.y - 1) / N;

    if (x_distance < N / 8) {
      d_grid_location[k].x -= N / 2 - N / 8;
    } else if (x_distance > 7 * N / 8) {
      d_grid_location[k].x -= N / 2 - 7 * N / 8;
    }

    if (y_distance < N / 8) {
      d_grid_location[k].y -= N / 2 - N / 8;
    } else if (y_distance > 7 * N / 8) {
      d_grid_location[k].y -= N / 2 - 7 * N / 8;
    }
  }
}

__device__ __forceinline__ Real
laplacian(Real* phi, int i, int j, int k, Param* p) {
  int i1 = (i + 1) % N;
  int i2 = (i - 1 + N) % N;
  int j1 = (j + 1) % N;
  int j2 = (j - 1 + N) % N;

  return (phi[k * N * N + j1 * N + i] + phi[k * N * N + j2 * N + i] +
          phi[k * N * N + j * N + i1] + phi[k * N * N + j * N + i2] -
          4 * phi[k * N * N + j * N + i]) /
         (p[k].dx * p[k].dx);
}

__device__ __forceinline__ Real laplacianChem(Real* C, int i, int j, Param* p) {
  int Nc = p[0].Nchem;

  // non-periodic boundary conditions, let the boundary be zero

  Real Cij1 = i + 1 < Nc ? C[j * Nc + i + 1] : 0;
  Real Cij2 = i - 1 >= 0 ? C[j * Nc + i - 1] : 0;
  Real Cj1i = j + 1 < Nc ? C[(j + 1) * Nc + i] : 0;
  Real Cj2i = j - 1 >= 0 ? C[(j - 1) * Nc + i] : 0;

  return (Cij1 + Cij2 + Cj1i + Cj2i - 4 * C[j * Nc + i]) / (p[0].dx * p[0].dx);
}

__device__ __forceinline__ Real
derivativeX(Real* phi, int i, int j, int k, Param* p) {
  int i1 = (i + 1) % N;
  int i2 = (i - 1 + N) % N;

  return (phi[k * N * N + j * N + i1] - phi[k * N * N + j * N + i2]) /
         (2 * p[k].dx);
}

__device__ __forceinline__ Real
derivativeY(Real* phi, int i, int j, int k, Param* p) {
  int j1 = (j + 1) % N;
  int j2 = (j - 1 + N) % N;

  return (phi[k * N * N + j1 * N + i] - phi[k * N * N + j2 * N + i]) /
         (2 * p[k].dx);
}

__device__ __forceinline__ Real derivativeXChem(Real* C,
                                                int i,
                                                int j,
                                                Param* p) {
  int Nc = p[0].Nchem;
  Real Cij1 = i + 1 < Nc ? C[j * Nc + i + 1] : 0;
  Real Cij2 = i - 1 >= 0 ? C[j * Nc + i - 1] : 0;

  return (Cij1 - Cij2) / (2 * p[0].dx);
}

__device__ __forceinline__ Real derivativeYChem(Real* C,
                                                int i,
                                                int j,
                                                Param* p) {
  int Nc = p[0].Nchem;
  Real Cj1i = j + 1 < Nc ? C[(j + 1) * Nc + i] : 0;
  Real Cj2i = j - 1 >= 0 ? C[(j - 1) * Nc + i] : 0;

  return (Cj1i - Cj2i) / (2 * p[0].dx);
}

// Device function for circular boundary condition

__device__ Real circular_boundary_condition(const int2& thread_loc,
                                            int t,
                                            Param* p,
                                            int k,
                                            Real phi_) {
  Real wall_force = 0;
  Real R_wall_osc = p[k].R_wall;
  const Real thread_R2 =
      (thread_loc.x * thread_loc.x + thread_loc.y * thread_loc.y) * p[k].dx *
      p[k].dx;
  if (thread_R2 > .5 * p[k].R_wall * p[k].R_wall) {
    wall_force =
        0.5 * (1 + tanh(3 * (sqrt(thread_R2) - R_wall_osc) / sqrt(p[k].eps2))) *
        p[k].m_wall_repulsion * phi_;
  }
  return wall_force;
}

// Device function for square boundary condition
__device__ Real square_boundary_condition(const int2& thread_loc,
                                          int t,
                                          Param* p,
                                          int k,
                                          Real phi_) {
  Real wall_force = 0;
  int wall_x_loc = (p[k].R_wall + p[0].chem_force * sin(2 * M_PI * t * p[k].dt /
                                                        p[0].chem_decay)) /
                   p[k].dx;
  if (thread_loc.x < -p[k].R_wall / p[k].dx || thread_loc.x > wall_x_loc ||
      abs(thread_loc.y) > p[k].R_wall / p[k].dx) {
    wall_force = p[k].m_wall_repulsion * phi_;
  }
  return wall_force;
}

__global__ void updatePhiKernel(Real* phi,
                                Real* A,
                                Real* R,
                                Param* p,
                                Real* area,
                                int2* grid_location,
                                Real* collision_force,
                                Real* phi_t,
                                Real* time_since_division,
                                int t,
                                int numCells,
                                BoundaryConditionFunc boundary_condition) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i < N && j < N && k < numCells) {
    int idx = k * N * N + j * N + i;
    phi_t[idx] = 0;
    Real dphi = 0;

    Real phi_ = phi[idx];

    dphi += p[k].gamma * laplacian(phi, i, j, k, p);
    dphi -= (36 * p[k].gamma / p[k].eps2) *
            (phi_ * (1 - phi_) * (1 - phi_) - phi_ * phi_ * (1 - phi_));

    Real phi_x = derivativeX(phi, i, j, k, p);
    Real phi_y = derivativeY(phi, i, j, k, p);

    const int2 thread_loc = {grid_location[k].x + i - N / 2,
                             grid_location[k].y + j - N / 2};
    Real wall_force = 0;
    if (boundary_condition != nullptr) {
      wall_force = boundary_condition(thread_loc, t, p, k, phi_);
    }

    Real t_r = time_since_division[k];

    dphi += sqrt(phi_x * phi_x + phi_y * phi_y) *
                (-p[k].B_s * (area[k] * p[k].dx * p[k].dx -
                              volumeGrowthCurve(t_r, p[k])) +
                 p[k].alpha * A[idx] * phi_) -
            collision_force[idx] - wall_force;

    phi_t[idx] = dphi / p[k].xi;
  }
}

__global__ void updatePhiKernelChemo(Real* phi,
                                     Real* A,
                                     Real* R,
                                     Real* C,
                                     Param* p,
                                     Real* area,
                                     int2* grid_location,
                                     Real* collision_force,
                                     Real* buffer,
                                     int t,
                                     int numCells) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i < N && j < N && k < numCells) {
    int idx = k * N * N + j * N + i;
    buffer[idx] = 0;
    Real dphi = 0;

    Real phi_ = phi[idx];

    dphi += p[k].gamma * laplacian(phi, i, j, k, p);
    dphi -= (36 * p[k].gamma / p[k].eps2) *
            (phi_ * (1 - phi_) * (1 - phi_) - phi_ * phi_ * (1 - phi_));

    Real phi_x = derivativeX(phi, i, j, k, p);
    Real phi_y = derivativeY(phi, i, j, k, p);

    const int2 thread_loc = {grid_location[k].x + i - N / 2,
                             grid_location[k].y + j - N / 2};
    int R_ind = p[0].Nchem / 2;
    Real wall_force = 0;
    if (thread_loc.x * thread_loc.x + thread_loc.y * thread_loc.y >
        p[k].R_wall * p[k].R_wall / p[k].dx / p[k].dx) {
      wall_force = p[k].m_wall_repulsion * phi_;
    }

    dphi += sqrt(phi_x * phi_x + phi_y * phi_y) *
            (-p[k].B_s * (area[k] * p[k].dx * p[k].dx - p[k].S0) +
             p[k].alpha * A[idx] * phi_ - collision_force[idx] - wall_force);

    buffer[idx] = dphi / p[k].xi;
    const int2 C_loc = {grid_location[k].x + R_ind + i,
                        grid_location[k].y + R_ind + j};
    if (phi_ > 0.5)
      // C[C_loc.y*p[0].Nchem * C_loc.x] += p[k].dt*p[k].chem_strength*phi_;
      C[C_loc.y * p[0].Nchem + C_loc.x] += p[k].dt * p[k].chem_strength * phi_;
  }
}

__global__ void updateAKernel(Real* phi,
                              Real* A,
                              Real* R,
                              Param* p,
                              Real* phi_t,
                              int2* grid_location,
                              curandState* state,
                              Real* buffer,
                              Real* time_since_division,
                              int t,
                              int numCells) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i < N && j < N && k < numCells) {
    int idx = k * N * N + j * N + i;
    Real phi_ = phi[idx];
    Real A_ = A[idx];
    Real dA = 0;
    if (phi_ > 0.0025) {
      Real phi_x = derivativeX(phi, i, j, k, p);
      Real phi_y = derivativeY(phi, i, j, k, p);
      Real A_x = derivativeX(A, i, j, k, p);
      Real A_y = derivativeY(A, i, j, k, p);

      int ridge_oscillation_amplitude =
          round(p[0].chem_force *
                (1 + sin(2 * M_PI * t * p[k].dt / p[0].chem_decay)));
      Real ridge_factor = positive_modulus(grid_location[k].x + i, 20) <
                                  ridge_oscillation_amplitude
                              ? p[k].chem_strength
                              : 1;
      // Real ridge_factor =
      // (1+p[0].chem_strength*0.5*(1+sin(2*M_PI*(grid_location[k].x+i)/15)));
      // Real ridge_factor = 1;

      dA += p[k].D_A * laplacian(A, i, j, k, p);
      dA += p[k].D_A * (phi_x * A_x + phi_y * A_y) / phi_;

      dA += (p[k].A_t - A_) * (p[k].k_a / (ridge_factor)*A_ * A_ /
                                   (p[k].K * p[k].K + A_ * A_) +
                               p[k].b) -
            (p[k].d1 + p[k].d2 * R[idx]) * A_;
      dA -= A_ * phi_t[idx] / phi_;

      curandState localState = state[idx];
      Real noise = sqrt(p[k].sigma / p[k].dt) * curand_normal(&localState);
      state[idx] = localState;

      dA += noise;

      buffer[idx] = dA;
    } else {
      buffer[idx] = -A_ / p[k].dt;
    }
  }
}

__global__ void updateRKernel(Real* phi,
                              Real* A,
                              Real* R,
                              Param* p,
                              Real* phi_t,
                              int2* grid_location,
                              curandState* state,
                              Real* buffer,
                              int numCells) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i < N && j < N && k < numCells) {
    int idx = k * N * N + j * N + i;
    buffer[idx] = 0;
    Real phi_ = phi[idx];
    Real R_ = R[idx];
    Real dR = 0;
    if (phi_ > 0.0025) {
      Real phi_x = derivativeX(phi, i, j, k, p);
      Real phi_y = derivativeY(phi, i, j, k, p);
      Real R_x = derivativeX(R, i, j, k, p);
      Real R_y = derivativeY(R, i, j, k, p);

      dR += p[k].D_R * laplacian(R, i, j, k, p);
      dR += p[k].D_R * (phi_x * R_x + phi_y * R_y) / phi_;

      // if chemotaxis is off use the chem_strength parameter to determine ridge
      // strength int ridge_width = 4; int ridge_dist = 20; Real k_c2 =
      // ((grid_location[k].x + i) % ridge_dist + ridge_dist) % ridge_dist <
      // ridge_width ? p[k].chem_strength : 0; Real k_tau = ((grid_location[k].x
      // +i) % ridge_dist + ridge_dist) % ridge_dist < ridge_width ?
      // p[k].chem_force : 0;
      Real k_c2 = 0;
      Real k_tau = 0;

      dR += (p[k].c2 * A[idx] / (1 + k_c2) - p[k].c1 * R_) / p[k].tau * 1 /
            (1 + k_tau);
      dR -= R_ * phi_t[idx] / phi_;

      curandState localState = state[idx];
      Real noise = sqrt(p[k].sigma / p[k].dt) * curand_normal(&localState);
      state[idx] = localState;

      dR += noise;
      buffer[idx] = dR;
    } else {
      buffer[idx] = -R_ / p[k].dt;
    }
  }
}

__global__ void updateRKernelChemo(Real* phi,
                                   Real* A,
                                   Real* R,
                                   Real* C,
                                   Param* p,
                                   Real* phi_t,
                                   int2* grid_location,
                                   curandState* state,
                                   Real* buffer,
                                   int numCells) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i < N && j < N && k < numCells) {
    int idx = k * N * N + j * N + i;
    buffer[idx] = 0;
    Real phi_ = phi[idx];
    Real R_ = R[idx];
    Real dR = 0;
    if (phi_ > 0.0025) {
      Real phi_x = derivativeX(phi, i, j, k, p);
      Real phi_y = derivativeY(phi, i, j, k, p);
      Real R_x = derivativeX(R, i, j, k, p);
      Real R_y = derivativeY(R, i, j, k, p);

      int R_ind = p[0].Nchem / 2;
      int2 C_loc = {grid_location[k].x + i + R_ind,
                    grid_location[k].y + j + R_ind};

      Real C_x = derivativeXChem(C, C_loc.x, C_loc.y, p);
      Real C_y = derivativeYChem(C, C_loc.x, C_loc.y, p);
      Real C_norm = sqrt(C_x * C_x + C_y * C_y);

      dR += p[k].D_R * laplacian(R, i, j, k, p);
      dR += p[k].D_R * (phi_x * R_x + phi_y * R_y) / phi_;

      dR += (p[k].c2 * exp(-p[0].chem_force * sqrt(C_x * C_x + C_y * C_y)) *
                 A[idx] -
             p[k].c1 * R_) /
            p[k].tau;
      dR -= R_ * phi_t[idx] / phi_;

      curandState localState = state[idx];
      Real noise = sqrt(p[k].sigma / p[k].dt) * curand_normal(&localState);
      state[idx] = localState;

      dR += noise;
      buffer[idx] = dR;
    } else {
      buffer[idx] = -R_ / p[k].dt;
    }
  }
}

__global__ void updateCKernel(Real* C, Real* Cbuffer, Param* p) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int Nchem = p[0].Nchem;

  if (i < Nchem && j < Nchem) {
    int idx = j * Nchem + i;
    Real dC = p[0].dt * (p[0].Dchem * laplacianChem(C, i, j, p) -
                         p[0].chem_decay * C[idx]);
    Cbuffer[idx] = C[idx] + dC;
  }
}

__global__ void updateCollisionForceKernel(Real* phi,
                                           int2* grid_locations,
                                           Param* p,
                                           Real* collision_force,
                                           int numCells) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i < N && j < N && k < numCells) {
    int idx = k * N * N + j * N + i;
    Real phi_ = phi[idx];
    collision_force[idx] = 0;
    if (phi_ > 0.0025) {
      for (int l = 0; l < k; ++l) {
        int dx = grid_locations[l].x - grid_locations[k].x;
        int dy = grid_locations[l].y - grid_locations[k].y;
        // const Real sharp_grad_factor = 32.5;
        const Real sharp_grad_factor = 42.5;

        if ((i - dx) >= 0 && (i - dx) < N && (j - dy) >= 0 && (j - dy) < N) {
          int idx_l = l * N * N + (j - dy) * N + (i - dx);
          Real phi_l = phi[idx_l];
          if (phi_l > 0.0025) {
            Real phi_kx = derivativeX(phi, i, j, k, p);
            Real phi_ky = derivativeY(phi, i, j, k, p);
            Real phi_lx = derivativeX(phi, i - dx, j - dy, l, p);
            Real phi_ly = derivativeY(phi, i - dx, j - dy, l, p);
            Real repulsion_force = p[k].m_repulsion * phi_ * phi_l;
            Real adhesion_force =
                p[k].m_adhesion * (phi_kx * phi_lx + phi_ky * phi_ly);

            collision_force[idx] +=
                repulsion_force * phi_l +
                adhesion_force / sqrt(1 + sharp_grad_factor * phi_lx * phi_lx +
                                      sharp_grad_factor * phi_ly * phi_ly);
            collision_force[idx_l] +=
                repulsion_force * phi_ +
                adhesion_force / sqrt(1 + sharp_grad_factor * phi_kx * phi_kx +
                                      sharp_grad_factor * phi_ky * phi_ky);
          }
        }
      }
    }
    // collision_force[idx] = force;
  }
}

__global__ void setupCurandStates(curandState* state,
                                  unsigned long long seed,
                                  unsigned long long offset,
                                  Param* p) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i < N && j < N && k < p[0].maxCells) {
    int idx = k * N * N + j * N + i;
    curand_init(seed, idx + offset, 0, &state[idx]);
  }
}

__global__ void addKernel(Real* a, Real* da, Param* p, int numCells) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i < N && j < N && k < numCells) {
    int idx = k * N * N + j * N + i;
    a[idx] = a[idx] + p[k].dt * da[idx];
  }
}

__global__ void createCOMXKernel(Real* phi,
                                 Real* buffer,
                                 Param* p,
                                 int numCells) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i < N && j < N && k < numCells) {
    int idx = k * N * N + j * N + i;
    Real phi_ = phi[idx];
    buffer[idx] = phi_ * i;
  }
}

__global__ void createCOMYKernel(Real* phi,
                                 Real* buffer,
                                 Param* p,
                                 int numCells) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i < N && j < N && k < numCells) {
    int idx = k * N * N + j * N + i;
    Real phi_ = phi[idx];
    buffer[idx] = phi_ * j;
  }
}

__global__ void createCOMAXKernel(Real* phi,
                                  Real* A,
                                  Real* buffer,
                                  Param* p,
                                  int numCells) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i < N && j < N && k < numCells) {
    int idx = k * N * N + j * N + i;
    buffer[idx] = phi[idx] * A[idx] * i;
  }
}

__global__ void createCOMAYKernel(Real* phi,
                                  Real* A,
                                  Real* buffer,
                                  Param* p,
                                  int numCells) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i < N && j < N && k < numCells) {
    int idx = k * N * N + j * N + i;
    buffer[idx] = phi[idx] * A[idx] * j;
  }
}

__global__ void multiplyKernel(Real* in1,
                               Real* in2,
                               Real* out,
                               Param* p,
                               int numCells) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i < N && j < N && k < numCells) {
    int idx = k * N * N + j * N + i;
    out[idx] = in1[idx] * in2[idx];
  }
}

__global__ void makeLabelImage(Real* phi,
                               Real* buffer,
                               int2* grid_location,
                               Param* p,
                               int numCells) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i < N && j < N && k < numCells) {
    int idx = k * N * N + j * N + i;
    buffer[idx] = 0;
    if (phi[idx] > 0.5)
      buffer[idx] = k + 1;
  }
}

__global__ void copyToBacklogKernel(Real* d_outputs,
                                    Real2* d_com,
                                    Real2* d_comA,
                                    Real* d_area,
                                    Real* d_Aarea,
                                    int2* d_grid_location,
                                    int numCells,
                                    int step,
                                    int backlog_index) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < numCells) {
    int offset = backlog_index * numCells * 10;
    d_outputs[offset + i * 10 + 0] = static_cast<Real>(i);
    d_outputs[offset + i * 10 + 1] = static_cast<Real>(step);
    d_outputs[offset + i * 10 + 2] = static_cast<Real>(d_grid_location[i].x);
    d_outputs[offset + i * 10 + 3] = static_cast<Real>(d_grid_location[i].y);
    d_outputs[offset + i * 10 + 4] = d_area[i];
    d_outputs[offset + i * 10 + 5] = d_Aarea[i];
    if (d_area[i] > 1e-6) {
      d_com[i].x /= d_area[i];
      d_com[i].y /= d_area[i];
    } else {
      d_com[i].x = 0;
      d_com[i].y = 0;
    }
    if (d_Aarea[i] > 1e-6) {
      d_comA[i].x /= d_Aarea[i];
      d_comA[i].y /= d_Aarea[i];
    } else {
      d_comA[i].x = 0;
      d_comA[i].y = 0;
    }
    d_outputs[offset + i * 10 + 6] = d_com[i].x;
    d_outputs[offset + i * 10 + 7] = d_com[i].y;
    d_outputs[offset + i * 10 + 8] = d_comA[i].x;
    d_outputs[offset + i * 10 + 9] = d_comA[i].y;
  }
}

__global__ void copyToX(Real* tmp_com, Real2* d_com, int numCells) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < numCells) {
    d_com[i].x = tmp_com[i];
  }
}

__global__ void copyToY(Real* tmp_com, Real2* d_com, int numCells) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < numCells) {
    d_com[i].y = tmp_com[i];
  }
}

__forceinline__ __device__ int positive_modulus(int a, int b) {
  return (a % b + b) % b;
}

__global__ void divideCells(Real* phi,
                            Real* A,
                            Real* R,
                            Param* param,
                            int* new_indices,
                            Real2* com,
                            Real* area,
                            Real2* division_axes,
                            int numCells) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i < N && j < N && k < numCells) {
    int idx = k * N * N + j * N + i;
    if (new_indices[k] > 0) {
      int new_idx = new_indices[k] * N * N + j * N + i;
      // split cells in half along a random axis
      Real x_cell = i - com[k].x / area[k];
      Real y_cell = j - com[k].y / area[k];
      Real phi_ = phi[idx];
      if (x_cell * division_axes[k].x + y_cell * division_axes[k].y > 0) {
        phi[new_idx] = phi_;
        phi[idx] = 0;

        A[new_idx] = 0;
        A[idx] = 0;

        R[new_idx] = 0;
        R[idx] = 0;
      } else {
        A[idx] = 0;
        R[idx] = 0;
      }
    }
  }
}

__global__ void add_time(Real* time, Real dt, int numCells) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < numCells) {
    time[i] += dt;
  }
}
