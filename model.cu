#include "include/indicators.hpp"
#include "include/model.cuh"
#include "include/error_check.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "include/kernels.cuh"
#include <iostream>
#include <fstream>
#include <math.h>
#include <thread>

#define USE_FLOAT

#ifdef USE_FLOAT
#define cublasSum(handle, n, x, incx, result) cublasSasum(handle, n, x, incx, result)
#define cublasAxpy(handle, n, alpha, x, incx, y, incy) cublasSaxpy(handle, n, alpha, x, incx, y, incy)
#define cublasIamin(handle, n, x, incx, result) cublasIsamin(handle, n, x, incx, result)
#define cublasIamax(handle, n, x, incx, result) cublasIsamax(handle, n, x, incx, result)
#define cublasSwap(handle, n, x, incx, y, incy) cublasSswap(handle, n, x, incx, y, incy)
#endif
#ifdef USE_DOUBLE
#define cublasSum(handle, n, x, incx, result) cublasDasum(handle, n, x, incx, result)
#define cublasAxpy(handle, n, alpha, x, incx, y, incy) cublasDaxpy(handle, n, alpha, x, incx, y, incy)
#define cublasIamin(handle, n, x, incx, result) cublasIdamin(handle, n, x, incx, result)
#define cublasIamax(handle, n, x, incx, result) cublasIdamax(handle, n, x, incx, result)
#define cublasSwap(handle, n, x, incx, y, incy) cublasDswap(handle, n, x, incx, y, incy)
#endif


Model::Model(Param param, int deviceID) {

    p = param;
    p.S0 = M_PI*p.r0*p.r0;

    numCells = p.numCells;

    phi = new Real[numCells*p.N*p.N];
    A = new Real[numCells*p.N*p.N];
    R = new Real[numCells*p.N*p.N];
    area = new Real[numCells];
    grid_location = new int2[numCells];

    cudaErrCheck(cudaSetDevice(deviceID));

    block = dim3(blockSizeX, blockSizeY, blockSizeZ);
    grid = dim3((p.N + blockSizeX - 1) / blockSizeX, (p.N + blockSizeY - 1) / blockSizeY, (numCells + blockSizeZ - 1) / blockSizeZ);

    // Allocate memory on device
    cudaErrCheck(cudaMalloc((void**)&d_Param, sizeof(Param)));
    cudaErrCheck(cudaMalloc((void**)&d_phi, numCells*p.N*p.N*sizeof(Real)));
    cudaErrCheck(cudaMalloc((void**)&d_A, numCells*p.N*p.N*sizeof(Real)));
    cudaErrCheck(cudaMalloc((void**)&d_R, numCells*p.N*p.N*sizeof(Real)));
    cudaErrCheck(cudaMalloc((void**)&d_area, numCells*sizeof(Real)));
    cudaErrCheck(cudaMalloc((void**)&d_collision_force, numCells*p.N*p.N*sizeof(Real)));
    cudaErrCheck(cudaMalloc((void**)&d_buffer, numCells*p.N*p.N*sizeof(Real)));
    cudaErrCheck(cudaMalloc((void**)&d_phi_t, numCells*p.N*p.N*sizeof(Real)));
    cudaErrCheck(cudaMalloc((void**)&d_grid_min, numCells*sizeof(int2)));
    cudaErrCheck(cudaMalloc((void**)&d_grid_max, numCells*sizeof(int2)));
    cudaErrCheck(cudaMalloc((void**)&d_grid_location, numCells*sizeof(int2)));

    // Initialize memory on device
    cudaErrCheck(cudaMemcpy(d_Param, &p, sizeof(Param), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemset(d_phi, 0, numCells*p.N*p.N*sizeof(Real)));
    cudaErrCheck(cudaMemset(d_A, 0, numCells*p.N*p.N*sizeof(Real)));
    cudaErrCheck(cudaMemset(d_R, 0, numCells*p.N*p.N*sizeof(Real)));
    cudaErrCheck(cudaMemset(d_area, 0, numCells*sizeof(Real)));
    cudaErrCheck(cudaMemset(d_collision_force, 0, numCells*p.N*p.N*sizeof(Real)));
    cudaErrCheck(cudaMemset(d_buffer, 0, numCells*p.N*p.N*sizeof(Real)));
    cudaErrCheck(cudaMemset(d_phi_t, 0, numCells*p.N*p.N*sizeof(Real)));

    // Initialize phi
    initializePhiKernel<<<grid, block>>>(d_phi, d_Param);

    // Initialize the grid location
    initializeCellGrid();
    cudaErrCheck(cudaMemcpy(d_grid_location, grid_location, numCells*sizeof(int2), cudaMemcpyHostToDevice));

    // Initialize cublas
    cublasErrCheck(cublasCreate(&handle));
    cublasErrCheck(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));

    // Initialize curand states
    cudaErrCheck(cudaMalloc((void**)&d_states, numCells*p.N*p.N*sizeof(curandState)));
    cudaErrCheck(cudaMalloc((void**)&d_Rstates, numCells*p.N*p.N*sizeof(curandState)));
    setupCurandStates<<<grid, block>>>(d_states, 0, d_Param);
    setupCurandStates<<<grid, block>>>(d_Rstates, 1, d_Param);
    curandErrCheck(cudaGetLastError());

    phi_output = std::ofstream("results/phi.dat", std::ios::out | std::ios::binary);
    A_output = std::ofstream("results/A.dat", std::ios::out | std::ios::binary);
    grid_location_output = std::ofstream("results/direction.dat", std::ios::out | std::ios::binary);
    grid_location_output.write(reinterpret_cast<char*>(&numCells), sizeof(int));

}

Model::~Model() {
    cudaErrCheck(cudaFree(d_phi));
    cudaErrCheck(cudaFree(d_A));
    cudaErrCheck(cudaFree(d_R));
    cudaErrCheck(cudaFree(d_area));
    cudaErrCheck(cudaFree(d_collision_force));
    cudaErrCheck(cudaFree(d_buffer));
    cudaErrCheck(cudaFree(d_phi_t));
    cudaErrCheck(cudaFree(d_grid_location));
    cudaErrCheck(cudaFree(d_grid_min));
    cudaErrCheck(cudaFree(d_grid_max));
    cudaErrCheck(cudaFree(d_Param));
    cudaErrCheck(cudaFree(d_states));
    cudaErrCheck(cudaFree(d_Rstates));

    delete[] phi;
    delete[] A;
    delete[] R;
    delete[] area;
    delete[] grid_location;

    cublasErrCheck(cublasDestroy(handle));

    // Close the output files after mutex lock
    std::lock_guard<std::mutex> guard(fileMutex);
    phi_output.close();
    A_output.close();
    grid_location_output.close();

}

void Model::initializeCellGrid() {
    int sqrt_numCells = sqrt(1.0*numCells)+1;
    for (int i = 0; i < sqrt_numCells; ++i) {
        for (int j = 0; j < sqrt_numCells; ++j) {
            if (i*sqrt_numCells + j >= numCells) {
                break;
            }
            grid_location[i*sqrt_numCells + j].x = p.initial_offset_x*j;
            grid_location[i*sqrt_numCells + j].y = p.initial_offset_y*i;
        }
    }
}

void Model::simulate(bool progressBar) {
    using namespace indicators;
    BlockProgressBar bar{
        option::BarWidth{80}, option::ForegroundColor{Color::white},
        option::FontStyles{std::vector<FontStyle>{FontStyle::bold}},
        option::MaxProgress{p.T}, option::ShowElapsedTime{true},option::ShowRemainingTime{true}};

    for (int t = 0; t < p.T; t++) {
        if (t % p.blind == 0) {
            updateGridLocation();
            writeInBackground();
        }

        updateCollisionForce();
        updateArea();
        updatePhi();
        updateA();
        updateR();

        if (progressBar) {
            bar.tick();
        }
    }
}


void Model::updateArea() {
    for (int i = 0; i < numCells; ++i) {
        cublasErrCheck(cublasSum(handle, p.N*p.N, d_phi + i*p.N*p.N, 1, d_area + i));
    }
}

void Model::updatePhi() {
    updatePhiKernel<<<grid, block>>>(d_phi, d_A, d_R, d_Param, d_area, d_collision_force, d_phi_t);
    cublasErrCheck(cublasAxpy(handle, numCells*p.N*p.N, &(d_Param->dt), d_phi_t, 1, d_phi, 1));
}

void Model::updateA() {
    UpdateAKernel<<<grid, block>>>(d_phi, d_A, d_R, d_Param, d_phi_t, d_states, d_buffer);
    cublasErrCheck(cublasAxpy(handle, numCells*p.N*p.N, &(d_Param->dt), d_buffer, 1, d_A, 1));
}

void Model::updateR() {
    updateRKernel<<<grid, block>>>(d_phi, d_A, d_R, d_Param, d_phi_t, d_Rstates, d_buffer);
    cublasErrCheck(cublasAxpy(handle, numCells*p.N*p.N, &(d_Param->dt), d_buffer, 1, d_R, 1));
}

void Model::updateCollisionForce() {
    updateCollisionForceKernel<<<grid, block>>>(d_phi, d_grid_location, d_Param, d_collision_force);
}

void Model::updateGridLocation() {
    getXIndexMaskedKernel<<<grid, block>>>(d_phi, d_buffer, d_Param);
    for (int i = 0; i < numCells; ++i) {
        cublasErrCheck(cublasIamax(handle, p.N*p.N, d_buffer + i*p.N*p.N, 1, &(d_grid_max + i)->x));
        cublasErrCheck(cublasIamin(handle, p.N*p.N, d_buffer + i*p.N*p.N, 1, &(d_grid_min + i)->x));
    }
    cudaErrCheck(cudaDeviceSynchronize());
    rollAllFieldsX();

    getYIndexMaskedKernel<<<grid, block>>>(d_phi, d_buffer, d_Param);
    for (int i = 0; i < numCells; ++i) {
        cublasErrCheck(cublasIamax(handle, p.N*p.N, d_buffer + i*p.N*p.N, 1, &((d_grid_max + i)->y)));
        cublasErrCheck(cublasIamin(handle, p.N*p.N, d_buffer + i*p.N*p.N, 1, &((d_grid_min + i)->y)));
    }
    cudaErrCheck(cudaDeviceSynchronize());
    rollAllFieldsY();
    cudaErrCheck(cudaDeviceSynchronize());

    updateGridLocationKernel<<<1, numCells>>>(d_grid_min, d_grid_location, d_Param);
    updateGridLocationKernel<<<1, numCells>>>(d_grid_max, d_grid_location, d_Param);

}

void Model::rollAllFieldsX() {
    rollGridXKernel<<<grid, block>>>(d_phi, d_buffer, d_grid_max, d_Param);
    cublasErrCheck(cublasSwap(handle, numCells*p.N*p.N, d_buffer, 1, d_phi, 1));

    rollGridXKernel<<<grid, block>>>(d_phi, d_buffer, d_grid_min, d_Param);
    cublasErrCheck(cublasSwap(handle, numCells*p.N*p.N, d_buffer, 1, d_phi, 1));

    rollGridXKernel<<<grid, block>>>(d_A, d_buffer, d_grid_max, d_Param);
    cublasErrCheck(cublasSwap(handle, numCells*p.N*p.N, d_buffer, 1, d_A, 1));

    rollGridXKernel<<<grid, block>>>(d_A, d_buffer, d_grid_min, d_Param);
    cublasErrCheck(cublasSwap(handle, numCells*p.N*p.N, d_buffer, 1, d_A, 1));

    rollGridXKernel<<<grid, block>>>(d_R, d_buffer, d_grid_max, d_Param);
    cublasErrCheck(cublasSwap(handle, numCells*p.N*p.N, d_buffer, 1, d_R, 1));

    rollGridXKernel<<<grid, block>>>(d_R, d_buffer, d_grid_min, d_Param);
    cublasErrCheck(cublasSwap(handle, numCells*p.N*p.N, d_buffer, 1, d_R, 1));
}

void Model::rollAllFieldsY() {
    rollGridYKernel<<<grid, block>>>(d_phi, d_buffer, d_grid_max, d_Param);
    cublasErrCheck(cublasSwap(handle, numCells*p.N*p.N, d_buffer, 1, d_phi, 1));

    rollGridYKernel<<<grid, block>>>(d_phi, d_buffer, d_grid_min, d_Param);
    cublasErrCheck(cublasSwap(handle, numCells*p.N*p.N, d_buffer, 1, d_phi, 1));

    rollGridYKernel<<<grid, block>>>(d_A, d_buffer, d_grid_max, d_Param);
    cublasErrCheck(cublasSwap(handle, numCells*p.N*p.N, d_buffer, 1, d_A, 1));

    rollGridYKernel<<<grid, block>>>(d_A, d_buffer, d_grid_min, d_Param);
    cublasErrCheck(cublasSwap(handle, numCells*p.N*p.N, d_buffer, 1, d_A, 1));

    rollGridYKernel<<<grid, block>>>(d_R, d_buffer, d_grid_max, d_Param);
    cublasErrCheck(cublasSwap(handle, numCells*p.N*p.N, d_buffer, 1, d_R, 1));

    rollGridYKernel<<<grid, block>>>(d_R, d_buffer, d_grid_min, d_Param);
    cublasErrCheck(cublasSwap(handle, numCells*p.N*p.N, d_buffer, 1, d_R, 1));
}


void Model::write() {
    // Write phi and A to file
    cudaErrCheck(cudaMemcpy(phi, d_phi, numCells*p.N*p.N*sizeof(Real), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(A, d_A, numCells*p.N*p.N*sizeof(Real), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(grid_location, d_grid_location, numCells*sizeof(int2), cudaMemcpyDeviceToHost));
    phi_output.write(reinterpret_cast<char*>(phi), numCells*p.N*p.N*sizeof(Real));
    A_output.write(reinterpret_cast<char*>(A), numCells*p.N*p.N*sizeof(Real));
    grid_location_output.write(reinterpret_cast<char*>(grid_location), numCells*sizeof(int2));
}

void Model::writeArray() {
    std::lock_guard<std::mutex> guard(fileMutex); // Lock the mutex
    cudaErrCheck(cudaMemcpy(phi, d_phi, numCells*p.N*p.N*sizeof(Real), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(A, d_A, numCells*p.N*p.N*sizeof(Real), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(grid_location, d_grid_location, numCells*sizeof(int2), cudaMemcpyDeviceToHost));
    phi_output.write(reinterpret_cast<char*>(phi), numCells*p.N*p.N*sizeof(Real));
    A_output.write(reinterpret_cast<char*>(A), numCells*p.N*p.N*sizeof(Real));
    grid_location_output.write(reinterpret_cast<char*>(grid_location), numCells*sizeof(int2));
}

// Function to run the file writing operation in a background thread
void Model::writeInBackground() {
    std::thread backgroundThread(&Model::writeArray, this);
    backgroundThread.detach(); // Detach the thread to run in the background
    
}
