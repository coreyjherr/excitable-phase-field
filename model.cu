#include "include/model.cuh"
#include "include/error_check.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "include/kernels.cuh"
#include <iostream>
#include <fstream>
#include <math.h>
#include <thread>

Model::Model(Param param) {
    p = param;
    numCells = p.numCells;
    phi = new Real[numCells*p.N*p.N];
    A = new Real[numCells*p.N*p.N];
    R = new Real[numCells*p.N*p.N];
    area = new Real[numCells];

    int blockSizeX = 16;
    int blockSizeY = 16;
    int blockSizeZ = 1;

    p.S0 = M_PI*p.r0*p.r0;

    phi_output = std::ofstream("phi.dat", std::ios::out | std::ios::binary);
    A_output = std::ofstream("A.dat", std::ios::out | std::ios::binary);

    cudaErrCheck(cudaSetDevice(0));

    block = dim3(blockSizeX, blockSizeY, blockSizeZ);
    grid = dim3((p.N + blockSizeX - 1) / blockSizeX, (p.N + blockSizeY - 1) / blockSizeY, (numCells + blockSizeZ - 1) / blockSizeZ);

    cudaErrCheck(cudaMalloc((void**)&d_Param, sizeof(Param)));
    cudaErrCheck(cudaMemcpy(d_Param, &p, sizeof(Param), cudaMemcpyHostToDevice));

    // Allocate memory on device
    cudaErrCheck(cudaMalloc((void**)&d_phi, numCells*p.N*p.N*sizeof(Real)));
    cudaErrCheck(cudaMalloc((void**)&d_A, numCells*p.N*p.N*sizeof(Real)));
    cudaErrCheck(cudaMalloc((void**)&d_R, numCells*p.N*p.N*sizeof(Real)));
    cudaErrCheck(cudaMalloc((void**)&d_area, numCells*sizeof(Real)));
    cudaErrCheck(cudaMalloc((void**)&d_collision_force, numCells*p.N*p.N*sizeof(Real)));
    cudaErrCheck(cudaMalloc((void**)&d_buffer, numCells*p.N*p.N*sizeof(Real)));
    cudaErrCheck(cudaMalloc((void**)&d_phi_t, numCells*p.N*p.N*sizeof(Real)));
    cudaErrCheck(cudaMalloc((void**)&d_grid_min_x, numCells*sizeof(int)));
    cudaErrCheck(cudaMalloc((void**)&d_grid_min_y, numCells*sizeof(int)));
    cudaErrCheck(cudaMalloc((void**)&d_grid_max_x, numCells*sizeof(int)));
    cudaErrCheck(cudaMalloc((void**)&d_grid_max_y, numCells*sizeof(int)));

    cudaErrCheck(cudaMemset(d_phi, 0, numCells*p.N*p.N*sizeof(Real)));
    cudaErrCheck(cudaMemset(d_A, 0, numCells*p.N*p.N*sizeof(Real)));
    cudaErrCheck(cudaMemset(d_R, 0, numCells*p.N*p.N*sizeof(Real)));
    cudaErrCheck(cudaMemset(d_area, 0, numCells*sizeof(Real)));
    cudaErrCheck(cudaMemset(d_collision_force, 0, numCells*p.N*p.N*sizeof(Real)));
    cudaErrCheck(cudaMemset(d_buffer, 0, numCells*p.N*p.N*sizeof(Real)));
    cudaErrCheck(cudaMemset(d_phi_t, 0, numCells*p.N*p.N*sizeof(Real)));

    initializePhiKernel<<<grid, block>>>(d_phi, d_Param);

    cublasErrCheck(cublasCreate(&handle));
    cublasErrCheck(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));

    cudaErrCheck(cudaMalloc((void**)&d_states, numCells*p.N*p.N*sizeof(curandState)));
    cudaErrCheck(cudaMalloc((void**)&d_Rstates, numCells*p.N*p.N*sizeof(curandState)));
    setupCurandStates<<<grid, block>>>(d_states, 0, d_Param);
    setupCurandStates<<<grid, block>>>(d_Rstates, 1, d_Param);
    curandErrCheck(cudaGetLastError());

}

Model::~Model() {
    cudaErrCheck(cudaFree(d_phi));
    cudaErrCheck(cudaFree(d_A));
    cudaErrCheck(cudaFree(d_R));

    delete[] phi;
    delete[] A;
    delete[] R;

    cublasErrCheck(cublasDestroy(handle));
    phi_output.close();
    A_output.close();

}

void Model::updateArea() {
    for (int i = 0; i < numCells; ++i) {
        cublasErrCheck(cublasSasum(handle, p.N*p.N, d_phi + i*p.N*p.N, 1, d_area + i));
    }
#ifdef VERBOSE
    cudaErrCheck(cudaMemcpy(area, d_area, numCells*sizeof(Real), cudaMemcpyDeviceToHost));
    for (int i = 0; i < numCells; ++i) {
        std::cout << area[i] << " ";
    }
#endif
}

void Model::updatePhi() {
    updatePhiKernel<<<grid, block>>>(d_phi, d_A, d_R, d_Param, d_area, d_collision_force, d_phi_t);
    cublasErrCheck(cublasSaxpy(handle, numCells*p.N*p.N, &(d_Param->dt), d_phi_t, 1, d_phi, 1));
}

void Model::updateA() {
    UpdateAKernel<<<grid, block>>>(d_phi, d_A, d_R, d_Param, d_phi_t, d_states, d_buffer);
    cublasErrCheck(cublasSaxpy(handle, numCells*p.N*p.N, &(d_Param->dt), d_buffer, 1, d_A, 1));
}

void Model::updateR() {
    updateRKernel<<<grid, block>>>(d_phi, d_A, d_R, d_Param, d_phi_t, d_Rstates, d_buffer);
    cublasErrCheck(cublasSaxpy(handle, numCells*p.N*p.N, &(d_Param->dt), d_buffer, 1, d_R, 1));
}

void Model::rollAllFieldsX() {
    rollGridXKernel<<<grid, block>>>(d_phi, d_buffer, d_grid_max_x, d_Param);
    cublasErrCheck(cublasSswap(handle, numCells*p.N*p.N, d_buffer, 1, d_phi, 1));

    rollGridXKernel<<<grid, block>>>(d_phi, d_buffer, d_grid_min_x, d_Param);
    cublasErrCheck(cublasSswap(handle, numCells*p.N*p.N, d_buffer, 1, d_phi, 1));

    rollGridXKernel<<<grid, block>>>(d_A, d_buffer, d_grid_max_x, d_Param);
    cublasErrCheck(cublasSswap(handle, numCells*p.N*p.N, d_buffer, 1, d_A, 1));

    rollGridXKernel<<<grid, block>>>(d_A, d_buffer, d_grid_min_x, d_Param);
    cublasErrCheck(cublasSswap(handle, numCells*p.N*p.N, d_buffer, 1, d_A, 1));

    rollGridXKernel<<<grid, block>>>(d_R, d_buffer, d_grid_max_x, d_Param);
    cublasErrCheck(cublasSswap(handle, numCells*p.N*p.N, d_buffer, 1, d_R, 1));

    rollGridXKernel<<<grid, block>>>(d_R, d_buffer, d_grid_min_x, d_Param);
    cublasErrCheck(cublasSswap(handle, numCells*p.N*p.N, d_buffer, 1, d_R, 1));
}

void Model::rollAllFieldsY() {
    rollGridYKernel<<<grid, block>>>(d_phi, d_buffer, d_grid_max_y, d_Param);
    cublasErrCheck(cublasSswap(handle, numCells*p.N*p.N, d_buffer, 1, d_phi, 1));

    rollGridYKernel<<<grid, block>>>(d_phi, d_buffer, d_grid_min_y, d_Param);
    cublasErrCheck(cublasSswap(handle, numCells*p.N*p.N, d_buffer, 1, d_phi, 1));

    rollGridYKernel<<<grid, block>>>(d_A, d_buffer, d_grid_max_y, d_Param);
    cublasErrCheck(cublasSswap(handle, numCells*p.N*p.N, d_buffer, 1, d_A, 1));

    rollGridYKernel<<<grid, block>>>(d_A, d_buffer, d_grid_min_y, d_Param);
    cublasErrCheck(cublasSswap(handle, numCells*p.N*p.N, d_buffer, 1, d_A, 1));

    rollGridYKernel<<<grid, block>>>(d_R, d_buffer, d_grid_max_y, d_Param);
    cublasErrCheck(cublasSswap(handle, numCells*p.N*p.N, d_buffer, 1, d_R, 1));

    rollGridYKernel<<<grid, block>>>(d_R, d_buffer, d_grid_min_y, d_Param);
    cublasErrCheck(cublasSswap(handle, numCells*p.N*p.N, d_buffer, 1, d_R, 1));
}


void Model::updateGridLocation() {
    getXIndexMaskedKernel<<<grid, block>>>(d_phi, d_buffer, d_Param);
    for (int i = 0; i < numCells; ++i) {
        cublasErrCheck(cublasIsamax(handle, p.N*p.N, d_buffer + i*p.N*p.N, 1, d_grid_max_x + i));
        cublasErrCheck(cublasIsamin(handle, p.N*p.N, d_buffer + i*p.N*p.N, 1, d_grid_min_x + i));
    }
    rollAllFieldsX();

    getYIndexMaskedKernel<<<grid, block>>>(d_phi, d_buffer, d_Param);
    for (int i = 0; i < numCells; ++i) {
        cublasErrCheck(cublasIsamax(handle, p.N*p.N, d_buffer + i*p.N*p.N, 1, d_grid_max_y + i));
        cublasErrCheck(cublasIsamin(handle, p.N*p.N, d_buffer + i*p.N*p.N, 1, d_grid_min_y + i));
    }
    rollAllFieldsY();

}

void Model::write() {
    // Write phi and A to file
    cudaErrCheck(cudaMemcpy(phi, d_phi, numCells*p.N*p.N*sizeof(Real), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(A, d_A, numCells*p.N*p.N*sizeof(Real), cudaMemcpyDeviceToHost));
    // start a thread to write to file
    std::thread t;
    phi_output.write(reinterpret_cast<char*>(phi), numCells*p.N*p.N*sizeof(Real));
    A_output.write(reinterpret_cast<char*>(A), numCells*p.N*p.N*sizeof(Real));
}

void Model::writeArray() {
    std::lock_guard<std::mutex> guard(fileMutex); // Lock the mutex
    cudaErrCheck(cudaMemcpy(phi, d_phi, numCells*p.N*p.N*sizeof(Real), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(A, d_A, numCells*p.N*p.N*sizeof(Real), cudaMemcpyDeviceToHost));
    phi_output.write(reinterpret_cast<char*>(phi), numCells*p.N*p.N*sizeof(Real));
    A_output.write(reinterpret_cast<char*>(A), numCells*p.N*p.N*sizeof(Real));
}

// Function to run the file writing operation in a background thread
void Model::writeArrayToFileInBackground() {
    std::thread backgroundThread(&Model::writeArray, this);
    backgroundThread.detach(); // Detach the thread to run in the background
}
