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
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>

#include "include/error_check.cuh"
#include "include/indicators.hpp"
#include "include/kernels.cuh"
#include "include/model.h"
#include "include/params.hpp"

#ifndef Real2
#define Real2 float2
#endif

#define USE_FLOAT

#ifdef USE_FLOAT
#define cublasRasum(handle, n, x, incx, result) \
  cublasSasum(handle, n, x, incx, result)
#define cublasRaxpy(handle, n, alpha, x, incx, y, incy) \
  cublasSaxpy(handle, n, alpha, x, incx, y, incy)
#define cublasIramin(handle, n, x, incx, result) \
  cublasIsamin(handle, n, x, incx, result)
#define cublasIramax(handle, n, x, incx, result) \
  cublasIsamax(handle, n, x, incx, result)
#define cublasRswap(handle, n, x, incx, y, incy) \
  cublasSswap(handle, n, x, incx, y, incy)
#define cublasRcopy(handle, n, x, incx, y, incy) \
  cublasScopy(handle, n, x, incx, y, incy)
#endif
#ifndef USE_FLOAT
#define cublasRasum(handle, n, x, incx, result) \
  cublasDasum(handle, n, x, incx, result)
#define cublasRaxpy(handle, n, alpha, x, incx, y, incy) \
  cublasDaxpy(handle, n, alpha, x, incx, y, incy)
#define cublasIramin(handle, n, x, incx, result) \
  cublasIdamin(handle, n, x, incx, result)
#define cublasIramax(handle, n, x, incx, result) \
  cublasIdamax(handle, n, x, incx, result)
#define cublasRswap(handle, n, x, incx, y, incy) \
  cublasDswap(handle, n, x, incx, y, incy)
#define cublasRcopy(handle, n, x, incx, y, incy) \
  cublasDcopy(handle, n, x, incx, y, incy)
#endif

#define NUM_OUTPUT_COLUMNS 10
#define BACKLOG_SIZE 20

Real initializeParamArray(Param p, Param* p_array) {
  // Initialize the parameter array with the same parameters but different
  // radius evenly distributed between r0 and r0 + delta_r Parameters: p: Param
  // struct with the parameters p_array: Array of Param structs to be
  // initialized Return: the total radius of the population for the grid
  // location initialization
  srand(0);
  Real total_area = 0;
  for (int i = 0; i < p.numCells; ++i) {
    p_array[i] = p;
    p_array[i].r0 = p.r0 + (p.delta_r * rand()) / (RAND_MAX + 1.0);
    p_array[i].S0 = M_PI * p_array[i].r0 * p_array[i].r0;
    total_area += p_array[i].S0;
  }
  for (int i = 0; i < p.numCells; ++i) {
    if (p.R_wall == 0)
      p_array[i].R_wall = sqrt(total_area / (M_PI * p.packing_fraction));
    p_array[i].Nchem = 2 * p_array[i].R_wall / p_array[i].dx + p.N;
  }

  return sqrt(total_area / (M_PI * p.packing_fraction));
}

struct Model::CudaTypes {
  curandState *d_states, *d_Rstates;
  int2 *d_grid_location, *grid_location, *d_grid_min, *d_grid_max;
  Real2 *com, *d_com, *comA, *d_comA;
  cublasHandle_t handle;
  dim3 grid, block, Cblocks, Cthreads_per_block;
};

Model::Model(Param param,
             int deviceID,
             std::string basePath,
             unsigned long long seed,
             std::string simulationID,
             int outputType) {
  cudaTypes = new CudaTypes;

  p = param;
  p.S0 = M_PI * p.r0 * p.r0;
  numCells = p.numCells;
  if (outputType == WRITE_FULL_OUTPUT || outputType == WRITE_CONDENSED_OUTPUT) {
    writeFullOutput_ = true;
  } else {
    writeFullOutput_ = false;
  }

  if (outputType == WRITE_CONDENSED_OUTPUT) {
    writeCondensedOutput_ = true;
  } else {
    writeCondensedOutput_ = false;
  }

  if (outputType == WRITE_FULL_OUTPUT || outputType == WRITE_CONDENSED_OUTPUT ||
      outputType == WRITE_DYNAMICS_OUTPUT) {
    writeDynamicsOutput_ = true;
  } else {
    writeDynamicsOutput_ = false;
  }

  deviceID_ = deviceID;
  cudaErrCheck(cudaSetDevice(deviceID_));
  rngSeed = seed;

  cudaTypes->block = dim3(blockSizeX, blockSizeY, blockSizeZ);
  cudaTypes->grid = dim3((p.N + blockSizeX - 1) / blockSizeX,
                         (p.N + blockSizeY - 1) / blockSizeY,
                         (numCells + blockSizeZ - 1) / blockSizeZ);

  Param* paramArray = new Param[numCells];
  Real populationRadius = initializeParamArray(p, paramArray);
#ifdef USE_CHEMO
  p.Nchem = paramArray[0].Nchem;
#else
  p.Nchem = 0;
#endif
  cudaTypes->Cblocks = dim3(p.Nchem / 32 + 1, p.Nchem / 32 + 1);
  cudaTypes->Cthreads_per_block = dim3(32, 32);
  // std::cout << "Nchem: " << p.Nchem << "\n";
  cudaErrCheck(cudaMalloc((void**)&d_Param, sizeof(Param) * numCells));
  cudaErrCheck(cudaMemcpy(d_Param, paramArray, sizeof(Param) * numCells,
                          cudaMemcpyHostToDevice));
  // delete[] paramArray;

  // Initialize device and host pointers
  initializeDeviceMemory();
  initializeHostMemory();

  Real* ones = new Real[p.N * p.N];
  for (int i = 0; i < p.N * p.N; ++i) {
    ones[i] = 1;
  }
  cudaErrCheck(cudaMalloc((void**)&d_ones, p.N * p.N * sizeof(Real)));
  cudaErrCheck(cudaMemcpy(d_ones, ones, p.N * p.N * sizeof(Real),
                          cudaMemcpyHostToDevice));

  Real *beta, *alpha;
  beta = new Real;
  alpha = new Real;
  *beta = 0;
  *alpha = 1;
  cudaErrCheck(cudaMalloc((void**)&d_beta, sizeof(Real)));
  cudaErrCheck(cudaMalloc((void**)&d_alpha, sizeof(Real)));
  cudaErrCheck(cudaMemcpy(d_beta, beta, sizeof(Real), cudaMemcpyHostToDevice));
  cudaErrCheck(
      cudaMemcpy(d_alpha, alpha, sizeof(Real), cudaMemcpyHostToDevice));

  // Initialize phi
  initializePhiKernel<<<cudaTypes->grid, cudaTypes->block>>>(d_phi, d_Param);
  cudaErrCheck(cudaGetLastError());

  // Initialize the grid location
  initializeCellGrid(populationRadius);
  cudaErrCheck(cudaMemcpy(cudaTypes->d_grid_location, cudaTypes->grid_location,
                          numCells * sizeof(int2), cudaMemcpyHostToDevice));

  // Initialize cublas
  cublasErrCheck(cublasCreate(&(cudaTypes->handle)));
  cublasErrCheck(
      cublasSetPointerMode(cudaTypes->handle, CUBLAS_POINTER_MODE_DEVICE));

  // Initialize curand states
  cudaErrCheck(cudaMalloc((void**)&cudaTypes->d_states,
                          numCells * p.N * p.N * sizeof(curandState)));
  // cudaErrCheck(cudaMalloc((void**)&d_Rstates,
  // numCells*p.N*p.N*sizeof(curandState)));
  setupCurandStates<<<cudaTypes->grid, cudaTypes->block>>>(cudaTypes->d_states,
                                                           rngSeed, 0, d_Param);
  // cudaErrCheck(cudaGetLastError());
  // setupCurandStates<<<cudaTypes->grid, cudaTypes->block>>>(d_Rstates,
  // rngSeed, numCells*p.N*p.N, d_Param);
  cudaErrCheck(cudaGetLastError());
  curandErrCheck(cudaGetLastError());

#ifdef USE_DEVICE_BACKLOG
  cudaErrCheck(
      cudaMalloc((void**)&d_output,
                 numCells * BACKLOG_SIZE * sizeof(Real) * NUM_OUTPUT_COLUMNS));
  cudaErrCheck(cudaMallocHost(
      (void**)&h_output,
      numCells * BACKLOG_SIZE * sizeof(Real) * NUM_OUTPUT_COLUMNS));
#endif
  cudaMalloc((void**)&d_tmp_com, numCells * sizeof(Real));

  // Get the current time and date
  std::string time_str = simulationID;
  if (time_str == "") {
    std::ostringstream oss;
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
    time_str = oss.str();
  }
  time_str += "_" + std::to_string(deviceID_);

  if (writeFullOutput_) {
    initializeOutput(basePath, time_str);
  }
  if (writeDynamicsOutput_) {
    initializeDynamicsOutput(basePath, time_str);
  }
  if (outputType != WRITE_NONE) {
    writeParamToTOML(p, basePath + "param_" + time_str + ".toml", rngSeed);
  }

  threadsNumCells = std::min(numCells, 1024);
  blocksNumCells = (numCells + threadsNumCells - 1) / threadsNumCells;
}

Model::~Model() {
  cudaErrCheck(cudaFree(d_phi));
  cudaErrCheck(cudaFree(d_A));
  cudaErrCheck(cudaFree(d_R));
  cudaErrCheck(cudaFree(d_area));
  cudaErrCheck(cudaFree(d_collision_force));
  cudaErrCheck(cudaFree(d_buffer));
  cudaErrCheck(cudaFree(d_phi_t));
  cudaErrCheck(cudaFree(cudaTypes->d_grid_location));
  cudaErrCheck(cudaFree(cudaTypes->d_grid_min));
  cudaErrCheck(cudaFree(cudaTypes->d_grid_max));
  cudaErrCheck(cudaFree(d_Param));
  cudaErrCheck(cudaFree(cudaTypes->d_states));
  // cudaErrCheck(cudaFree(d_Rstates));
  cudaErrCheck(cudaFree(cudaTypes->d_com));
  cudaErrCheck(cudaFree(cudaTypes->d_comA));

  cudaErrCheck(cudaFreeHost(phi));
  cudaErrCheck(cudaFreeHost(A));
  cudaErrCheck(cudaFreeHost(R));
  cudaErrCheck(cudaFreeHost(area));
  cudaErrCheck(cudaFreeHost(Aarea));
  cudaErrCheck(cudaFreeHost(cudaTypes->grid_location));
  cudaErrCheck(cudaFreeHost(cudaTypes->com));
  cudaErrCheck(cudaFreeHost(cudaTypes->comA));
  cudaErrCheck(cudaFreeHost(C));

  cublasErrCheck(cublasDestroy(cudaTypes->handle));

  // Close the output files after mutex lock
  std::lock_guard<std::mutex> guard(fileMutex);
  if (writeFullOutput_) {
    phi_output.close();
    A_output.close();
    label_output.close();
  }
  dynamics_output.close();
#ifdef USE_CHEMO
  C_output.close();
#endif
  delete cudaTypes;
}

void Model::initializeHostMemory() {
  cudaErrCheck(
      cudaMallocHost((void**)&phi, numCells * p.N * p.N * sizeof(Real)));
  cudaErrCheck(cudaMallocHost((void**)&A, numCells * p.N * p.N * sizeof(Real)));
  cudaErrCheck(cudaMallocHost((void**)&R, numCells * p.N * p.N * sizeof(Real)));
  cudaErrCheck(cudaMallocHost((void**)&area, numCells * sizeof(Real)));
  cudaErrCheck(cudaMallocHost((void**)&Aarea, numCells * sizeof(Real)));
  cudaErrCheck(cudaMallocHost((void**)&cudaTypes->grid_location,
                              numCells * sizeof(int2)));
  cudaErrCheck(
      cudaMallocHost((void**)&cudaTypes->com, numCells * sizeof(Real2)));
  cudaErrCheck(
      cudaMallocHost((void**)&cudaTypes->comA, numCells * sizeof(Real2)));
  cudaErrCheck(cudaMallocHost((void**)&C, p.Nchem * p.Nchem * sizeof(Real)));
}

void Model::initializeDeviceMemory() {
  // Allocate memory on device
  cudaErrCheck(cudaMalloc((void**)&d_phi, numCells * p.N * p.N * sizeof(Real)));
  cudaErrCheck(cudaMalloc((void**)&d_A, numCells * p.N * p.N * sizeof(Real)));
  cudaErrCheck(cudaMalloc((void**)&d_R, numCells * p.N * p.N * sizeof(Real)));
  cudaErrCheck(cudaMalloc((void**)&d_area, numCells * sizeof(Real)));
  cudaErrCheck(cudaMalloc((void**)&d_Aarea, numCells * sizeof(Real)));
  cudaErrCheck(cudaMalloc((void**)&d_collision_force,
                          numCells * p.N * p.N * sizeof(Real)));
  cudaErrCheck(
      cudaMalloc((void**)&d_buffer, numCells * p.N * p.N * sizeof(Real)));
  cudaErrCheck(
      cudaMalloc((void**)&d_phi_t, numCells * p.N * p.N * sizeof(Real)));
  cudaErrCheck(
      cudaMalloc((void**)&cudaTypes->d_grid_min, numCells * sizeof(int2)));
  cudaErrCheck(
      cudaMalloc((void**)&cudaTypes->d_grid_max, numCells * sizeof(int2)));
  cudaErrCheck(
      cudaMalloc((void**)&cudaTypes->d_grid_location, numCells * sizeof(int2)));
  cudaErrCheck(cudaMalloc((void**)&cudaTypes->d_com, numCells * sizeof(Real2)));
  cudaErrCheck(
      cudaMalloc((void**)&cudaTypes->d_comA, numCells * sizeof(Real2)));
  cudaErrCheck(cudaMalloc((void**)&d_C, p.Nchem * p.Nchem * sizeof(Real)));
  cudaErrCheck(
      cudaMalloc((void**)&d_Cbuffer, p.Nchem * p.Nchem * sizeof(Real)));

  // Initialize memory on device
  cudaErrCheck(cudaMemset(d_phi, 0, numCells * p.N * p.N * sizeof(Real)));
  cudaErrCheck(cudaMemset(d_A, 0, numCells * p.N * p.N * sizeof(Real)));
  cudaErrCheck(cudaMemset(d_R, 0, numCells * p.N * p.N * sizeof(Real)));
  cudaErrCheck(cudaMemset(d_area, 0, numCells * sizeof(Real)));
  cudaErrCheck(cudaMemset(d_Aarea, 0, numCells * sizeof(Real)));
  cudaErrCheck(
      cudaMemset(d_collision_force, 0, numCells * p.N * p.N * sizeof(Real)));
  cudaErrCheck(cudaMemset(d_buffer, 0, numCells * p.N * p.N * sizeof(Real)));
  cudaErrCheck(cudaMemset(d_phi_t, 0, numCells * p.N * p.N * sizeof(Real)));
  cudaErrCheck(cudaMemset(cudaTypes->d_com, 0, numCells * sizeof(Real2)));
  cudaErrCheck(cudaMemset(cudaTypes->d_comA, 0, numCells * sizeof(Real2)));
  cudaErrCheck(cudaMemset(d_C, 0, p.Nchem * p.Nchem * sizeof(Real)));
  cudaErrCheck(cudaMemset(d_Cbuffer, 0, p.Nchem * p.Nchem * sizeof(Real)));
}

void Model::initializeCellGrid(Real populationRadius) {
  Real phi = (1 + sqrt(5)) / 2;
  int b = round(2 * sqrt(numCells));
  for (int i = 0; i < numCells; ++i) {
    int k = i + 1;
    Real r;
    if (k > numCells - b)
      r = 1;
    else
      r = sqrt(1.0 * k - 0.5) / sqrt(1.0 * numCells - (1.0 * b + 1) / 2);
    Real theta = 2 * M_PI * k / (phi * phi);
    Real rCircle = populationRadius * r;
    cudaTypes->grid_location[i].x = rCircle / p.dx * cos(theta);
    cudaTypes->grid_location[i].y = rCircle / p.dx * sin(theta);
  }
}

void Model::initializeOutput(std::string basePath, std::string time_str) {
  A_output.open(basePath + "A_" + time_str + ".dat", std::ios::binary);
  phi_output.open(basePath + "phi_" + time_str + ".dat", std::ios::binary);
  label_output.open(basePath + "label_" + time_str + ".dat", std::ios::binary);
#ifdef USE_CHEMO
  C_output.open(basePath + "C_" + time_str + ".dat", std::ios::binary);
#endif
}

void Model::initializeDynamicsOutput(std::string basePath,
                                     std::string time_str) {
  dynamics_output.open(basePath + "dynamics_" + time_str + ".dat",
                       std::ios::binary);
  // dynamics_output <<
  // "t,cell,grid_offset_x,grid_offset_y,phi_area,A_area,com_x,com_y,comA_x,comA_y"
  // << "\n"; write the number of columns and byte size of columns in the
  // dynamics output file
  float numColumns = 10;
  float realWidth = sizeof(Real);
  dynamics_output.write(reinterpret_cast<char*>(&numColumns), sizeof(float));
  dynamics_output.write(reinterpret_cast<char*>(&realWidth), sizeof(float));
}

void Model::simulate(bool progressBar, bool useCondensedOutput) {
  using namespace indicators;
  BlockProgressBar bar{
      option::BarWidth{80},
      option::ForegroundColor{Color::white},
      option::FontStyles{std::vector<FontStyle>{FontStyle::bold}},
      option::MaxProgress{p.T / 100},
      option::ShowElapsedTime{true},
      option::ShowRemainingTime{true}};

  BlockProgressBar startup_bar{
      option::BarWidth{80},
      option::ForegroundColor{Color::white},
      option::FontStyles{std::vector<FontStyle>{FontStyle::bold}},
      option::MaxProgress{startupSteps / 100},
      option::ShowElapsedTime{true},
      option::ShowRemainingTime{true}};

  int earlyExit = 0;
  cudaErrCheck(cudaDeviceSynchronize());
  for (int t = 0; t < startupSteps; t++) {
    // Early stages to stabilize the system
    updateArea();
    updateCollisionForce();
    updatePhi(t);
    if (t % 100 == 0) {
      updateGridLocation();
      if (progressBar)
        startup_bar.tick();
    }
  }
  if (progressBar)
    std::cout << "\n";

  for (int t = 0; t < p.T; t++) {
    updateArea();

    if (t % 100 == 0) {
      updateGridLocation();
      if (progressBar) {
        bar.tick();
      }
    }

    if (t % p.blind == 0) {
      updateCOM();
      if (writeDynamicsOutput_) {
        writeDynamics(t);
      }
    }

    // Hardcode the full output to every 400 steps
    if (t % 400 == 0 && writeFullOutput_) {
      earlyExit = writeFullOutput();
    }

    updateCollisionForce();
    updatePhi(t);
#ifdef USE_CHEMO
    updateC();
#endif
    updateA();
    updateR();
  }
}

void Model::updateArea() {
  cublasErrCheck(cublasSgemv(cudaTypes->handle, CUBLAS_OP_T, p.N * p.N,
                             numCells, d_alpha, d_phi, p.N * p.N, d_ones, 1,
                             d_beta, d_area, 1));
}

void Model::updateA_Area() {}

void Model::updatePhi(int t) {
#ifdef USE_CHEMO
  updatePhiKernelChemo<<<cudaTypes->grid, cudaTypes->block>>>(
      d_phi, d_A, d_R, d_C, d_Param, d_area, cudaTypes->d_grid_location,
      d_collision_force, d_phi_t, t);
#else
  updatePhiKernel<<<cudaTypes->grid, cudaTypes->block>>>(
      d_phi, d_A, d_R, d_Param, d_area, cudaTypes->d_grid_location,
      d_collision_force, d_phi_t, t);
#endif
  // updatePhiKernel<<<cudaTypes->grid, cudaTypes->block>>>(d_phi, d_A, d_R,
  // d_Param, d_area, d_grid_location, d_collision_force, d_phi_t, t);
  cudaErrCheck(cudaGetLastError());
  cublasErrCheck(cublasRaxpy(cudaTypes->handle, numCells * p.N * p.N,
                             &(d_Param->dt), d_phi_t, 1, d_phi, 1));
}

void Model::updateC() {
  updateCKernel<<<cudaTypes->Cblocks, cudaTypes->Cthreads_per_block>>>(
      d_C, d_Cbuffer, d_Param);
  cudaErrCheck(cudaGetLastError());
  cublasErrCheck(
      cublasRcopy(cudaTypes->handle, p.Nchem * p.Nchem, d_Cbuffer, 1, d_C, 1));
}

void Model::updateA() {
  updateAKernel<<<cudaTypes->grid, cudaTypes->block>>>(
      d_phi, d_A, d_R, d_Param, d_phi_t, cudaTypes->d_grid_location,
      cudaTypes->d_states, d_buffer);
  cudaErrCheck(cudaGetLastError());
  cublasErrCheck(cublasRaxpy(cudaTypes->handle, numCells * p.N * p.N,
                             &(d_Param->dt), d_buffer, 1, d_A, 1));
}

void Model::updateR() {
#ifdef USE_CHEMO
  updateRKernelChemo<<<cudaTypes->grid, cudaTypes->block>>>(
      d_phi, d_A, d_R, d_C, d_Param, d_phi_t, cudaTypes->d_grid_location,
      cudaTypes->d_Rstates, d_buffer);
#else
  updateRKernel<<<cudaTypes->grid, cudaTypes->block>>>(
      d_phi, d_A, d_R, d_Param, d_phi_t, cudaTypes->d_grid_location,
      cudaTypes->d_states, d_buffer);
#endif
  cudaErrCheck(cudaGetLastError());
  cublasErrCheck(cublasRaxpy(cudaTypes->handle, numCells * p.N * p.N,
                             &(d_Param->dt), d_buffer, 1, d_R, 1));
}

void Model::updateCollisionForce() {
  updateCollisionForceKernel<<<cudaTypes->grid, cudaTypes->block>>>(
      d_phi, cudaTypes->d_grid_location, d_Param, d_collision_force);
  cudaErrCheck(cudaGetLastError());
}

void Model::updateGridLocation() {
  getXIndexMaskedKernel<<<cudaTypes->grid, cudaTypes->block>>>(d_phi, d_buffer,
                                                               d_Param, 0.025);
  cudaErrCheck(cudaGetLastError());
  for (int i = 0; i < numCells; ++i) {
    cublasErrCheck(cublasIramax(cudaTypes->handle, p.N * p.N,
                                d_buffer + i * p.N * p.N, 1,
                                &(cudaTypes->d_grid_max + i)->x));
    cublasErrCheck(cublasIramin(cudaTypes->handle, p.N * p.N,
                                d_buffer + i * p.N * p.N, 1,
                                &(cudaTypes->d_grid_min + i)->x));
  }
  cudaErrCheck(cudaDeviceSynchronize());
  rollAllFieldsX();

  getYIndexMaskedKernel<<<cudaTypes->grid, cudaTypes->block>>>(d_phi, d_buffer,
                                                               d_Param, 0.025);
  cudaErrCheck(cudaGetLastError());
  for (int i = 0; i < numCells; ++i) {
    cublasErrCheck(cublasIramax(cudaTypes->handle, p.N * p.N,
                                d_buffer + i * p.N * p.N, 1,
                                &((cudaTypes->d_grid_max + i)->y)));
    cublasErrCheck(cublasIramin(cudaTypes->handle, p.N * p.N,
                                d_buffer + i * p.N * p.N, 1,
                                &((cudaTypes->d_grid_min + i)->y)));
  }
  cudaErrCheck(cudaDeviceSynchronize());
  rollAllFieldsY();
  cudaErrCheck(cudaDeviceSynchronize());

  updateGridLocationKernel<<<blocksNumCells, threadsNumCells>>>(
      cudaTypes->d_grid_min, cudaTypes->d_grid_location, d_Param);
  cudaErrCheck(cudaGetLastError());
  updateGridLocationKernel<<<blocksNumCells, threadsNumCells>>>(
      cudaTypes->d_grid_max, cudaTypes->d_grid_location, d_Param);
  cudaErrCheck(cudaGetLastError());
}

void Model::updateBoundingBox() {
  // Store the bounding box location for each cell inside d_grid_max and
  // d_grid_min
  getXIndexMaskedKernel<<<cudaTypes->grid, cudaTypes->block>>>(d_phi, d_buffer,
                                                               d_Param, 0.0025);
  for (int i = 0; i < numCells; ++i) {
    cublasErrCheck(cublasIramax(cudaTypes->handle, p.N * p.N,
                                d_buffer + i * p.N * p.N, 1,
                                &(cudaTypes->d_grid_max + i)->x));
    cublasErrCheck(cublasIramin(cudaTypes->handle, p.N * p.N,
                                d_buffer + i * p.N * p.N, 1,
                                &(cudaTypes->d_grid_min + i)->x));
  }

  getYIndexMaskedKernel<<<cudaTypes->grid, cudaTypes->block>>>(d_phi, d_buffer,
                                                               d_Param, 0.0025);
  for (int i = 0; i < numCells; ++i) {
    cublasErrCheck(cublasIramax(cudaTypes->handle, p.N * p.N,
                                d_buffer + i * p.N * p.N, 1,
                                &((cudaTypes->d_grid_max + i)->y)));
    cublasErrCheck(cublasIramin(cudaTypes->handle, p.N * p.N,
                                d_buffer + i * p.N * p.N, 1,
                                &((cudaTypes->d_grid_min + i)->y)));
  }

  updateBoundingIndex<<<blocksNumCells, threadsNumCells>>>(
      cudaTypes->d_grid_min, d_Param);
  updateBoundingIndex<<<blocksNumCells, threadsNumCells>>>(
      cudaTypes->d_grid_max, d_Param);
}

void Model::updateCOM() {
  // Calculate the center of mass of each cell within its own grid
  // Calculate the center of mass of each cell within its own grid
  createCOMXKernel<<<cudaTypes->grid, cudaTypes->block>>>(d_phi, d_buffer,
                                                          d_Param);
  cudaErrCheck(cudaGetLastError());

  // Commented out the loop and replaced with cublasSgemv
  // for (int i = 0; i < numCells; ++i) {
  //     cublasRasum(cudaTypes->handle, p.N*p.N, d_buffer + i*p.N*p.N, 1,
  //     &((cudaTypes->d_com + i)->x));
  // }
  cublasErrCheck(cublasSgemv(cudaTypes->handle, CUBLAS_OP_T, p.N * p.N,
                             numCells, d_alpha, d_buffer, p.N * p.N, d_ones, 1,
                             d_beta, d_tmp_com, 1));
  copyToX<<<blocksNumCells, threadsNumCells>>>(d_tmp_com, cudaTypes->d_com,
                                               numCells);

  createCOMYKernel<<<cudaTypes->grid, cudaTypes->block>>>(d_phi, d_buffer,
                                                          d_Param);
  cudaErrCheck(cudaGetLastError());

  // Commented out the loop and replaced with cublasSgemv
  // for (int i = 0; i < numCells; ++i) {
  //     cublasRasum(cudaTypes->handle, p.N*p.N, d_buffer + i*p.N*p.N, 1,
  //     &((cudaTypes->d_com + i)->y));
  // }
  cublasErrCheck(cublasSgemv(cudaTypes->handle, CUBLAS_OP_T, p.N * p.N,
                             numCells, d_alpha, d_buffer, p.N * p.N, d_ones, 1,
                             d_beta, d_tmp_com, 1));
  copyToY<<<blocksNumCells, threadsNumCells>>>(d_tmp_com, cudaTypes->d_com,
                                               numCells);

  multiplyKernel<<<cudaTypes->grid, cudaTypes->block>>>(d_phi, d_A, d_buffer,
                                                        d_Param);
  cudaErrCheck(cudaGetLastError());

  // Commented out the loop and replaced with cublasSgemv
  // for (int i = 0; i < numCells; ++i) {
  //     cublasRasum(cudaTypes->handle, p.N*p.N, d_buffer + i*p.N*p.N, 1,
  //     &d_Aarea[i]);
  // }
  cublasErrCheck(cublasSgemv(cudaTypes->handle, CUBLAS_OP_T, p.N * p.N,
                             numCells, d_alpha, d_buffer, p.N * p.N, d_ones, 1,
                             d_beta, d_Aarea, 1));

  createCOMAXKernel<<<cudaTypes->grid, cudaTypes->block>>>(d_phi, d_A, d_buffer,
                                                           d_Param);
  cudaErrCheck(cudaGetLastError());

  // Commented out the loop and replaced with cublasSgemv
  // for (int i = 0; i < numCells; ++i) {
  //     cublasRasum(cudaTypes->handle, p.N*p.N, d_buffer + i*p.N*p.N, 1,
  //     &((cudaTypes->d_comA + i)->x));
  // }
  cublasErrCheck(cublasSgemv(cudaTypes->handle, CUBLAS_OP_T, p.N * p.N,
                             numCells, d_alpha, d_buffer, p.N * p.N, d_ones, 1,
                             d_beta, d_tmp_com, 1));
  copyToX<<<blocksNumCells, threadsNumCells>>>(d_tmp_com, cudaTypes->d_comA,
                                               numCells);

  createCOMAYKernel<<<cudaTypes->grid, cudaTypes->block>>>(d_phi, d_A, d_buffer,
                                                           d_Param);
  cudaErrCheck(cudaGetLastError());

  // Commented out the loop and replaced with cublasSgemv
  // for (int i = 0; i < numCells; ++i) {
  //     cublasRasum(cudaTypes->handle, p.N*p.N, d_buffer + i*p.N*p.N, 1,
  //     &((cudaTypes->d_comA + i)->y));
  // }
  cublasErrCheck(cublasSgemv(cudaTypes->handle, CUBLAS_OP_T, p.N * p.N,
                             numCells, d_alpha, d_buffer, p.N * p.N, d_ones, 1,
                             d_beta, d_tmp_com, 1));
  copyToY<<<blocksNumCells, threadsNumCells>>>(d_tmp_com, cudaTypes->d_comA,
                                               numCells);

  // copy
}

void Model::rollAllFieldsX() {
  rollGridXKernel<<<cudaTypes->grid, cudaTypes->block>>>(
      d_phi, d_buffer, cudaTypes->d_grid_max, d_Param);
  cublasErrCheck(cublasRswap(cudaTypes->handle, numCells * p.N * p.N, d_buffer,
                             1, d_phi, 1));

  rollGridXKernel<<<cudaTypes->grid, cudaTypes->block>>>(
      d_phi, d_buffer, cudaTypes->d_grid_min, d_Param);
  cublasErrCheck(cublasRswap(cudaTypes->handle, numCells * p.N * p.N, d_buffer,
                             1, d_phi, 1));

  rollGridXKernel<<<cudaTypes->grid, cudaTypes->block>>>(
      d_A, d_buffer, cudaTypes->d_grid_max, d_Param);
  cublasErrCheck(cublasRswap(cudaTypes->handle, numCells * p.N * p.N, d_buffer,
                             1, d_A, 1));

  rollGridXKernel<<<cudaTypes->grid, cudaTypes->block>>>(
      d_A, d_buffer, cudaTypes->d_grid_min, d_Param);
  cublasErrCheck(cublasRswap(cudaTypes->handle, numCells * p.N * p.N, d_buffer,
                             1, d_A, 1));

  rollGridXKernel<<<cudaTypes->grid, cudaTypes->block>>>(
      d_R, d_buffer, cudaTypes->d_grid_max, d_Param);
  cublasErrCheck(cublasRswap(cudaTypes->handle, numCells * p.N * p.N, d_buffer,
                             1, d_R, 1));

  rollGridXKernel<<<cudaTypes->grid, cudaTypes->block>>>(
      d_R, d_buffer, cudaTypes->d_grid_min, d_Param);
  cublasErrCheck(cublasRswap(cudaTypes->handle, numCells * p.N * p.N, d_buffer,
                             1, d_R, 1));
}

void Model::rollAllFieldsY() {
  rollGridYKernel<<<cudaTypes->grid, cudaTypes->block>>>(
      d_phi, d_buffer, cudaTypes->d_grid_max, d_Param);
  cublasErrCheck(cublasRswap(cudaTypes->handle, numCells * p.N * p.N, d_buffer,
                             1, d_phi, 1));

  rollGridYKernel<<<cudaTypes->grid, cudaTypes->block>>>(
      d_phi, d_buffer, cudaTypes->d_grid_min, d_Param);
  cublasErrCheck(cublasRswap(cudaTypes->handle, numCells * p.N * p.N, d_buffer,
                             1, d_phi, 1));

  rollGridYKernel<<<cudaTypes->grid, cudaTypes->block>>>(
      d_A, d_buffer, cudaTypes->d_grid_max, d_Param);
  cublasErrCheck(cublasRswap(cudaTypes->handle, numCells * p.N * p.N, d_buffer,
                             1, d_A, 1));

  rollGridYKernel<<<cudaTypes->grid, cudaTypes->block>>>(
      d_A, d_buffer, cudaTypes->d_grid_min, d_Param);
  cublasErrCheck(cublasRswap(cudaTypes->handle, numCells * p.N * p.N, d_buffer,
                             1, d_A, 1));

  rollGridYKernel<<<cudaTypes->grid, cudaTypes->block>>>(
      d_R, d_buffer, cudaTypes->d_grid_max, d_Param);
  cublasErrCheck(cublasRswap(cudaTypes->handle, numCells * p.N * p.N, d_buffer,
                             1, d_R, 1));

  rollGridYKernel<<<cudaTypes->grid, cudaTypes->block>>>(
      d_R, d_buffer, cudaTypes->d_grid_min, d_Param);
  cublasErrCheck(cublasRswap(cudaTypes->handle, numCells * p.N * p.N, d_buffer,
                             1, d_R, 1));
}

WriteError Model::makeCondensedOutput(Real* field1,
                                      Real* field2,
                                      std::ofstream& outputField) {
  int2 min_grid_location;
  min_grid_location.x = cudaTypes->grid_location[0].x;
  min_grid_location.y = cudaTypes->grid_location[0].y;
  int2 max_grid_location;
  max_grid_location.x = cudaTypes->grid_location[0].x;
  max_grid_location.y = cudaTypes->grid_location[0].y;

  for (int i = 0; i < numCells; i++) {
    min_grid_location.x = cudaTypes->grid_location[i].x < min_grid_location.x
                              ? cudaTypes->grid_location[i].x
                              : min_grid_location.x;
    min_grid_location.y = cudaTypes->grid_location[i].y < min_grid_location.y
                              ? cudaTypes->grid_location[i].y
                              : min_grid_location.y;
    max_grid_location.x = cudaTypes->grid_location[i].x > max_grid_location.x
                              ? cudaTypes->grid_location[i].x
                              : max_grid_location.x;
    max_grid_location.y = cudaTypes->grid_location[i].y > max_grid_location.y
                              ? cudaTypes->grid_location[i].y
                              : max_grid_location.y;
  }

  int condensed_size_x = max_grid_location.x - min_grid_location.x + p.N;
  int condensed_size_y = max_grid_location.y - min_grid_location.y + p.N;

  if (condensed_size_x < 0 || condensed_size_y < 0) {
    std::cerr << "Error: condensed_size_x or condensed_size_y is negative"
              << "\n";
    return 1;
  }
  if (condensed_size_x > 10000 || condensed_size_y > 10000) {
    std::cerr << "Error: condensed_size_x or condensed_size_y is too large"
              << "\n";
    return 1;
  }

  Real* condensed_field = new Real[condensed_size_x * condensed_size_y];
  memset(condensed_field, 0,
         condensed_size_x * condensed_size_y * sizeof(Real));

  if (field2 == nullptr) {
    for (int k = 0; k < numCells; k++) {
      for (int i = 0; i < p.N; i++) {
        for (int j = 0; j < p.N; j++) {
          int xcoord = cudaTypes->grid_location[k].x + i - min_grid_location.x;
          int ycoord = cudaTypes->grid_location[k].y + j - min_grid_location.y;
          condensed_field[ycoord * condensed_size_x + xcoord] +=
              field1[k * p.N * p.N + j * p.N + i];
        }
      }
    }
  } else {
    for (int k = 0; k < numCells; k++) {
      for (int i = 0; i < p.N; i++) {
        for (int j = 0; j < p.N; j++) {
          int xcoord = cudaTypes->grid_location[k].x + i - min_grid_location.x;
          int ycoord = cudaTypes->grid_location[k].y + j - min_grid_location.y;
          condensed_field[ycoord * condensed_size_x + xcoord] +=
              field1[k * p.N * p.N + j * p.N + i] *
              field2[k * p.N * p.N + j * p.N + i];
        }
      }
    }
  }

  outputField.write(reinterpret_cast<char*>(&condensed_size_x), sizeof(int));
  outputField.write(reinterpret_cast<char*>(&condensed_size_y), sizeof(int));
  outputField.write(reinterpret_cast<char*>(condensed_field),
                    condensed_size_x * condensed_size_y * sizeof(Real));
  delete[] condensed_field;
  return 0;
}

WriteError Model::writeDynamics(int step) {
#ifdef USE_DEVICE_BACKLOG
  int threadsPerBlock = 256;
  int numBlocks = (numCells + threadsPerBlock - 1) / threadsPerBlock;
  static int backlog_step = 0;
  copyToBacklogKernel<<<numBlocks, threadsPerBlock>>>(
      d_output, cudaTypes->d_com, cudaTypes->d_comA, d_area, d_Aarea,
      cudaTypes->d_grid_location, numCells, step, backlog_step);
  cudaErrCheck(cudaGetLastError());

  if (backlog_step == BACKLOG_SIZE - 1) {
    cudaErrCheck(
        cudaMemcpy(h_output, d_output,
                   numCells * BACKLOG_SIZE * sizeof(Real) * NUM_OUTPUT_COLUMNS,
                   cudaMemcpyDeviceToHost));
    dynamics_output.write(
        reinterpret_cast<char*>(h_output),
        numCells * BACKLOG_SIZE * sizeof(Real) * NUM_OUTPUT_COLUMNS);
    // Error check for any nan values
    for (int i = 0; i < numCells * BACKLOG_SIZE * NUM_OUTPUT_COLUMNS; ++i) {
      if (std::isnan(h_output[i])) {
        std::cerr << "Error: Dynamics output" << i % NUM_OUTPUT_COLUMNS << ", "
                  << i / NUM_OUTPUT_COLUMNS << " is NaN" << "\n";
        throw std::runtime_error("Dynamics output is NaN");
      }
    }
  }
  backlog_step = (backlog_step + 1) % BACKLOG_SIZE;

  return 0;

#else
  cudaErrCheck(cudaMemcpy(cudaTypes->com, cudaTypes->d_com,
                          numCells * sizeof(Real2), cudaMemcpyDeviceToHost));
  cudaErrCheck(cudaMemcpy(cudaTypes->comA, cudaTypes->d_comA,
                          numCells * sizeof(Real2), cudaMemcpyDeviceToHost));
  cudaErrCheck(cudaMemcpy(area, d_area, numCells * sizeof(Real),
                          cudaMemcpyDeviceToHost));
  cudaErrCheck(cudaMemcpy(Aarea, d_Aarea, numCells * sizeof(Real),
                          cudaMemcpyDeviceToHost));
  cudaErrCheck(cudaDeviceSynchronize());
  for (int i = 0; i < numCells; ++i) {
    cudaTypes->com[i].x /= area[i];
    cudaTypes->com[i].y /= area[i];
    if (Aarea[i] < 1e-6) {
      cudaTypes->comA[i].x = 0;
      cudaTypes->comA[i].y = 0;
    } else {
      cudaTypes->comA[i].x /= Aarea[i];
      cudaTypes->comA[i].y /= Aarea[i];
    }

    if (std::isnan(cudaTypes->com[i].x) || std::isnan(cudaTypes->com[i].y)) {
      std::cerr << "Error: COM" << i << " is NaN" << "\n";
      std::cerr << "Area: " << area[i] << "\n";
      std::cerr << "COM: " << cudaTypes->com[i].x << ", " << cudaTypes->com[i].y
                << "\n";
      throw std::runtime_error("COM is NaN");
    }
    if (std::isnan(cudaTypes->comA[i].x) || std::isnan(cudaTypes->comA[i].y)) {
      std::cerr << "Error: COMA is NaN" << "\n";
      std::cerr << "Area: " << Aarea[i] << "\n";
      std::cerr << "COMA: " << cudaTypes->comA[i].x << ", "
                << cudaTypes->comA[i].y << "\n";
      throw std::runtime_error("COMA is NaN");
    }
  }

  cudaErrCheck(cudaMemcpy(cudaTypes->grid_location, cudaTypes->d_grid_location,
                          numCells * sizeof(int2), cudaMemcpyDeviceToHost));
  Real real_i, real_step, real_x, real_y;
  for (int i = 0; i < numCells; ++i) {
    real_i = (Real)i;
    real_step = (Real)step;
    real_x = (Real)cudaTypes->grid_location[i].x;
    real_y = (Real)cudaTypes->grid_location[i].y;
    dynamics_output.write(reinterpret_cast<char*>(&real_i), sizeof(Real));
    dynamics_output.write(reinterpret_cast<char*>(&real_step), sizeof(Real));
    dynamics_output.write(reinterpret_cast<char*>(&real_x), sizeof(Real));
    dynamics_output.write(reinterpret_cast<char*>(&real_y), sizeof(Real));
    dynamics_output.write(reinterpret_cast<char*>(&area[i]), sizeof(Real));
    dynamics_output.write(reinterpret_cast<char*>(&Aarea[i]), sizeof(Real));
    dynamics_output.write(reinterpret_cast<char*>(&cudaTypes->com[i].x),
                          sizeof(Real));
    dynamics_output.write(reinterpret_cast<char*>(&cudaTypes->com[i].y),
                          sizeof(Real));
    dynamics_output.write(reinterpret_cast<char*>(&cudaTypes->comA[i].x),
                          sizeof(Real));
    dynamics_output.write(reinterpret_cast<char*>(&cudaTypes->comA[i].y),
                          sizeof(Real));
  }
  return 0;
#endif
}

WriteError Model::writeFullOutput() {
  // Write phi and A to file
  // copy the grid location to the host
  cudaErrCheck(cudaMemcpy(cudaTypes->grid_location, cudaTypes->d_grid_location,
                          numCells * sizeof(int2), cudaMemcpyDeviceToHost));
  std::lock_guard<std::mutex> guard(fileMutex);  // Lock the mutex
  cudaErrCheck(cudaMemcpy(phi, d_phi, numCells * p.N * p.N * sizeof(Real),
                          cudaMemcpyDeviceToHost));
  cudaErrCheck(cudaMemcpy(A, d_A, numCells * p.N * p.N * sizeof(Real),
                          cudaMemcpyDeviceToHost));

  if (writeCondensedOutput_) {
    WriteError earlyExit = makeCondensedOutput(phi, nullptr, phi_output);
    WriteError earlyExitA = makeCondensedOutput(A, phi, A_output);
    makeLabelImage<<<cudaTypes->grid, cudaTypes->block>>>(
        d_phi, d_buffer, cudaTypes->d_grid_location, d_Param);
    cudaErrCheck(cudaMemcpy(phi, d_buffer, numCells * p.N * p.N * sizeof(Real),
                            cudaMemcpyDeviceToHost));
    WriteError earlyExitLabel = makeCondensedOutput(phi, nullptr, label_output);
    if (earlyExit || earlyExitA || earlyExitLabel) {
      return 1;
    }
  } else {
    phi_output.write(reinterpret_cast<char*>(phi),
                     numCells * p.N * p.N * sizeof(Real));
    A_output.write(reinterpret_cast<char*>(A),
                   numCells * p.N * p.N * sizeof(Real));
  }

  // Write the concentration field to file
#ifdef USE_CHEMO
  cudaErrCheck(cudaMemcpy(C, d_C, p.Nchem * p.Nchem * sizeof(Real),
                          cudaMemcpyDeviceToHost));
  C_output.write(reinterpret_cast<char*>(C), p.Nchem * p.Nchem * sizeof(Real));
#endif

  return 0;
}

// Function to run the file writing operation in a background thread
void Model::writeInBackground(bool useCondensedOutput) {
  std::thread backgroundThread(&Model::writeFullOutput, this);
  backgroundThread.detach();  // Detach the thread to run in the background
}

std::vector<Real> Model::getCOMX() {
  std::vector<Real> comX(numCells);
  cudaErrCheck(cudaMemcpy(comX.data(), cudaTypes->d_com,
                          numCells * sizeof(Real2), cudaMemcpyDeviceToHost));
  for (int i = 0; i < numCells; ++i) {
    comX[i] = cudaTypes->com[i].x;
  }
  return comX;
}

std::vector<Real> Model::getCOMY() {
  std::vector<Real> comY(numCells);
  cudaErrCheck(cudaMemcpy(comY.data(), cudaTypes->d_com,
                          numCells * sizeof(Real2), cudaMemcpyDeviceToHost));
  for (int i = 0; i < numCells; ++i) {
    comY[i] = cudaTypes->com[i].y;
  }
  return comY;
}

std::vector<Real> Model::getCOMAX() {
  std::vector<Real> comAX(numCells);
  cudaErrCheck(cudaMemcpy(comAX.data(), cudaTypes->d_comA,
                          numCells * sizeof(Real2), cudaMemcpyDeviceToHost));
  for (int i = 0; i < numCells; ++i) {
    comAX[i] = cudaTypes->comA[i].x;
  }
  return comAX;
}

std::vector<Real> Model::getCOMAY() {
  std::vector<Real> comAY(numCells);
  cudaErrCheck(cudaMemcpy(comAY.data(), cudaTypes->d_comA,
                          numCells * sizeof(Real2), cudaMemcpyDeviceToHost));
  for (int i = 0; i < numCells; ++i) {
    comAY[i] = cudaTypes->comA[i].y;
  }
  return comAY;
}

std::vector<Real> Model::getArea() {
  std::vector<Real> areaVec(numCells);
  cudaErrCheck(cudaMemcpy(areaVec.data(), d_area, numCells * sizeof(Real),
                          cudaMemcpyDeviceToHost));
  return areaVec;
}
