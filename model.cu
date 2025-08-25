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
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <math.h>
#include <cstdlib>

#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>

#include "dependencies/indicators.hpp"
#include "include/error_check.cuh"
#include "include/kernels.cuh"
#include "include/model.cuh"
#include "include/model.h"
#include "include/params.hpp"

#define USE_FLOAT
#define SKIP_FRAMES_FULL_OUTPUT 800

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
#define BACKLOG_SIZE 10

Real initializeParamArray(Param p,
                          Param* p_array,
                          const std::string& boundaryType) {
  // Initialize the parameter array with the same parameters but different
  // radius evenly distributed between r0 and r0 + delta_r Parameters: p: Param
  // struct with the parameters p_array: Array of Param structs to be
  // initialized Return: the total radius of the population for the grid
  // location initialization
  srand(0);
  Real total_area = 0;
  Real populationRadius = 0;
  for (int i = 0; i < p.maxCells; ++i) {
    p_array[i] = p;
    p_array[i].r0 = p.r0 + (p.delta_r * rand()) / (RAND_MAX + 1.0);
    p_array[i].division_time *= 0.9 + 0.2 * rand() / (RAND_MAX + 1.0);
    p_array[i].S0 = M_PI * p_array[i].r0 * p_array[i].r0;
    if (i < p.numCellsInitial)
      total_area += p_array[i].S0;
  }
  for (int i = 0; i < p.maxCells; ++i) {
    if (p.R_wall == 0) {
      if (boundaryType == "circular") {
        p_array[i].R_wall = sqrt(total_area / (M_PI * p.packing_fraction));
      } else if (boundaryType == "square") {
        p_array[i].R_wall = sqrt(total_area / (p.packing_fraction)) / 2;
      }
    }
    p_array[i].Nchem = 2 * p_array[i].R_wall / p_array[i].dx + N;
  }
  if (boundaryType == "circular") {
    populationRadius = sqrt(total_area / 4 / (M_PI * p.packing_fraction));
  } else if (boundaryType == "square") {
    populationRadius = sqrt(total_area / (p.packing_fraction)) / 2;
  }

  return populationRadius;
}

// define the circular boundary condition function to get access for host code
__device__ BoundaryConditionFunc d_circular_boundary =
    circular_boundary_condition;
__device__ BoundaryConditionFunc d_square_boundary = square_boundary_condition;

void Model::setBoundaryConditionFunc(const std::string& type) {
  if (type == "circular") {
    cudaMemcpyFromSymbol(&boundary_condition_func_, d_circular_boundary,
                         sizeof(BoundaryConditionFunc));
  } else if (type == "square") {
    cudaMemcpyFromSymbol(&boundary_condition_func_, d_square_boundary,
                         sizeof(BoundaryConditionFunc));
  } else {
    boundary_condition_func_ = nullptr;
  }
}

Model::Model(Param param,
             int deviceID,
             std::string basePath,
             unsigned long long seed,
             std::string simulationID,
             int outputType,
             std::string boundaryType) {
  cudaTypes = new CudaTypes;

  p = param;
  p.S0 = M_PI * p.r0 * p.r0;
  p.gridN = N;
  currentNumCells = p.numCellsInitial;
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
  std::cout << "Using device: " << deviceID_ << "\n";
  rngSeed = seed;
  std::cout << "Using RNG seed: " << rngSeed << "\n";

  cudaTypes->block = dim3(blockSizeX, blockSizeY, blockSizeZ);
  cudaTypes->grid =
      dim3((N + blockSizeX - 1) / blockSizeX, (N + blockSizeY - 1) / blockSizeY,
           (currentNumCells + blockSizeZ - 1) / blockSizeZ);

  paramArray = new Param[p.maxCells];
  Real populationRadius = initializeParamArray(p, paramArray, boundaryType);
  p.R_wall = populationRadius;
#ifdef USE_CHEMO
  p.gridNchem = paramArray[0].gridNchem;
#else
  p.Nchem = 0;
#endif
  cudaTypes->Cblocks = dim3(p.Nchem / 32 + 1, p.Nchem / 32 + 1);
  cudaTypes->Cthreads_per_block = dim3(32, 32);
  // std::cout << "Nchem: " << p.Nchem << "\n";
  cudaErrCheck(cudaMalloc((void**)&d_Param, sizeof(Param) * p.maxCells));
  cudaErrCheck(cudaMemcpy(d_Param, paramArray, sizeof(Param) * p.maxCells,
                          cudaMemcpyHostToDevice));

  // Initialize device and host pointers
  initializeDeviceMemory();
  initializeHostMemory();

  // Helper variables for cublas
  Real* ones = new Real[N * N];
  for (int i = 0; i < N * N; ++i) {
    ones[i] = 1;
  }
  cudaErrCheck(cudaMalloc((void**)&d_ones, N * N * sizeof(Real)));
  cudaErrCheck(
      cudaMemcpy(d_ones, ones, N * N * sizeof(Real), cudaMemcpyHostToDevice));

  Real *zero, *one;
  zero = new Real;
  one = new Real;
  *zero = 0;
  *one = 1;
  cudaErrCheck(cudaMalloc((void**)&d_zero, sizeof(Real)));
  cudaErrCheck(cudaMalloc((void**)&d_one, sizeof(Real)));
  cudaErrCheck(cudaMemcpy(d_zero, zero, sizeof(Real), cudaMemcpyHostToDevice));
  cudaErrCheck(cudaMemcpy(d_one, one, sizeof(Real), cudaMemcpyHostToDevice));

  // Initialize phi
  initializePhiKernel<<<cudaTypes->grid, cudaTypes->block>>>(d_phi, d_Param);
  cudaErrCheck(cudaGetLastError());

  // Initialize the grid location
  initializeCellGrid(populationRadius, boundaryType);
  cudaErrCheck(cudaMemcpy(cudaTypes->d_grid_location, cudaTypes->grid_location,
                          currentNumCells * sizeof(int2),
                          cudaMemcpyHostToDevice));

  // Initialize cublas
  cublasErrCheck(cublasCreate(&(cudaTypes->handle)));
  cublasErrCheck(
      cublasSetPointerMode(cudaTypes->handle, CUBLAS_POINTER_MODE_DEVICE));

  // Initialize curand states
  cudaErrCheck(cudaMalloc((void**)&cudaTypes->d_states,
                          p.maxCells * N * N * sizeof(curandState)));
  // cudaErrCheck(cudaMalloc((void**)&d_Rstates,
  // currentNumCells*N*N*sizeof(curandState)));

  dim3 gridMax =
      dim3((N + blockSizeX - 1) / blockSizeX, (N + blockSizeY - 1) / blockSizeY,
           (p.maxCells + blockSizeZ - 1) / blockSizeZ);
  setupCurandStates<<<gridMax, cudaTypes->block>>>(cudaTypes->d_states, rngSeed,
                                                   0, d_Param);
  // cudaErrCheck(cudaGetLastError());
  // setupCurandStates<<<cudaTypes->grid, cudaTypes->block>>>(d_Rstates,
  // rngSeed, currentNumCells*N*N, d_Param);
  cudaErrCheck(cudaGetLastError());
  curandErrCheck(cudaGetLastError());

#ifdef USE_DEVICE_BACKLOG
  cudaErrCheck(cudaMalloc(
      (void**)&d_output,
      currentNumCells * BACKLOG_SIZE * sizeof(Real) * NUM_OUTPUT_COLUMNS));
  cudaErrCheck(cudaMallocHost(
      (void**)&h_output,
      currentNumCells * BACKLOG_SIZE * sizeof(Real) * NUM_OUTPUT_COLUMNS));
#endif
  cudaMalloc((void**)&d_tmp_com, currentNumCells * sizeof(Real));

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
    writeParamToTOML(p, basePath + "param_" + time_str + ".toml");
  }

  threadsNumCells = std::min(currentNumCells, 1024);
  blocksNumCells = (currentNumCells + threadsNumCells - 1) / threadsNumCells;

  // Set boundary condition function pointer based on boundaryType argument
  setBoundaryConditionFunc(boundaryType);
  cudaErrCheck(cudaGetLastError());
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
#ifdef USE_CHEMO
  cudaErrCheck(cudaFreeHost(C));
#endif

  cublasErrCheck(cublasDestroy(cudaTypes->handle));

  // Close the output files after mutex lock
  std::lock_guard<std::mutex> guard(fileMutex);
  if (writeFullOutput_) {
    phi_output.close();
    A_output.close();
    label_output.close();
  }
  if (writeDynamicsOutput_) {
    dynamics_output.close();
  }
#ifdef USE_CHEMO
  C_output.close();
#endif
  delete cudaTypes;
}

void Model::initializeHostMemory() {
  cudaErrCheck(cudaMallocHost((void**)&phi, p.maxCells * N * N * sizeof(Real)));
  cudaErrCheck(cudaMallocHost((void**)&A, p.maxCells * N * N * sizeof(Real)));
  cudaErrCheck(cudaMallocHost((void**)&R, p.maxCells * N * N * sizeof(Real)));
  cudaErrCheck(cudaMallocHost((void**)&area, p.maxCells * sizeof(Real)));
  cudaErrCheck(
      cudaMallocHost((void**)&new_cell_indices, p.maxCells * sizeof(int)));
  cudaErrCheck(
      cudaMallocHost((void**)&time_since_division, p.maxCells * sizeof(Real)));
  cudaErrCheck(cudaMallocHost((void**)&Aarea, p.maxCells * sizeof(Real)));
  cudaErrCheck(cudaMallocHost((void**)&cudaTypes->grid_location,
                              p.maxCells * sizeof(int2)));
  cudaErrCheck(
      cudaMallocHost((void**)&cudaTypes->com, p.maxCells * sizeof(Real2)));
  cudaErrCheck(
      cudaMallocHost((void**)&cudaTypes->comA, p.maxCells * sizeof(Real2)));
  cudaErrCheck(cudaMallocHost((void**)&cudaTypes->division_axes,
                              p.maxCells * sizeof(Real2)));
  memset(new_cell_indices, 0, p.maxCells * sizeof(int));
#ifdef USE_CHEMO
  cudaErrCheck(cudaMallocHost((void**)&C, p.Nchem * p.Nchem * sizeof(Real)));
#endif
}

void Model::initializeDeviceMemory() {
  // Allocate memory on device
  cudaErrCheck(cudaMalloc((void**)&d_phi, p.maxCells * N * N * sizeof(Real)));
  cudaErrCheck(cudaMalloc((void**)&d_A, p.maxCells * N * N * sizeof(Real)));
  cudaErrCheck(cudaMalloc((void**)&d_R, p.maxCells * N * N * sizeof(Real)));
  cudaErrCheck(cudaMalloc((void**)&d_area, p.maxCells * sizeof(Real)));
  cudaErrCheck(cudaMalloc((void**)&d_Aarea, p.maxCells * sizeof(Real)));
  cudaErrCheck(
      cudaMalloc((void**)&d_new_cell_indices, p.maxCells * sizeof(int)));
  cudaErrCheck(
      cudaMalloc((void**)&d_time_since_division, p.maxCells * sizeof(int)));
  cudaErrCheck(cudaMalloc((void**)&d_collision_force,
                          p.maxCells * N * N * sizeof(Real)));
  cudaErrCheck(
      cudaMalloc((void**)&d_buffer, p.maxCells * N * N * sizeof(Real)));
  cudaErrCheck(cudaMalloc((void**)&d_phi_t, p.maxCells * N * N * sizeof(Real)));
  cudaErrCheck(
      cudaMalloc((void**)&cudaTypes->d_grid_min, p.maxCells * sizeof(int2)));
  cudaErrCheck(
      cudaMalloc((void**)&cudaTypes->d_grid_max, p.maxCells * sizeof(int2)));
  cudaErrCheck(cudaMalloc((void**)&cudaTypes->d_grid_location,
                          p.maxCells * sizeof(int2)));
  cudaErrCheck(
      cudaMalloc((void**)&cudaTypes->d_com, p.maxCells * sizeof(Real2)));
  cudaErrCheck(cudaMalloc((void**)&cudaTypes->d_division_axes,
                          p.maxCells * sizeof(Real2)));
  cudaErrCheck(
      cudaMalloc((void**)&cudaTypes->d_comA, p.maxCells * sizeof(Real2)));
#ifdef USE_CHEMO
  cudaErrCheck(cudaMalloc((void**)&d_C, p.Nchem * p.Nchem * sizeof(Real)));
  cudaErrCheck(
      cudaMalloc((void**)&d_Cbuffer, p.Nchem * p.Nchem * sizeof(Real)));
  cudaErrCheck(cudaMemset(d_C, 0, p.Nchem * p.Nchem * sizeof(Real)));
  cudaErrCheck(cudaMemset(d_Cbuffer, 0, p.Nchem * p.Nchem * sizeof(Real)));
#endif

  // Initialize memory on device
  cudaErrCheck(cudaMemset(d_phi, 0, p.maxCells * N * N * sizeof(Real)));
  cudaErrCheck(cudaMemset(d_A, 0, p.maxCells * N * N * sizeof(Real)));
  cudaErrCheck(cudaMemset(d_R, 0, p.maxCells * N * N * sizeof(Real)));
  cudaErrCheck(cudaMemset(d_area, 0, p.maxCells * sizeof(Real)));
  cudaErrCheck(cudaMemset(d_time_since_division, 0, p.maxCells * sizeof(Real)));
  cudaErrCheck(cudaMemset(d_Aarea, 0, p.maxCells * sizeof(Real)));
  cudaErrCheck(
      cudaMemset(d_collision_force, 0, p.maxCells * N * N * sizeof(Real)));
  cudaErrCheck(cudaMemset(d_buffer, 0, p.maxCells * N * N * sizeof(Real)));
  cudaErrCheck(cudaMemset(d_phi_t, 0, p.maxCells * N * N * sizeof(Real)));
  cudaErrCheck(cudaMemset(cudaTypes->d_com, 0, p.maxCells * sizeof(Real2)));
  cudaErrCheck(cudaMemset(cudaTypes->d_comA, 0, p.maxCells * sizeof(Real2)));
}

void Model::initializeCellGrid(Real populationRadius,
                               const std::string& boundaryType) {
  if (boundaryType == "circular") {
    Real phi = (1 + sqrt(5)) / 2;
    int b = round(2 * sqrt(currentNumCells));
    for (int i = 0; i < currentNumCells; ++i) {
      int k = i + 1;
      Real r;
      if (k > currentNumCells - b)
        r = 1;
      else
        r = sqrt(1.0 * k - 0.5) /
            sqrt(1.0 * currentNumCells - (1.0 * b + 1) / 2);
      Real theta = 2 * M_PI * k / (phi * phi);
      Real rCircle = populationRadius * r;
      cudaTypes->grid_location[i].x = rCircle / p.dx * cos(theta);
      cudaTypes->grid_location[i].y = rCircle / p.dx * sin(theta);
    }
  } else if (boundaryType == "square") {
    int numRows = sqrt(currentNumCells);
    int numCols = currentNumCells / numRows + 1;
    for (int i = 0; i < numRows; ++i) {
      for (int j = 0; j < numCols; ++j) {
        int index = i * sqrt(currentNumCells) + j;
        if (index < currentNumCells) {
          Real x = -populationRadius + (2 * populationRadius / numRows) * i;
          Real y = -populationRadius + (2 * populationRadius / numCols) * j;
          cudaTypes->grid_location[index].x = x / p.dx;
          cudaTypes->grid_location[index].y = y / p.dx;
        }
      }
    }
  }
}

void Model::initializeOutput(std::string basePath, std::string time_str) {
  // First initialize the files using file descriptors from fcntl.h
  // This is to preallocate the files to avoid fragmentation

  Real preAllocSize =
      4.0 * p.T / 400. * p.R_wall * p.R_wall / p.dx / p.dx * sizeof(Real);
  phi_filename = basePath + "phi_" + time_str + ".dat";
  // preallocateOutput(phi_filename, (size_t)preAllocSize);
  phi_output.open(phi_filename, std::ios::binary | std::ios::out);

  A_filename = basePath + "A_" + time_str + ".dat";
  // preallocateOutput(A_filename, (size_t)preAllocSize);
  A_output.open(A_filename, std::ios::binary | std::ios::out);

  label_filename = basePath + "label_" + time_str + ".dat";
  // preallocateOutput(label_filename, (size_t)preAllocSize);
  label_output.open(label_filename, std::ios::binary | std::ios::out);

#ifdef USE_CHEMO
  std::string C_filename = basePath + "C_" + time_str + ".dat";
  preallocateOutput(C_filename, p.Nchem * p.Nchem * sizeof(Real) * p.T / 400);
  C_output.open(C_filename, std::ios::binary);
#endif
}

void Model::initializeDynamicsOutput(std::string basePath,
                                     std::string time_str) {
  std::string dynamics_filename = basePath + "dynamics_" + time_str + ".dat";
  // preallocateOutput(dynamics_filename, currentNumCells * sizeof(Real) *
  //                                          NUM_OUTPUT_COLUMNS * p.T /
  //                                          p.blind);
  dynamics_output.open(dynamics_filename, std::ios::binary | std::ios::out);
  // rewind the file to the beginning

  // dynamics_output <<
  // "t,cell,grid_offset_x,grid_offset_y,phi_area,A_area,com_x,com_y,comA_x,comA_y"
  // << "\n"; write the number of columns and byte size of columns in the
  // dynamics output file
  float numColumns = 10;
  float realWidth = sizeof(Real);
  dynamics_output.write(reinterpret_cast<char*>(&numColumns), sizeof(float));
  dynamics_output.write(reinterpret_cast<char*>(&realWidth), sizeof(float));
}

void Model::preallocateOutput(std::string filePath, size_t fileSize) {
  // Preallocate the output file to avoid fragmentation
  // THIS ONLY WORKS ON LINUX
  // TODO: Add support for other operating systems
  int fd = open(filePath.c_str(), O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
  if (fd == -1) {
    deleteOutput();
    std::cerr << "Error opening file: " << filePath << std::endl;
    return;
  }
  if (fallocate(fd, 0, 0, fileSize) == -1) {
    deleteOutput();
    std::cerr << "Error preallocating file: " << filePath << std::endl;
    close(fd);
    return;
  }
  close(fd);
}

void Model::deleteOutput() {
  // Delete the output files from the disk
  // this only gets called if there is an error in the allocation
  // So if the file is not open, we don't need to delete it
  if (phi_output.is_open()) {
    phi_output.close();
  }
  if (A_output.is_open()) {
    A_output.close();
  }
  if (label_output.is_open()) {
    label_output.close();
  }
  std::remove(phi_filename.c_str());
  std::remove(A_filename.c_str());
  std::remove(label_filename.c_str());
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

  cudaErrCheck(cudaDeviceSynchronize());
  for (int t = 0; t < startupSteps; t++) {
    // Early stages to stabilize the system
    updateArea();
    updateCollisionForce();
    updatePhi(0);
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

    // Hardcode the full output to every
    if (t % SKIP_FRAMES_FULL_OUTPUT == 0 && writeFullOutput_) {
      writeFullOutput();
    }

    updateCollisionForce();
    updatePhi(t);
#ifdef USE_CHEMO
    updateC();
#endif
    updateA(0);
    updateR();
    if (currentNumCells < p.maxCells)
      updateCellDivisions();
  }
}

void Model::updateArea() {
  cublasErrCheck(cublasSgemv(cudaTypes->handle, CUBLAS_OP_T, N * N,
                             currentNumCells, d_one, d_phi, N * N, d_ones, 1,
                             d_zero, d_area, 1));
}

void Model::updateA_Area() {}

void Model::updatePhi(int t) {
#ifdef USE_CHEMO
  updatePhiKernelChemo<<<cudaTypes->grid, cudaTypes->block>>>(
      d_phi, d_A, d_R, d_C, d_Param, d_area, cudaTypes->d_grid_location,
      d_collision_force, d_phi_t, t, currentNumCells);
#else
  updatePhiKernel<<<cudaTypes->grid, cudaTypes->block>>>(
      d_phi, d_A, d_R, d_Param, d_area, cudaTypes->d_grid_location,
      d_collision_force, d_phi_t, d_time_since_division, t, currentNumCells,
      boundary_condition_func_);
#endif
  cudaErrCheck(cudaGetLastError());
  cublasErrCheck(cublasRaxpy(cudaTypes->handle, currentNumCells * N * N,
                             &(d_Param->dt), d_phi_t, 1, d_phi, 1));
}

void Model::updateC() {
  updateCKernel<<<cudaTypes->Cblocks, cudaTypes->Cthreads_per_block>>>(
      d_C, d_Cbuffer, d_Param);
  cudaErrCheck(cudaGetLastError());
  cublasErrCheck(
      cublasRcopy(cudaTypes->handle, p.Nchem * p.Nchem, d_Cbuffer, 1, d_C, 1));
}

void Model::updateA(int t) {
  updateAKernel<<<cudaTypes->grid, cudaTypes->block>>>(
      d_phi, d_A, d_R, d_Param, d_phi_t, cudaTypes->d_grid_location,
      cudaTypes->d_states, d_buffer, d_time_since_division, t, currentNumCells);
  cudaErrCheck(cudaGetLastError());
  cublasErrCheck(cublasRaxpy(cudaTypes->handle, currentNumCells * N * N,
                             &(d_Param->dt), d_buffer, 1, d_A, 1));
}

void Model::updateR() {
#ifdef USE_CHEMO
  updateRKernelChemo<<<cudaTypes->grid, cudaTypes->block>>>(
      d_phi, d_A, d_R, d_C, d_Param, d_phi_t, cudaTypes->d_grid_location,
      cudaTypes->d_Rstates, d_buffer, currentNumCells);
#else
  updateRKernel<<<cudaTypes->grid, cudaTypes->block>>>(
      d_phi, d_A, d_R, d_Param, d_phi_t, cudaTypes->d_grid_location,
      cudaTypes->d_states, d_buffer, currentNumCells);
#endif
  cudaErrCheck(cudaGetLastError());
  cublasErrCheck(cublasRaxpy(cudaTypes->handle, currentNumCells * N * N,
                             &(d_Param->dt), d_buffer, 1, d_R, 1));
}

void Model::updateCollisionForce() {
  updateCollisionForceKernel<<<cudaTypes->grid, cudaTypes->block>>>(
      d_phi, cudaTypes->d_grid_location, d_Param, d_collision_force,
      currentNumCells);
  cudaErrCheck(cudaGetLastError());
}

void Model::updateGridLocation() {
  getXIndexMaskedKernel<<<cudaTypes->grid, cudaTypes->block>>>(
      d_phi, d_buffer, d_Param, 0.025, currentNumCells);
  cudaErrCheck(cudaGetLastError());
  for (int i = 0; i < currentNumCells; ++i) {
    cublasErrCheck(cublasIramax(cudaTypes->handle, N * N, d_buffer + i * N * N,
                                1, &(cudaTypes->d_grid_max + i)->x));
    cublasErrCheck(cublasIramin(cudaTypes->handle, N * N, d_buffer + i * N * N,
                                1, &(cudaTypes->d_grid_min + i)->x));
  }
  cudaErrCheck(cudaDeviceSynchronize());
  rollAllFieldsX();

  getYIndexMaskedKernel<<<cudaTypes->grid, cudaTypes->block>>>(
      d_phi, d_buffer, d_Param, 0.025, currentNumCells);
  cudaErrCheck(cudaGetLastError());
  for (int i = 0; i < currentNumCells; ++i) {
    cublasErrCheck(cublasIramax(cudaTypes->handle, N * N, d_buffer + i * N * N,
                                1, &((cudaTypes->d_grid_max + i)->y)));
    cublasErrCheck(cublasIramin(cudaTypes->handle, N * N, d_buffer + i * N * N,
                                1, &((cudaTypes->d_grid_min + i)->y)));
  }
  cudaErrCheck(cudaDeviceSynchronize());
  rollAllFieldsY();
  cudaErrCheck(cudaDeviceSynchronize());

  updateGridLocationKernel<<<blocksNumCells, threadsNumCells>>>(
      cudaTypes->d_grid_min, cudaTypes->d_grid_location, d_Param,
      currentNumCells);
  cudaErrCheck(cudaGetLastError());
  updateGridLocationKernel<<<blocksNumCells, threadsNumCells>>>(
      cudaTypes->d_grid_max, cudaTypes->d_grid_location, d_Param,
      currentNumCells);
  cudaErrCheck(cudaGetLastError());
}

void Model::updateBoundingBox() {
  getXIndexMaskedKernel<<<cudaTypes->grid, cudaTypes->block>>>(
      d_phi, d_buffer, d_Param, 0.0025, currentNumCells);
  for (int i = 0; i < currentNumCells; ++i) {
    cublasErrCheck(cublasIramax(cudaTypes->handle, N * N, d_buffer + i * N * N,
                                1, &(cudaTypes->d_grid_max + i)->x));
    cublasErrCheck(cublasIramin(cudaTypes->handle, N * N, d_buffer + i * N * N,
                                1, &(cudaTypes->d_grid_min + i)->x));
  }

  getYIndexMaskedKernel<<<cudaTypes->grid, cudaTypes->block>>>(
      d_phi, d_buffer, d_Param, 0.0025, currentNumCells);
  for (int i = 0; i < currentNumCells; ++i) {
    cublasErrCheck(cublasIramax(cudaTypes->handle, N * N, d_buffer + i * N * N,
                                1, &((cudaTypes->d_grid_max + i)->y)));
    cublasErrCheck(cublasIramin(cudaTypes->handle, N * N, d_buffer + i * N * N,
                                1, &((cudaTypes->d_grid_min + i)->y)));
  }

  /*
  updateBoundingIndex<<<blocksNumCells, threadsNumCells>>>(
      cudaTypes->d_grid_min, d_Param);
  updateBoundingIndex<<<blocksNumCells, threadsNumCells>>>(
      cudaTypes->d_grid_max, d_Param);
    */
}

void Model::updateCOM() {
  createCOMXKernel<<<cudaTypes->grid, cudaTypes->block>>>(
      d_phi, d_buffer, d_Param, currentNumCells);
  cudaErrCheck(cudaGetLastError());
  cublasErrCheck(cublasSgemv(cudaTypes->handle, CUBLAS_OP_T, N * N,
                             currentNumCells, d_one, d_buffer, N * N, d_ones, 1,
                             d_zero, d_tmp_com, 1));
  copyToX<<<blocksNumCells, threadsNumCells>>>(d_tmp_com, cudaTypes->d_com,
                                               currentNumCells);

  createCOMYKernel<<<cudaTypes->grid, cudaTypes->block>>>(
      d_phi, d_buffer, d_Param, currentNumCells);
  cudaErrCheck(cudaGetLastError());
  cublasErrCheck(cublasSgemv(cudaTypes->handle, CUBLAS_OP_T, N * N,
                             currentNumCells, d_one, d_buffer, N * N, d_ones, 1,
                             d_zero, d_tmp_com, 1));
  copyToY<<<blocksNumCells, threadsNumCells>>>(d_tmp_com, cudaTypes->d_com,
                                               currentNumCells);

  multiplyKernel<<<cudaTypes->grid, cudaTypes->block>>>(
      d_phi, d_A, d_buffer, d_Param, currentNumCells);
  cudaErrCheck(cudaGetLastError());
  cublasErrCheck(cublasSgemv(cudaTypes->handle, CUBLAS_OP_T, N * N,
                             currentNumCells, d_one, d_buffer, N * N, d_ones, 1,
                             d_zero, d_Aarea, 1));

  createCOMAXKernel<<<cudaTypes->grid, cudaTypes->block>>>(
      d_phi, d_A, d_buffer, d_Param, currentNumCells);
  cudaErrCheck(cudaGetLastError());
  cublasErrCheck(cublasSgemv(cudaTypes->handle, CUBLAS_OP_T, N * N,
                             currentNumCells, d_one, d_buffer, N * N, d_ones, 1,
                             d_zero, d_tmp_com, 1));
  copyToX<<<blocksNumCells, threadsNumCells>>>(d_tmp_com, cudaTypes->d_comA,
                                               currentNumCells);

  createCOMAYKernel<<<cudaTypes->grid, cudaTypes->block>>>(
      d_phi, d_A, d_buffer, d_Param, currentNumCells);
  cudaErrCheck(cudaGetLastError());
  cublasErrCheck(cublasSgemv(cudaTypes->handle, CUBLAS_OP_T, N * N,
                             currentNumCells, d_one, d_buffer, N * N, d_ones, 1,
                             d_zero, d_tmp_com, 1));
  copyToY<<<blocksNumCells, threadsNumCells>>>(d_tmp_com, cudaTypes->d_comA,
                                               currentNumCells);
}

void Model::rollAllFieldsX() {
  rollGridXKernel<<<cudaTypes->grid, cudaTypes->block>>>(
      d_phi, d_buffer, cudaTypes->d_grid_max, d_Param, currentNumCells);
  cudaErrCheck(cudaGetLastError());
  cublasErrCheck(cublasRswap(cudaTypes->handle, currentNumCells * N * N,
                             d_buffer, 1, d_phi, 1));

  rollGridXKernel<<<cudaTypes->grid, cudaTypes->block>>>(
      d_phi, d_buffer, cudaTypes->d_grid_min, d_Param, currentNumCells);
  cudaErrCheck(cudaGetLastError());
  cublasErrCheck(cublasRswap(cudaTypes->handle, currentNumCells * N * N,
                             d_buffer, 1, d_phi, 1));

  rollGridXKernel<<<cudaTypes->grid, cudaTypes->block>>>(
      d_A, d_buffer, cudaTypes->d_grid_max, d_Param, currentNumCells);
  cudaErrCheck(cudaGetLastError());
  cublasErrCheck(cublasRswap(cudaTypes->handle, currentNumCells * N * N,
                             d_buffer, 1, d_A, 1));

  rollGridXKernel<<<cudaTypes->grid, cudaTypes->block>>>(
      d_A, d_buffer, cudaTypes->d_grid_min, d_Param, currentNumCells);
  cudaErrCheck(cudaGetLastError());
  cublasErrCheck(cublasRswap(cudaTypes->handle, currentNumCells * N * N,
                             d_buffer, 1, d_A, 1));

  rollGridXKernel<<<cudaTypes->grid, cudaTypes->block>>>(
      d_R, d_buffer, cudaTypes->d_grid_max, d_Param, currentNumCells);
  cudaErrCheck(cudaGetLastError());
  cublasErrCheck(cublasRswap(cudaTypes->handle, currentNumCells * N * N,
                             d_buffer, 1, d_R, 1));

  rollGridXKernel<<<cudaTypes->grid, cudaTypes->block>>>(
      d_R, d_buffer, cudaTypes->d_grid_min, d_Param, currentNumCells);
  cudaErrCheck(cudaGetLastError());
  cublasErrCheck(cublasRswap(cudaTypes->handle, currentNumCells * N * N,
                             d_buffer, 1, d_R, 1));
}

void Model::rollAllFieldsY() {
  rollGridYKernel<<<cudaTypes->grid, cudaTypes->block>>>(
      d_phi, d_buffer, cudaTypes->d_grid_max, d_Param, currentNumCells);
  cublasErrCheck(cublasRswap(cudaTypes->handle, currentNumCells * N * N,
                             d_buffer, 1, d_phi, 1));

  rollGridYKernel<<<cudaTypes->grid, cudaTypes->block>>>(
      d_phi, d_buffer, cudaTypes->d_grid_min, d_Param, currentNumCells);
  cublasErrCheck(cublasRswap(cudaTypes->handle, currentNumCells * N * N,
                             d_buffer, 1, d_phi, 1));

  rollGridYKernel<<<cudaTypes->grid, cudaTypes->block>>>(
      d_A, d_buffer, cudaTypes->d_grid_max, d_Param, currentNumCells);
  cublasErrCheck(cublasRswap(cudaTypes->handle, currentNumCells * N * N,
                             d_buffer, 1, d_A, 1));

  rollGridYKernel<<<cudaTypes->grid, cudaTypes->block>>>(
      d_A, d_buffer, cudaTypes->d_grid_min, d_Param, currentNumCells);
  cublasErrCheck(cublasRswap(cudaTypes->handle, currentNumCells * N * N,
                             d_buffer, 1, d_A, 1));

  rollGridYKernel<<<cudaTypes->grid, cudaTypes->block>>>(
      d_R, d_buffer, cudaTypes->d_grid_max, d_Param, currentNumCells);
  cublasErrCheck(cublasRswap(cudaTypes->handle, currentNumCells * N * N,
                             d_buffer, 1, d_R, 1));

  rollGridYKernel<<<cudaTypes->grid, cudaTypes->block>>>(
      d_R, d_buffer, cudaTypes->d_grid_min, d_Param, currentNumCells);
  cublasErrCheck(cublasRswap(cudaTypes->handle, currentNumCells * N * N,
                             d_buffer, 1, d_R, 1));
}

std::vector<Real> Model::getCOMX() {
  std::vector<Real> comX(currentNumCells);
  cudaErrCheck(cudaMemcpy(comX.data(), cudaTypes->d_com,
                          currentNumCells * sizeof(Real2),
                          cudaMemcpyDeviceToHost));
  for (int i = 0; i < currentNumCells; ++i) {
    comX[i] = cudaTypes->com[i].x;
  }
  return comX;
}

std::vector<Real> Model::getCOMY() {
  std::vector<Real> comY(currentNumCells);
  cudaErrCheck(cudaMemcpy(comY.data(), cudaTypes->d_com,
                          currentNumCells * sizeof(Real2),
                          cudaMemcpyDeviceToHost));
  for (int i = 0; i < currentNumCells; ++i) {
    comY[i] = cudaTypes->com[i].y;
  }
  return comY;
}

std::vector<Real> Model::getCOMAX() {
  std::vector<Real> comAX(currentNumCells);
  cudaErrCheck(cudaMemcpy(comAX.data(), cudaTypes->d_comA,
                          currentNumCells * sizeof(Real2),
                          cudaMemcpyDeviceToHost));
  for (int i = 0; i < currentNumCells; ++i) {
    comAX[i] = cudaTypes->comA[i].x;
  }
  return comAX;
}

std::vector<Real> Model::getCOMAY() {
  std::vector<Real> comAY(currentNumCells);
  cudaErrCheck(cudaMemcpy(comAY.data(), cudaTypes->d_comA,
                          currentNumCells * sizeof(Real2),
                          cudaMemcpyDeviceToHost));
  for (int i = 0; i < currentNumCells; ++i) {
    comAY[i] = cudaTypes->comA[i].y;
  }
  return comAY;
}

std::vector<Real> Model::getArea() {
  std::vector<Real> areaVec(currentNumCells);
  cudaErrCheck(cudaMemcpy(areaVec.data(), d_area,
                          currentNumCells * sizeof(Real),
                          cudaMemcpyDeviceToHost));
  return areaVec;
}

void Model::updateCellDivisions() {
  int numNewCells = 0;
  bool triggerUpdate = false;
  for (int k = 0; k < currentNumCells; ++k) {
    time_since_division[k] += p.dt;
    if (time_since_division[k] >= paramArray[k].division_time) {
      triggerUpdate = true;
    }
  }
  if (triggerUpdate) {
    cudaErrCheck(cudaMemcpy(area, d_area, currentNumCells * sizeof(Real),
                            cudaMemcpyDeviceToHost));
    cudaErrCheck(
        cudaMemcpy(cudaTypes->grid_location, cudaTypes->d_grid_location,
                   currentNumCells * sizeof(int2), cudaMemcpyDeviceToHost));
    for (int k = 0; k < currentNumCells; ++k) {
      if (time_since_division[k] >= paramArray[k].division_time &&
          area[k] >= 0.95 * p.S0 / p.dx / p.dx) {
        new_cell_indices[k] = numNewCells + currentNumCells;
        numNewCells++;

        std::cout << "Dividing cell " << k << " at time "
                  << time_since_division[k] << " with area " << area[k]
                  << std::endl;
        std::cout << "New cell index: " << new_cell_indices[k] << std::endl;
        time_since_division[k] = 0.0;
        // Update the grid location for the new cell
        cudaTypes->grid_location[new_cell_indices[k]].x =
            cudaTypes->grid_location[k].x;
        cudaTypes->grid_location[new_cell_indices[k]].y =
            cudaTypes->grid_location[k].y;
        std::cout << "New cell location: ("
                  << cudaTypes->grid_location[new_cell_indices[k]].x << ", "
                  << cudaTypes->grid_location[new_cell_indices[k]].y << ")"
                  << std::endl;
        Real division_angle = M_PI * rand() / (RAND_MAX + 1.0);
        cudaTypes->division_axes[k].x = cos(division_angle);
        cudaTypes->division_axes[k].y = sin(division_angle);

      } else {
        new_cell_indices[k] = 0;  // no division
      }
    }
  }
  add_time<<<blocksNumCells, threadsNumCells>>>(d_time_since_division, p.dt,
                                                currentNumCells);

  if (numNewCells > 0) {
    // update the device memory with the new cell indices
    cudaErrCheck(cudaMemcpy(d_new_cell_indices, new_cell_indices,
                            currentNumCells * sizeof(int),
                            cudaMemcpyHostToDevice));
    // Update the COM for accurate division plane
    updateCOM();

    cudaMemcpy(cudaTypes->d_division_axes, cudaTypes->division_axes,
               currentNumCells * sizeof(Real2), cudaMemcpyHostToDevice);

    divideCells<<<cudaTypes->grid, cudaTypes->block>>>(
        d_phi, d_A, d_R, d_Param, d_new_cell_indices, cudaTypes->d_com, d_area,
        cudaTypes->d_division_axes, currentNumCells);
    cudaErrCheck(cudaGetLastError());

    currentNumCells += numNewCells;
    cudaTypes->grid = dim3((N + blockSizeX - 1) / blockSizeX,
                           (N + blockSizeY - 1) / blockSizeY,
                           (currentNumCells + blockSizeZ - 1) / blockSizeZ);

    threadsNumCells = std::min(currentNumCells, 1024);
    blocksNumCells = (currentNumCells + threadsNumCells - 1) / threadsNumCells;

    cudaErrCheck(
        cudaMemcpy(cudaTypes->d_grid_location, cudaTypes->grid_location,
                   currentNumCells * sizeof(int2), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(d_time_since_division, time_since_division,
                            currentNumCells * sizeof(int),
                            cudaMemcpyHostToDevice));
    updateGridLocation();
  }
}
