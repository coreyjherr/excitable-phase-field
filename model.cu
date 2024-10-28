#include "include/indicators.hpp"
#include "include/model.cuh"
#include "include/error_check.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "include/kernels.cuh"
#include "include/params.cuh"
#include <iostream>
#include <fstream>
#include <math.h>
#include <thread>
#include <sstream>
#include <iomanip>

#define USE_FLOAT

#ifdef USE_FLOAT
#define cublasRasum(handle, n, x, incx, result) cublasSasum(handle, n, x, incx, result)
#define cublasRaxpy(handle, n, alpha, x, incx, y, incy) cublasSaxpy(handle, n, alpha, x, incx, y, incy)
#define cublasIramin(handle, n, x, incx, result) cublasIsamin(handle, n, x, incx, result)
#define cublasIramax(handle, n, x, incx, result) cublasIsamax(handle, n, x, incx, result)
#define cublasRswap(handle, n, x, incx, y, incy) cublasSswap(handle, n, x, incx, y, incy)
#endif
#ifndef USE_FLOAT
#define cublasRasum(handle, n, x, incx, result) cublasDasum(handle, n, x, incx, result)
#define cublasRaxpy(handle, n, alpha, x, incx, y, incy) cublasDaxpy(handle, n, alpha, x, incx, y, incy)
#define cublasIramin(handle, n, x, incx, result) cublasIdamin(handle, n, x, incx, result)
#define cublasIramax(handle, n, x, incx, result) cublasIdamax(handle, n, x, incx, result)
#define cublasRswap(handle, n, x, incx, y, incy) cublasDswap(handle, n, x, incx, y, incy)
#endif


Model::Model(Param param, int deviceID, std::string basePath, unsigned long long seed) {
    p = param;
    p.S0 = M_PI*p.r0*p.r0;
    numCells = p.numCells;

    deviceID_ = deviceID;
    cudaErrCheck(cudaSetDevice(deviceID_));
    rngSeed = seed;

    block = dim3(blockSizeX, blockSizeY, blockSizeZ);
    grid = dim3((p.N + blockSizeX - 1) / blockSizeX, (p.N + blockSizeY - 1) / blockSizeY, (numCells + blockSizeZ - 1) / blockSizeZ);

    cudaErrCheck(cudaMalloc((void**)&d_Param, sizeof(Param)));
    cudaErrCheck(cudaMemcpy(d_Param, &p, sizeof(Param), cudaMemcpyHostToDevice));

    // Initialize device and host pointers
    initializeDeviceMemory();
    initializeHostMemory();

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
    setupCurandStates<<<grid, block>>>(d_states, rngSeed, 0, d_Param);
    setupCurandStates<<<grid, block>>>(d_Rstates, rngSeed, numCells*p.N*p.N, d_Param);
    curandErrCheck(cudaGetLastError());

    initializeOutput(basePath);
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
    cudaErrCheck(cudaFree(d_com));
    cudaErrCheck(cudaFree(d_comA));


    delete[] phi;
    delete[] A;
    delete[] R;
    delete[] area;
    delete[] Aarea;
    delete[] grid_location;
    delete[] com;
    delete[] comA;


    cublasErrCheck(cublasDestroy(handle));

    // Close the output files after mutex lock
    std::lock_guard<std::mutex> guard(fileMutex);
    phi_output.close();
    A_output.close();
    grid_location_output.close();
    com_output.close();
    comA_output.close();
}

void Model::initializeHostMemory() {
    phi = new Real[numCells*p.N*p.N];
    A = new Real[numCells*p.N*p.N];
    R = new Real[numCells*p.N*p.N];
    area = new Real[numCells];
    Aarea = new Real[numCells];
    grid_location = new int2[numCells];
    com = new Real2[numCells];
    comA = new Real2[numCells];
}

void Model::initializeDeviceMemory() {
    // Allocate memory on device
    cudaErrCheck(cudaMalloc((void**)&d_phi, numCells*p.N*p.N*sizeof(Real)));
    cudaErrCheck(cudaMalloc((void**)&d_A, numCells*p.N*p.N*sizeof(Real)));
    cudaErrCheck(cudaMalloc((void**)&d_R, numCells*p.N*p.N*sizeof(Real)));
    cudaErrCheck(cudaMalloc((void**)&d_area, numCells*sizeof(Real)));
    cudaErrCheck(cudaMalloc((void**)&d_Aarea, numCells*sizeof(Real)));
    cudaErrCheck(cudaMalloc((void**)&d_collision_force, numCells*p.N*p.N*sizeof(Real)));
    cudaErrCheck(cudaMalloc((void**)&d_buffer, numCells*p.N*p.N*sizeof(Real)));
    cudaErrCheck(cudaMalloc((void**)&d_phi_t, numCells*p.N*p.N*sizeof(Real)));
    cudaErrCheck(cudaMalloc((void**)&d_grid_min, numCells*sizeof(int2)));
    cudaErrCheck(cudaMalloc((void**)&d_grid_max, numCells*sizeof(int2)));
    cudaErrCheck(cudaMalloc((void**)&d_grid_location, numCells*sizeof(int2)));
    cudaErrCheck(cudaMalloc((void**)&d_com, numCells*sizeof(Real2)));
    cudaErrCheck(cudaMalloc((void**)&d_comA, numCells*sizeof(Real2)));


    // Initialize memory on device
    cudaErrCheck(cudaMemset(d_phi, 0, numCells*p.N*p.N*sizeof(Real)));
    cudaErrCheck(cudaMemset(d_A, 0, numCells*p.N*p.N*sizeof(Real)));
    cudaErrCheck(cudaMemset(d_R, 0, numCells*p.N*p.N*sizeof(Real)));
    cudaErrCheck(cudaMemset(d_area, 0, numCells*sizeof(Real)));
    cudaErrCheck(cudaMemset(d_Aarea, 0, numCells*sizeof(Real)));
    cudaErrCheck(cudaMemset(d_collision_force, 0, numCells*p.N*p.N*sizeof(Real)));
    cudaErrCheck(cudaMemset(d_buffer, 0, numCells*p.N*p.N*sizeof(Real)));
    cudaErrCheck(cudaMemset(d_phi_t, 0, numCells*p.N*p.N*sizeof(Real)));
    cudaErrCheck(cudaMemset(d_com, 0, numCells*sizeof(Real2)));
    cudaErrCheck(cudaMemset(d_comA, 0, numCells*sizeof(Real2)));

}

void Model::initializeCellGrid() {
    // int sqrt_numCells = sqrt(1.0*numCells)+1;
    // for (int i = 0; i < sqrt_numCells; ++i) {
    //     for (int j = 0; j < sqrt_numCells; ++j) {
    //         if (i*sqrt_numCells + j >= numCells) {
    //             break;
    //         }
    //         grid_location[i*sqrt_numCells + j].x = p.initial_offset_x*i;
    //         grid_location[i*sqrt_numCells + j].y = p.initial_offset_y*j;
    //     }
    // }

    Real phi = (1 + sqrt(5))/2;
    for (int i = 0; i < numCells; ++i) {
        int k = i+1;
        int b = round(2*sqrt(numCells));
        Real r;
        if (k > numCells-b)
            r  = 1;
        else
            r = sqrt(1.0*k-0.5)/sqrt(1.0*numCells - (1.0*b+1)/2);
        Real theta = 2*M_PI*k/(phi*phi);
        Real rCircle = (sqrt(numCells)-1)*p.r0*r;
        grid_location[i].x = rCircle/p.dx*cos(theta);
        grid_location[i].y = rCircle/p.dx*sin(theta);
    }
}

void Model::initializeOutput(std::string basePath) {
    using std::string;
    // Get the current time and date
    std::ostringstream oss;
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
    string time_str = oss.str();
    time_str += "_" + std::to_string(deviceID_);

    string phi_filestring = basePath + "phi_" + time_str + ".dat";
    string A_filestring = basePath + "A_" + time_str + ".dat";
    string grid_location_filestring = basePath + "grid_location_" + time_str + ".dat";
    string com_filestring = basePath + "com_" + time_str + ".dat";
    string comA_filestring = basePath + "comA_" + time_str + ".dat";

    writeParamFile(basePath +  "param_" + time_str + ".txt", p, rngSeed);
    phi_output = std::ofstream(phi_filestring, std::ios::out | std::ios::binary);
    A_output = std::ofstream(A_filestring, std::ios::out | std::ios::binary);
    grid_location_output = std::ofstream(grid_location_filestring, std::ios::out | std::ios::binary);
    com_output = std::ofstream(com_filestring, std::ios::out | std::ios::binary);
    comA_output = std::ofstream(comA_filestring, std::ios::out | std::ios::binary);

    grid_location_output.write(reinterpret_cast<char*>(&numCells), sizeof(int));
    int numTimeOutputs = p.T/p.blind;
    grid_location_output.write(reinterpret_cast<char*>(&numTimeOutputs), sizeof(int));
}

void Model::simulate(bool progressBar, bool useCondensedOutput) {
    using namespace indicators;
    BlockProgressBar bar{
        option::BarWidth{80}, option::ForegroundColor{Color::white},
        option::FontStyles{std::vector<FontStyle>{FontStyle::bold}},
        option::MaxProgress{p.T+startupSteps}, option::ShowElapsedTime{true},option::ShowRemainingTime{true}};

    int earlyExit = 0;
    for (int t=0; t < startupSteps; t++)
    {
        // Early stages to stabilize the system
        updateArea();
        updateCollisionForce();
        updatePhi(t);
        if (t%100 == 0)
        {
            updateGridLocation();
        }
        if (progressBar) {
            bar.tick();
        }
    }

    for (int t = 0; t < p.T; t++) {
        updateArea();

        if (t % p.blind == 0) {
            updateGridLocation();
            updateCOM();
            earlyExit = write(useCondensedOutput);
            if (earlyExit)
            {
                return;
            }
        }

        updateCollisionForce();
        updatePhi(t);
        updateA();
        updateR();

        if (progressBar) {
            bar.tick();
        }
    }
}


void Model::updateArea() {
    for (int i = 0; i < numCells; ++i) {
        cublasErrCheck(cublasRasum(handle, p.N*p.N, d_phi + i*p.N*p.N, 1, d_area + i));
    }
}

void Model::updateA_Area() {
}

void Model::updatePhi(int t) {
    updatePhiKernel<<<grid, block>>>(d_phi, d_A, d_R, d_Param, d_area, d_grid_location, d_collision_force, d_phi_t, t);
    cublasErrCheck(cublasRaxpy(handle, numCells*p.N*p.N, &(d_Param->dt), d_phi_t, 1, d_phi, 1));
}

void Model::updateA() {
    updateAKernel<<<grid, block>>>(d_phi, d_A, d_R, d_Param, d_phi_t, d_states, d_buffer);
    cublasErrCheck(cublasRaxpy(handle, numCells*p.N*p.N, &(d_Param->dt), d_buffer, 1, d_A, 1));
}

void Model::updateR() {
    updateRKernel<<<grid, block>>>(d_phi, d_A, d_R, d_Param, d_phi_t, d_Rstates, d_buffer);
    cublasErrCheck(cublasRaxpy(handle, numCells*p.N*p.N, &(d_Param->dt), d_buffer, 1, d_R, 1));
}

void Model::updateCollisionForce() {
    updateCollisionForceKernel<<<grid, block>>>(d_phi, d_grid_location, d_Param, d_collision_force);
}

void Model::updateGridLocation() {
    getXIndexMaskedKernel<<<grid, block>>>(d_phi, d_buffer, d_Param, 0.5);
    for (int i = 0; i < numCells; ++i) {
        cublasErrCheck(cublasIramax(handle, p.N*p.N, d_buffer + i*p.N*p.N, 1, &(d_grid_max + i)->x));
        cublasErrCheck(cublasIramin(handle, p.N*p.N, d_buffer + i*p.N*p.N, 1, &(d_grid_min + i)->x));
    }
    cudaErrCheck(cudaDeviceSynchronize());
    rollAllFieldsX();

    getYIndexMaskedKernel<<<grid, block>>>(d_phi, d_buffer, d_Param, 0.5);
    for (int i = 0; i < numCells; ++i) {
        cublasErrCheck(cublasIramax(handle, p.N*p.N, d_buffer + i*p.N*p.N, 1, &((d_grid_max + i)->y)));
        cublasErrCheck(cublasIramin(handle, p.N*p.N, d_buffer + i*p.N*p.N, 1, &((d_grid_min + i)->y)));
    }
    cudaErrCheck(cudaDeviceSynchronize());
    rollAllFieldsY();
    cudaErrCheck(cudaDeviceSynchronize());

    updateGridLocationKernel<<<1, numCells>>>(d_grid_min, d_grid_location, d_Param);
    updateGridLocationKernel<<<1, numCells>>>(d_grid_max, d_grid_location, d_Param);

}

void Model::updateBoundingBox() {
    // Store the bounding box location for each cell inside d_grid_max and d_grid_min
    getXIndexMaskedKernel<<<grid, block>>>(d_phi, d_buffer, d_Param, 0.0025);
    for (int i = 0; i < numCells; ++i) {
        cublasErrCheck(cublasIramax(handle, p.N*p.N, d_buffer + i*p.N*p.N, 1, &(d_grid_max + i)->x));
        cublasErrCheck(cublasIramin(handle, p.N*p.N, d_buffer + i*p.N*p.N, 1, &(d_grid_min + i)->x));
    }

    getYIndexMaskedKernel<<<grid, block>>>(d_phi, d_buffer, d_Param, 0.0025);
    for (int i = 0; i < numCells; ++i) {
        cublasErrCheck(cublasIramax(handle, p.N*p.N, d_buffer + i*p.N*p.N, 1, &((d_grid_max + i)->y)));
        cublasErrCheck(cublasIramin(handle, p.N*p.N, d_buffer + i*p.N*p.N, 1, &((d_grid_min + i)->y)));
    }

    updateBoundingIndex<<<1, numCells>>>(d_grid_min, d_Param);
    updateBoundingIndex<<<1, numCells>>>(d_grid_max, d_Param);
}

void Model::updateCOM() {
    // Calculate the center of mass of each cell within its own grid
    createCOMXKernel<<<grid, block>>>(d_phi, d_buffer, d_Param);
    for (int i =0; i<numCells; ++i)
    {
        cublasRasum(handle, p.N*p.N, d_buffer + i*p.N*p.N, 1, &((d_com + i)->x));
    }

    createCOMYKernel<<<grid, block>>>(d_phi, d_buffer, d_Param);
    for (int i =0; i<numCells; ++i)
    {
        cublasRasum(handle, p.N*p.N, d_buffer + i*p.N*p.N, 1, &((d_com + i)->y));
    }

    multiplyKernel<<<grid, block>>>(d_phi, d_A, d_buffer, d_Param);
    for (int i=0; i<numCells; ++i) {
        cublasErrCheck(cublasRasum(handle, p.N*p.N, d_buffer + i*p.N*p.N, 1, d_Aarea + i));
    }
    
    createCOMAXKernel<<<grid, block>>>(d_phi, d_A, d_buffer, d_Param);
    for (int i =0; i<numCells; ++i)
    {
        cublasRasum(handle, p.N*p.N, d_buffer + i*p.N*p.N, 1, &((d_comA + i)->x));
    }
    
    createCOMAYKernel<<<grid, block>>>(d_phi, d_A, d_buffer, d_Param);
    for (int i =0; i<numCells; ++i)
    {
        cublasRasum(handle, p.N*p.N, d_buffer + i*p.N*p.N, 1, &((d_comA + i)->y));
    }

    cudaErrCheck(cudaMemcpy(com, d_com, numCells*sizeof(Real2), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(comA, d_comA, numCells*sizeof(Real2), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(area, d_area, numCells*sizeof(Real), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(Aarea, d_Aarea, numCells*sizeof(Real), cudaMemcpyDeviceToHost));
    for (int i=0; i < numCells; ++i)
    {
        com[i].x /= area[i];
        com[i].y /= area[i];
        comA[i].x /= Aarea[i];
        comA[i].y /= Aarea[i];
    }

}


void Model::rollAllFieldsX() {
    rollGridXKernel<<<grid, block>>>(d_phi, d_buffer, d_grid_max, d_Param);
    cublasErrCheck(cublasRswap(handle, numCells*p.N*p.N, d_buffer, 1, d_phi, 1));

    rollGridXKernel<<<grid, block>>>(d_phi, d_buffer, d_grid_min, d_Param);
    cublasErrCheck(cublasRswap(handle, numCells*p.N*p.N, d_buffer, 1, d_phi, 1));

    rollGridXKernel<<<grid, block>>>(d_A, d_buffer, d_grid_max, d_Param);
    cublasErrCheck(cublasRswap(handle, numCells*p.N*p.N, d_buffer, 1, d_A, 1));

    rollGridXKernel<<<grid, block>>>(d_A, d_buffer, d_grid_min, d_Param);
    cublasErrCheck(cublasRswap(handle, numCells*p.N*p.N, d_buffer, 1, d_A, 1));

    rollGridXKernel<<<grid, block>>>(d_R, d_buffer, d_grid_max, d_Param);
    cublasErrCheck(cublasRswap(handle, numCells*p.N*p.N, d_buffer, 1, d_R, 1));

    rollGridXKernel<<<grid, block>>>(d_R, d_buffer, d_grid_min, d_Param);
    cublasErrCheck(cublasRswap(handle, numCells*p.N*p.N, d_buffer, 1, d_R, 1));
}

void Model::rollAllFieldsY() {
    rollGridYKernel<<<grid, block>>>(d_phi, d_buffer, d_grid_max, d_Param);
    cublasErrCheck(cublasRswap(handle, numCells*p.N*p.N, d_buffer, 1, d_phi, 1));

    rollGridYKernel<<<grid, block>>>(d_phi, d_buffer, d_grid_min, d_Param);
    cublasErrCheck(cublasRswap(handle, numCells*p.N*p.N, d_buffer, 1, d_phi, 1));

    rollGridYKernel<<<grid, block>>>(d_A, d_buffer, d_grid_max, d_Param);
    cublasErrCheck(cublasRswap(handle, numCells*p.N*p.N, d_buffer, 1, d_A, 1));

    rollGridYKernel<<<grid, block>>>(d_A, d_buffer, d_grid_min, d_Param);
    cublasErrCheck(cublasRswap(handle, numCells*p.N*p.N, d_buffer, 1, d_A, 1));

    rollGridYKernel<<<grid, block>>>(d_R, d_buffer, d_grid_max, d_Param);
    cublasErrCheck(cublasRswap(handle, numCells*p.N*p.N, d_buffer, 1, d_R, 1));

    rollGridYKernel<<<grid, block>>>(d_R, d_buffer, d_grid_min, d_Param);
    cublasErrCheck(cublasRswap(handle, numCells*p.N*p.N, d_buffer, 1, d_R, 1));
}

WriteError Model::makeCondensedOutput(Real* field1, Real* field2, std::ofstream& outputField)
{
    int2 min_grid_location;
    min_grid_location.x = grid_location[0].x;
    min_grid_location.y = grid_location[0].y;
    int2 max_grid_location;
    max_grid_location.x = grid_location[0].x;
    max_grid_location.y = grid_location[0].y;

    for (int i=0; i<numCells; i++)
    {
        min_grid_location.x = grid_location[i].x < min_grid_location.x ? grid_location[i].x : min_grid_location.x;
        min_grid_location.y = grid_location[i].y < min_grid_location.y ? grid_location[i].y : min_grid_location.y;
        max_grid_location.x = grid_location[i].x > max_grid_location.x ? grid_location[i].x : max_grid_location.x;
        max_grid_location.y = grid_location[i].y > max_grid_location.y ? grid_location[i].y : max_grid_location.y;
    }
    int condensed_size_x = max_grid_location.x - min_grid_location.x + p.N;
    int condensed_size_y = max_grid_location.y - min_grid_location.y + p.N;
    if (condensed_size_x < 0 || condensed_size_y < 0)
    {
        std::cerr << "Error: condensed_size_x or condensed_size_y is negative" << std::endl;
        return 1;
    }
    if (condensed_size_x > 10000 || condensed_size_y > 10000)
    {
        std::cerr << "Error: condensed_size_x or condensed_size_y is too large" << std::endl;
        return 1;
    }

    Real* condensed_field = new Real[condensed_size_x*condensed_size_y];
    for (int i=0; i<condensed_size_x*condensed_size_y; i++)
    {
        condensed_field[i] = 0;
    }

    if (field2 == nullptr)
    {
    for (int k=0; k<numCells; k++)
    {
        for (int i=0; i<p.N; i++)
        {
            for (int j=0; j<p.N; j++)
            {
                int xcoord = grid_location[k].x + i - min_grid_location.x;
                int ycoord = grid_location[k].y + j - min_grid_location.y;
                condensed_field[ycoord*condensed_size_x + xcoord] += field1[k*p.N*p.N + j*p.N + i];
            }
        }
    }
    }
    else
    {
    for (int k=0; k<numCells; k++)
    {
        for (int i=0; i<p.N; i++)
        {
            for (int j=0; j<p.N; j++)
            {
                int xcoord = grid_location[k].x + i - min_grid_location.x;
                int ycoord = grid_location[k].y + j - min_grid_location.y;
                condensed_field[ycoord*condensed_size_x + xcoord] += field1[k*p.N*p.N + j*p.N + i] * field2[k*p.N*p.N + j*p.N + i];
            }
        }
    }
    }

    outputField.write(reinterpret_cast<char*>(&condensed_size_x), sizeof(int));
    outputField.write(reinterpret_cast<char*>(&condensed_size_y), sizeof(int));
    outputField.write(reinterpret_cast<char*>(condensed_field), condensed_size_x*condensed_size_y*sizeof(Real));
    delete[] condensed_field;
    return 0;
}

WriteError Model::write(bool useCondensedOutput) {
    // Write phi and A to file
    std::lock_guard<std::mutex> guard(fileMutex); // Lock the mutex
    cudaErrCheck(cudaMemcpy(phi, d_phi, numCells*p.N*p.N*sizeof(Real), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(A, d_A, numCells*p.N*p.N*sizeof(Real), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(grid_location, d_grid_location, numCells*sizeof(int2), cudaMemcpyDeviceToHost));

    if (useCondensedOutput) {
        WriteError earlyExit = makeCondensedOutput(phi, nullptr, phi_output);
        WriteError earlyExitA = makeCondensedOutput(A, phi, A_output);
        if (earlyExit || earlyExitA)
        {
            return 1;
        }
    }
    else{
        phi_output.write(reinterpret_cast<char*>(phi), numCells*p.N*p.N*sizeof(Real));
        A_output.write(reinterpret_cast<char*>(A), numCells*p.N*p.N*sizeof(Real));
    }

    grid_location_output.write(reinterpret_cast<char*>(grid_location), numCells*sizeof(int2));

    for (int i = 0; i < numCells; ++i) {
        com[i].x += grid_location[i].x;
        com[i].y += grid_location[i].y;
        comA[i].x += grid_location[i].x;
        comA[i].y += grid_location[i].y;
    }
    com_output.write(reinterpret_cast<char*>(com), numCells*sizeof(Real2));
    comA_output.write(reinterpret_cast<char*>(comA), numCells*sizeof(Real2));
    return 0;
}

// Function to run the file writing operation in a background thread
void Model::writeInBackground(bool useCondensedOutput) {
    std::thread backgroundThread(&Model::write, this, useCondensedOutput);
    backgroundThread.detach(); // Detach the thread to run in the background
    
}

std::vector<Real> range(Real start, Real stop, Real step) {
    std::vector<Real> result;
    for (Real i = start; i < stop; i += step) {
        result.push_back(i);
    }
    return result;
}

std::vector<std::vector<Real>> getAllCombinations(std::vector<std::vector<Real>> params) {
    // get all the combinations from all lists of parameters
    std::vector<std::vector<Real>> result;
    if (params.size() == 0)
        return result;
    if (params.size() == 1)
    {
        for (auto p : params[0])
        {
            result.push_back({p});
        }
        return result;
    }
    std::vector<std::vector<Real>> rest = getAllCombinations(std::vector<std::vector<Real>>(params.begin()+1, params.end()));
    for (auto p : params[0])
    {
        for (auto r : rest)
        {
            std::vector<Real> temp = {p};
            temp.insert(temp.end(), r.begin(), r.end());
            result.push_back(temp);
        }
    }
    return result;
}

