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
#ifndef MODEL_CUH
#define MODEL_CUH

#include <cublas_v2.h>
#include <curand_kernel.h>
#include "params.hpp"
#include <fstream>
#include <mutex>
#include <vector>

#ifndef Real2
#define Real2 float2
#endif

typedef int WriteError ;

class Model {
private:
    Real* d_ones, *d_beta, *d_alpha;

    cudaStream_t streams[32];
    int blockSizeX = 16;
    int blockSizeY = 16;
    int blockSizeZ = 1;
    int startupSteps = 50000;

    Real *phi, *A, *R, *C, *area, *Aarea;
    Real *d_A, *d_R, *d_C, *d_area, *d_Aarea, *d_collision_force, *d_buffer, *d_phi_t, *d_Cbuffer;
    int2 *d_grid_location, *grid_location, *d_grid_min, *d_grid_max;
    Real2 *com, *d_com, *comA, *d_comA;

    Param p;
    Param *d_Param;

    int numCells, deviceID_;

    cublasHandle_t handle;

    curandState *d_states, *d_Rstates;
    std::ofstream phi_output, A_output, label_output, C_output;
    std::ofstream dynamics_output;

    void initializeDeviceMemory();
    void initializeHostMemory();
    void initializePhi();
    std::mutex fileMutex;

    void rollAllFieldsX();
    void rollAllFieldsY();
    void initializeCellGrid(Real totalArea);
    void initializeOutput(std::string basePath, std::string simulationID);

public:
    unsigned long long rngSeed;
    dim3 grid, block, Cblocks, Cthreads_per_block;
    Real *d_phi;
    Model(Param p, int deviceID = 0, std::string basePath = "./results/", unsigned long long seed=0, std::string simulationID="");
    ~Model();
    void simulate(bool progressBar = true, bool useCondensedOutput = false);
    void updatePhi(int t);
    void updateC();
    void updateA_Area();
    void updateA();
    void updateR();
    void updateArea();
    void updateGridLocation();
    void updateCollisionForce();
    void updateBoundingBox();
    WriteError write(bool useCondensedOutput=false);
    WriteError writeDynamics(int step);
    void writeInBackground(bool useCondensedOutput=false);
    void updateCOM();
    WriteError makeCondensedOutput(Real* field1, Real* field2,std::ofstream& outputField);
    std::string simulationID;
};

#endif // MODEL_CUH
