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
#ifndef MODEL_H
#define MODEL_H

#include "params.hpp"
#include <fstream>
#include <mutex>
#include <vector>

#define WRITE_FULL_OUTPUT 0
#define WRITE_CONDENSED_OUTPUT 1
#define WRITE_DYNAMICS_OUTPUT 2
#define WRITE_PARAMETERS_OUTPUT 3
#define WRITE_NONE 4

typedef int WriteError ;

class Model {
private:
    struct CudaTypes;
    CudaTypes *cudaTypes;

    Real* d_ones, *d_beta, *d_alpha;

    int blockSizeX = 16;
    int blockSizeY = 16;
    int blockSizeZ = 1;
    int blocksNumCells, threadsNumCells;
    int startupSteps = 50000;

    Real *phi, *A, *R, *C, *area, *Aarea;
    Real *d_A, *d_R, *d_C, *d_area, *d_Aarea, *d_collision_force, *d_buffer, *d_phi_t, *d_Cbuffer;

    Param p;
    Param *d_Param;

    int numCells, deviceID_;

    std::ofstream phi_output, A_output, label_output, C_output;
    std::ofstream dynamics_output;

    void initializeDeviceMemory();
    void initializeHostMemory();
    void initializePhi();
    std::mutex fileMutex;
    Real *d_output, *h_output;
    Real *d_tmp_com;

    void rollAllFieldsX();
    void rollAllFieldsY();
    void initializeCellGrid(Real totalArea);
    void initializeOutput(std::string basePath, std::string time_str);
    void initializeDynamicsOutput(std::string basePath, std::string time_str);
    void copyToBacklog(int step, int backlog_index); 
    bool writeFullOutput_;
    bool writeCondensedOutput_;
    bool writeDynamicsOutput_;

public:
    unsigned long long rngSeed;
    Real *d_phi;
    Model(Param p, int deviceID = 0, std::string basePath = "./results/", unsigned long long seed=0, std::string simulationID="", int outputType=WRITE_CONDENSED_OUTPUT);
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
    std::vector<Real> getCOMX();
    std::vector<Real> getCOMY();
    std::vector<Real> getCOMAX();
    std::vector<Real> getCOMAY();
    std::vector<Real> getArea();
    WriteError writeFullOutput();
    WriteError writeDynamics(int step);
    void writeInBackground(bool useCondensedOutput=false);
    void updateCOM();
    WriteError makeCondensedOutput(Real* field1, Real* field2,std::ofstream& outputField);
    std::string simulationID;
};

#endif // MODEL_H
