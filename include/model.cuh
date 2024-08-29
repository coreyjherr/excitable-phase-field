#ifndef MODEL_CUH
#define MODEL_CUH

#include <cublas_v2.h>
#include <curand_kernel.h>
#include "params.cuh"
#include <fstream>
#include <mutex>

class Model {
private:
    int blockSizeX = 16;
    int blockSizeY = 16;
    int blockSizeZ = 1;

    Real *phi, *A, *R, *area;
    Real *d_phi, *d_A, *d_R, *d_area, *d_collision_force, *d_buffer, *d_phi_t;
    int2 *d_grid_location, *grid_location, *d_grid_min, *d_grid_max;
    Real2 *com, *d_com;

    Param p;
    Param *d_Param;

    int numCells, deviceID_;

    cublasHandle_t handle;

    curandState *d_states, *d_Rstates;
    std::ofstream phi_output, A_output, grid_location_output, com_output;

    dim3 grid, block;
    void initializePhi();
    std::mutex fileMutex;

    void rollAllFieldsX();
    void rollAllFieldsY();
    void initializeCellGrid();
    void setupOutput();

public:
    Model(Param p, int deviceID = 0);
    ~Model();
    void simulate(bool progressBar = true, bool useCondensedOutput = false);
    void updatePhi(int t);
    void updateA();
    void updateR();
    void updateArea();
    void updateGridLocation();
    void updateCollisionForce();
    void updateBoundingBox();
    void write(bool useCondensedOutput=false);
    void writeInBackground(bool useCondensedOutput=false);
    void updateCOM();
    void makeCondensedOutput(Real* field1, Real* field2,std::ofstream& outputField);
};


#endif // MODEL_CUH
