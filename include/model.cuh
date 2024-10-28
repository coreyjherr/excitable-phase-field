#ifndef MODEL_CUH
#define MODEL_CUH

#include <cublas_v2.h>
#include <curand_kernel.h>
#include "params.cuh"
#include <fstream>
#include <mutex>
#include <vector>

typedef int WriteError ;

class Model {
private:
    int blockSizeX = 16;
    int blockSizeY = 16;
    int blockSizeZ = 1;
    int startupSteps = 80000;

    Real *phi, *A, *R, *area, *Aarea;
    Real *d_A, *d_R, *d_area, *d_Aarea, *d_collision_force, *d_buffer, *d_phi_t;
    int2 *d_grid_location, *grid_location, *d_grid_min, *d_grid_max;
    Real2 *com, *d_com, *comA, *d_comA;

    Param p;
    Param *d_Param;

    int numCells, deviceID_;

    cublasHandle_t handle;

    curandState *d_states, *d_Rstates;
    std::ofstream phi_output, A_output, grid_location_output, com_output, comA_output;

    void initializeDeviceMemory();
    void initializeHostMemory();
    void initializePhi();
    std::mutex fileMutex;

    void rollAllFieldsX();
    void rollAllFieldsY();
    void initializeCellGrid();
    void initializeOutput(std::string basePath);

public:
    unsigned long long rngSeed;
    dim3 grid, block;
    Real *d_phi;
    Model(Param p, int deviceID = 0, std::string basePath = "./results/", unsigned long long seed=0);
    ~Model();
    void simulate(bool progressBar = true, bool useCondensedOutput = false);
    void updatePhi(int t);
    void updateA_Area();
    void updateA();
    void updateR();
    void updateArea();
    void updateGridLocation();
    void updateCollisionForce();
    void updateBoundingBox();
    WriteError write(bool useCondensedOutput=false);
    void writeInBackground(bool useCondensedOutput=false);
    void updateCOM();
    WriteError makeCondensedOutput(Real* field1, Real* field2,std::ofstream& outputField);
};

std::vector<std::vector<Real>> getAllCombinations(std::vector<std::vector<Real>> params);
std::vector<Real> range(Real start, Real stop, Real step); 

#endif // MODEL_CUH
