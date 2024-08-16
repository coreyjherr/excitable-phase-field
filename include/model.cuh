#ifndef MODEL_CUH
#define MODEL_CUH

#include <cublas_v2.h>
#include <curand_kernel.h>
#include "params.cuh"
#include <fstream>
#include <mutex>

class Model {
private:
    Real *phi, *A, *R, *area;
    Real *d_phi, *d_A, *d_R, *d_area, *d_collision_force, *d_buffer, *d_phi_t;
    int *d_grid_min_x, *d_grid_min_y, *d_grid_max_x, *d_grid_max_y;

    Param p;
    Param *d_Param;

    int numCells;

    cublasHandle_t handle;

    curandState *d_states, *d_Rstates;
    std::ofstream phi_output;
    std::ofstream A_output;

    dim3 grid, block;
    void initializePhi();
    std::mutex fileMutex;

    void rollAllFieldsX();
    void rollAllFieldsY();

public:
    Model(Param p);
    ~Model();
    void updatePhi();
    void updateA();
    void updateR();
    void updateArea();
    void updateGridLocation();
    void write();
    void writeArray();
    void writeArrayToFileInBackground();
};


#endif // MODEL_CUH
