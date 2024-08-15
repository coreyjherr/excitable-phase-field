#include "model.cuh"
#include "error_check.cuh"

Model::Model(int numCells, Param param) {

    p = param;
    phi = new Real[numCells*p.N*p.N];
    A = new Real[numCells*p.N*p.N];
    R = new Real[numCells*p.N*p.N];

    cudaSetDevice(0);

    cudaErrCheck(cudaMalloc((void**)&d_phi, numCells*p.N*p.N*sizeof(Real)));
    cudaErrCheck(cudaMalloc((void**)&d_A, numCells*p.N*p.N*sizeof(Real)));
    cudaErrCheck(cudaMalloc((void**)&d_R, numCells*p.N*p.N*sizeof(Real)));

    cudaErrCheck(cudaMemset(d_phi, 0, numCells*p.N*p.N*sizeof(Real)));
    cudaErrCheck(cudaMemset(d_A, 0, numCells*p.N*p.N*sizeof(Real)));
    cudaErrCheck(cudaMemset(d_R, 0, numCells*p.N*p.N*sizeof(Real)));

}

Model::~Model() {
    std::cout << "Destructor called\n";

    cudaErrCheck(cudaFree(d_phi));
    cudaErrCheck(cudaFree(d_A));
    cudaErrCheck(cudaFree(d_R));

    delete[] phi;
    delete[] A;
    delete[] R;

}
