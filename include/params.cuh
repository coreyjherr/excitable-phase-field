#ifndef PARAMS_CUH
#define PARAMS_CUH

typedef float Real;

struct Param { 
    int N = 256;
    int T = 15000;
    int blind = 100;
    Real dt = 0.001;
    Real dx = 0.2;

    Real gamma = 2;
    Real eps2 = 4;
    Real B_s = 10;
    Real xi = 4;
    Real alpha = 5;
    Real k_a = 10;
    Real K = 1;
    Real b = 0.1;
    Real A_t = 2;
    Real d1 = 1;
    Real d2 = 1;
    Real c1 = 1;
    Real c2 = 30;
    Real tau = 10;
    Real D_A = 0.5;
    Real D_R = 0.5;
    Real sigma = 0.1;
    Real A0 = 0.5;
    Real S0 = 114;
    Real r0 = 6;
    Real m_repulsion = 10;
    int numCells = 1;

    int  initial_offset_x = 0;
    int initial_offset_y = 0;
} typedef Param;
#endif // PARAMS_CUH
