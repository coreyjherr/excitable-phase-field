#ifndef PARAMS_CUH
#define PARAMS_CUH

typedef float Real;
typedef float2 Real2;
#define FLOAT

#include <fstream>

struct Param { 
    int N = 256;
    int T = 15000;
    int blind = 400;
    Real dt = 0.001;
    Real dx = 0.2;

    Real gamma = 2;
    Real eps2 = 4;
    Real B_s = 10;
    Real xi = 3;
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

    int  initial_offset_x = 40;
    int initial_offset_y = 0;
    Real R_wall = 20;
    Real m_adhesion = 1;
} typedef Param;

inline void writeParamFile(std::string filename, Param param) {
    std::ofstream fout(filename.c_str(), std::ios::out);
    fout << "N " << param.N << std::endl;
    fout << "T " << param.T << std::endl;
    fout << "blind " << param.blind << std::endl;
    fout << "dt " << param.dt << std::endl;
    fout << "dx " << param.dx << std::endl;
    fout << "gamma " << param.gamma << std::endl;
    fout << "eps2 " << param.eps2 << std::endl;
    fout << "B_s " << param.B_s << std::endl;
    fout << "xi " << param.xi << std::endl;
    fout << "alpha " << param.alpha << std::endl;
    fout << "k_a " << param.k_a << std::endl;
    fout << "K " << param.K << std::endl;
    fout << "b " << param.b << std::endl;
    fout << "A_t " << param.A_t << std::endl;
    fout << "d1 " << param.d1 << std::endl;
    fout << "d2 " << param.d2 << std::endl;
    fout << "c1 " << param.c1 << std::endl;
    fout << "c2 " << param.c2 << std::endl;
    fout << "tau " << param.tau << std::endl;
    fout << "D_A " << param.D_A << std::endl;
    fout << "D_R " << param.D_R << std::endl;
    fout << "sigma " << param.sigma << std::endl;
    fout << "A0 " << param.A0 << std::endl;
    fout << "S0 " << param.S0 << std::endl;
    fout << "r0 " << param.r0 << std::endl;
    fout << "m_repulsion " << param.m_repulsion << std::endl;
    fout << "numCells " << param.numCells << std::endl;
    fout << "initial_offset_x " << param.initial_offset_x << std::endl;
    fout << "initial_offset_y " << param.initial_offset_y << std::endl;
    fout << "R_wall " << param.R_wall << std::endl;
    fout << "m_adhesion " << param.m_adhesion << std::endl;
    fout.close();
}

#endif // PARAMS_CUH
