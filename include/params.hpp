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
#ifndef PARAMS_CUH
#define PARAMS_CUH
#define USE_FLOAT

#ifdef USE_FLOAT
using Real = float;
#else
using Real = double;
#endif

#include <fstream>
#include <iostream>
#include <vector>

struct Param {
  int gridN = 256;
  int T = 1000 / 0.001;
  int blind = 25;
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
  Real m_repulsion = 30;
  int numCells = 1;

  Real R_wall = 0.;
  Real m_adhesion = 0.;
  Real m_wall_repulsion = 30;
  Real packing_fraction = 0.85;
  Real chem_strength = 0.001;
  Real chem_decay = 0.1;
  Real Dchem = 0.5;
  int Nchem = 1024;
  Real chem_force = 1;
  Real delta_r = 0.;
} typedef Param;

inline void writeParamFile(std::string filename,
                           Param param,
                           unsigned long long seed) {
  std::ofstream fout(filename.c_str(), std::ios::out);
  fout << "N " << param.gridN << std::endl;
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
  fout << "R_wall " << param.R_wall << std::endl;
  fout << "m_adhesion " << param.m_adhesion << std::endl;
  fout << "m_wall_repulsion " << param.m_wall_repulsion << std::endl;
  fout << "seed " << seed << std::endl;
  fout << "packing_fraction " << param.packing_fraction << std::endl;
  fout << "chem_strength " << param.chem_strength << std::endl;
  fout << "chem_decay " << param.chem_decay << std::endl;
  fout << "Dchem " << param.Dchem << std::endl;
  fout << "Nchem " << param.Nchem << std::endl;
  fout << "chem_force " << param.chem_force << std::endl;
  fout << "delta_r " << param.delta_r << std::endl;
  fout.close();
}

inline std::vector<Real> range(Real start, Real stop, Real step) {
  std::vector<Real> result;
  for (Real i = start; i < stop; i += step) {
    result.push_back(i);
  }
  return result;
}

inline std::vector<std::vector<Real>> getAllCombinations(
    std::vector<std::vector<Real>> params) {
  // get all the combinations from all lists of parameters
  std::vector<std::vector<Real>> result;
  if (params.size() == 0)
    return result;
  if (params.size() == 1) {
    for (auto p : params[0]) {
      result.push_back({p});
    }
    return result;
  }
  std::vector<std::vector<Real>> rest = getAllCombinations(
      std::vector<std::vector<Real>>(params.begin() + 1, params.end()));
  for (auto p : params[0]) {
    for (auto r : rest) {
      std::vector<Real> temp = {p};
      temp.insert(temp.end(), r.begin(), r.end());
      result.push_back(temp);
    }
  }
  return result;
}

// Function to load parameters from TOML file
std::vector<std::vector<Real>> loadParamsFromTOML(
    const std::string& filepath,
    const Param& defaults = Param());
Param mapCombinationToParam(const std::vector<Real>& combination);
bool writeParamToTOML(const Param& param,
                      const std::string& filepath,
                      unsigned long long seed);
void printParam(const Param& param);

// Function to write a Param struct to a TOML file
#endif  // PARAMS_CUH
