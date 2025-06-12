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
  Real chem_strength = 1.0;
  Real chem_decay = 0.1;
  Real Dchem = 0.5;
  int Nchem = 1024;
  Real chem_force = 1;
  Real delta_r = 0.;
  int seed = 0;
} typedef Param;

enum class ParamType { Int, Real };

// Parameter metadata: key, pointer to member in Param
struct ParamMeta {
  const char* key;
  ParamType type;
  size_t offset;
};

#define PARAM_OFFSET(member) offsetof(Param, member)

constexpr ParamMeta paramMetaList[] = {
    {"gridN", ParamType::Int, PARAM_OFFSET(gridN)},
    {"T", ParamType::Int, PARAM_OFFSET(T)},
    {"blind", ParamType::Int, PARAM_OFFSET(blind)},
    {"dt", ParamType::Real, PARAM_OFFSET(dt)},
    {"dx", ParamType::Real, PARAM_OFFSET(dx)},
    {"gamma", ParamType::Real, PARAM_OFFSET(gamma)},
    {"eps2", ParamType::Real, PARAM_OFFSET(eps2)},
    {"B_s", ParamType::Real, PARAM_OFFSET(B_s)},
    {"xi", ParamType::Real, PARAM_OFFSET(xi)},
    {"alpha", ParamType::Real, PARAM_OFFSET(alpha)},
    {"k_a", ParamType::Real, PARAM_OFFSET(k_a)},
    {"K", ParamType::Real, PARAM_OFFSET(K)},
    {"b", ParamType::Real, PARAM_OFFSET(b)},
    {"A_t", ParamType::Real, PARAM_OFFSET(A_t)},
    {"d1", ParamType::Real, PARAM_OFFSET(d1)},
    {"d2", ParamType::Real, PARAM_OFFSET(d2)},
    {"c1", ParamType::Real, PARAM_OFFSET(c1)},
    {"c2", ParamType::Real, PARAM_OFFSET(c2)},
    {"tau", ParamType::Real, PARAM_OFFSET(tau)},
    {"D_A", ParamType::Real, PARAM_OFFSET(D_A)},
    {"D_R", ParamType::Real, PARAM_OFFSET(D_R)},
    {"sigma", ParamType::Real, PARAM_OFFSET(sigma)},
    {"A0", ParamType::Real, PARAM_OFFSET(A0)},
    {"S0", ParamType::Real, PARAM_OFFSET(S0)},
    {"r0", ParamType::Real, PARAM_OFFSET(r0)},
    {"m_repulsion", ParamType::Real, PARAM_OFFSET(m_repulsion)},
    {"numCells", ParamType::Int, PARAM_OFFSET(numCells)},
    {"R_wall", ParamType::Real, PARAM_OFFSET(R_wall)},
    {"m_adhesion", ParamType::Real, PARAM_OFFSET(m_adhesion)},
    {"m_wall_repulsion", ParamType::Real, PARAM_OFFSET(m_wall_repulsion)},
    {"packing_fraction", ParamType::Real, PARAM_OFFSET(packing_fraction)},
    {"chem_strength", ParamType::Real, PARAM_OFFSET(chem_strength)},
    {"chem_decay", ParamType::Real, PARAM_OFFSET(chem_decay)},
    {"Dchem", ParamType::Real, PARAM_OFFSET(Dchem)},
    {"Nchem", ParamType::Int, PARAM_OFFSET(Nchem)},
    {"chem_force", ParamType::Real, PARAM_OFFSET(chem_force)},
    {"delta_r", ParamType::Real, PARAM_OFFSET(delta_r)},
    {"seed", ParamType::Int, PARAM_OFFSET(seed)}};

// Compile-time check: paramMetaList must have exactly as many entries as there
// are fields in Param. If float is used than int and Real are the same size,
// so we can get deterministic behavior. If double is used, then we need to
// allow for more entries in paramMetaList than fields in Param, since some
// fields may be Real and some may be int
#ifdef USE_FLOAT
static_assert(
    sizeof(paramMetaList) / sizeof(ParamMeta) == sizeof(Param) / sizeof(Real),
    "paramMetaList must have exactly as many entries as Param has fields. "
    "Update paramMetaList and PARAM_FIELD_COUNT if you add fields to Param.");
#else
static_assert(
    sizeof(paramMetaList) / sizeof(ParamMeta) >= sizeof(Param) / sizeof(Real),
    "paramMetaList must have exactly as many entries as Param has fields. "
    "Update paramMetaList and PARAM_FIELD_COUNT if you add fields to Param.");
#endif

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
bool writeParamToTOML(const Param& param, const std::string& filepath);
void printParam(const Param& param);

void checkParamMetaListLayout();

// Function to write a Param struct to a TOML file
#endif  // PARAMS_CUH
