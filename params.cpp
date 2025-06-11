#include "include/params.hpp"

#include "dependencies/toml.hpp"

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

std::vector<std::vector<Real>> loadParamsFromTOML(const std::string& filepath,
                                                  const Param& defaults) {
  std::vector<std::vector<Real>> params;

  try {
    auto config = toml::parse_file(filepath);

    // Define a lambda to simplify reading arrays or single values from the TOML
    // file
    auto readArray = [&config](const std::string& key, ParamType type,
                               Real defaultReal,
                               int defaultInt) -> std::vector<Real> {
      std::vector<Real> values;
      if (config.contains(key)) {
        auto node = config[key];
        if (node.is_array()) {
          auto array = node.as_array();
          if (array->size() > 0 && array->get(0)->value_or(-1.0) == -1) {
            // Handle range case
            if (array->size() == 4) {
              Real start = array->get(1)->value_or(0.0);
              Real stop = array->get(2)->value_or(0.0);
              Real step = array->get(3)->value_or(1.0);
              values = range(start, stop, step);
            } else {
              // Invalid range specification, use default value
              values.push_back(type == ParamType::Int ? defaultInt
                                                      : defaultReal);
            }
          } else {
            // Regular array case
            for (const auto& value : *array) {
              if (type == ParamType::Int) {
                values.push_back(value.value_or(defaultInt));
              } else {
                values.push_back(value.value_or(defaultReal));
              }
            }
          }
        } else if (type == ParamType::Int) {
          if (auto val = node.value<int64_t>()) {
            values.push_back(static_cast<Real>(*val));
          } else if (auto val = node.value<Real>()) {
            values.push_back(*val);
          } else {
            values.push_back(defaultInt);
          }
        } else {
          if (auto val = node.value<Real>()) {
            values.push_back(*val);
          } else if (auto val = node.value<int64_t>()) {
            values.push_back(static_cast<Real>(*val));
          } else {
            values.push_back(defaultReal);
          }
        }
      } else {
        values.push_back(type == ParamType::Int ? defaultInt : defaultReal);
      }
      return values;
    };

    // Use paramMetaList for all parameters
    for (const auto& meta : paramMetaList) {
      Real defaultReal = 0;
      int defaultInt = 0;
      if (meta.type == ParamType::Int) {
        defaultInt = *(const int*)((const char*)&defaults + meta.offset);
        params.push_back(readArray(meta.key, meta.type, 0, defaultInt));
      } else {
        defaultReal = *(const Real*)((const char*)&defaults + meta.offset);
        params.push_back(readArray(meta.key, meta.type, defaultReal, 0));
      }
    }
  } catch (const toml::parse_error& err) {
    std::cerr << "Error parsing TOML file: " << err.what() << std::endl;
  }

  return params;
}

Param mapCombinationToParam(const std::vector<Real>& combination) {
  Param params;

  // Ensure the combination has the correct number of elements
  if (combination.size() < sizeof(paramMetaList) / sizeof(paramMetaList[0])) {
    std::cerr << "Error: Combination vector has insufficient elements."
              << std::endl;
    return params;
  }

  // Map the values to the Param struct using paramMetaList
  for (size_t i = 0; i < sizeof(paramMetaList) / sizeof(paramMetaList[0]);
       ++i) {
    const auto& meta = paramMetaList[i];
    if (meta.type == ParamType::Int) {
      int* ptr = (int*)((char*)&params + meta.offset);
      *ptr = static_cast<int>(combination[i]);
    } else {
      Real* ptr = (Real*)((char*)&params + meta.offset);
      *ptr = combination[i];
    }
  }

  return params;
}

bool writeParamToTOML(const Param& params, const std::string& filepath) {
  try {
    toml::table tbl;
    for (const auto& meta : paramMetaList) {
      if (meta.type == ParamType::Int) {
        int value = *(const int*)((const char*)&params + meta.offset);
        tbl.insert_or_assign(meta.key, static_cast<int64_t>(value));
      } else {
        Real value = *(const Real*)((const char*)&params + meta.offset);
        tbl.insert_or_assign(meta.key, value);
      }
    }

    // Write the table to a TOML file
    std::ofstream ofs(filepath);
    if (!ofs) {
      std::cerr << "Error: Could not open file for writing." << std::endl;
      return false;
    }
    ofs << tbl;
    ofs.close();

    return true;
  } catch (const std::exception& e) {
    std::cerr << "Error writing TOML file: " << e.what() << std::endl;
    return false;
  }
}

void printParam(const Param& param) {
  std::cout << "N " << param.gridN << std::endl;
  std::cout << "T " << param.T << std::endl;
  std::cout << "blind " << param.blind << std::endl;
  std::cout << "dt " << param.dt << std::endl;
  std::cout << "dx " << param.dx << std::endl;
  std::cout << "gamma " << param.gamma << std::endl;
  std::cout << "eps2 " << param.eps2 << std::endl;
  std::cout << "B_s " << param.B_s << std::endl;
  std::cout << "xi " << param.xi << std::endl;
  std::cout << "alpha " << param.alpha << std::endl;
  std::cout << "k_a " << param.k_a << std::endl;
  std::cout << "K " << param.K << std::endl;
  std::cout << "b " << param.b << std::endl;
  std::cout << "A_t " << param.A_t << std::endl;
  std::cout << "d1 " << param.d1 << std::endl;
  std::cout << "d2 " << param.d2 << std::endl;
  std::cout << "c1 " << param.c1 << std::endl;
  std::cout << "c2 " << param.c2 << std::endl;
  std::cout << "tau " << param.tau << std::endl;
  std::cout << "D_A " << param.D_A << std::endl;
  std::cout << "D_R " << param.D_R << std::endl;
  std::cout << "sigma " << param.sigma << std::endl;
  std::cout << "A0 " << param.A0 << std::endl;
  std::cout << "S0 " << param.S0 << std::endl;
  std::cout << "r0 " << param.r0 << std::endl;
  std::cout << "m_repulsion " << param.m_repulsion << std::endl;
  std::cout << "numCells " << param.numCells << std::endl;
  std::cout << "R_wall " << param.R_wall << std::endl;
  std::cout << "m_adhesion " << param.m_adhesion << std::endl;
  std::cout << "m_wall_repulsion " << param.m_wall_repulsion << std::endl;
  std::cout << "packing_fraction " << param.packing_fraction << std::endl;
  std::cout << "chem_strength " << param.chem_strength << std::endl;
  std::cout << "chem_decay " << param.chem_decay << std::endl;
  std::cout << "Dchem " << param.Dchem << std::endl;
  std::cout << "gridNchem " << param.Nchem << std::endl;
  std::cout << "chem_force " << param.chem_force << std::endl;
  std::cout << "delta_r " << param.delta_r << std::endl;
}
