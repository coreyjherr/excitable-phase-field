#include "include/params.hpp"
#include "include/toml.hpp"

std::vector<std::vector<Real>>
loadParamsFromTOML(const std::string& filepath, const Param& defaults)
{
  std::vector<std::vector<Real>> params;

  try {
    auto config = toml::parse_file(filepath);

    // Define a lambda to simplify reading arrays from the TOML file
    auto readArray = [&config](const std::string& key,
                               Real defaultValue) -> std::vector<Real> {
      std::vector<Real> values;
      if (config.contains(key)) {
        auto array = config[key].as_array();
        if (array->size() > 0 && array->get(0)->value_or(-1.0) == -1) {
          // Handle range case
          if (array->size() == 4) {
            Real start = array->get(1)->value_or(0.0);
            Real stop = array->get(2)->value_or(0.0);
            Real step = array->get(3)->value_or(1.0);
            values = range(start, stop, step);
          } else {
            // Invalid range specification, use default value
            values.push_back(defaultValue);
          }
        } else {
          // Regular array case
          for (const auto& value : *array) {
            values.push_back(value.value_or(defaultValue));
          }
        }
      } else {
        // If the key is missing, push back the default value
        values.push_back(defaultValue);
      }
      return values;
    };

    // Read each parameter as a vector, using the default value if the key is
    // missing
    params.push_back(readArray("N", defaults.N));
    params.push_back(readArray("T", defaults.T));
    params.push_back(readArray("blind", defaults.blind));
    params.push_back(readArray("dt", defaults.dt));
    params.push_back(readArray("dx", defaults.dx));

    params.push_back(readArray("gamma", defaults.gamma));
    params.push_back(readArray("eps2", defaults.eps2));
    params.push_back(readArray("B_s", defaults.B_s));
    params.push_back(readArray("xi", defaults.xi));
    params.push_back(readArray("alpha", defaults.alpha));
    params.push_back(readArray("k_a", defaults.k_a));
    params.push_back(readArray("K", defaults.K));
    params.push_back(readArray("b", defaults.b));
    params.push_back(readArray("A_t", defaults.A_t));
    params.push_back(readArray("d1", defaults.d1));
    params.push_back(readArray("d2", defaults.d2));
    params.push_back(readArray("c1", defaults.c1));
    params.push_back(readArray("c2", defaults.c2));
    params.push_back(readArray("tau", defaults.tau));
    params.push_back(readArray("D_A", defaults.D_A));
    params.push_back(readArray("D_R", defaults.D_R));
    params.push_back(readArray("sigma", defaults.sigma));
    params.push_back(readArray("A0", defaults.A0));
    params.push_back(readArray("S0", defaults.S0));
    params.push_back(readArray("r0", defaults.r0));
    params.push_back(readArray("m_repulsion", defaults.m_repulsion));
    params.push_back(readArray("numCells", defaults.numCells));

    params.push_back(readArray("R_wall", defaults.R_wall));
    params.push_back(readArray("m_adhesion", defaults.m_adhesion));
    params.push_back(readArray("m_wall_repulsion", defaults.m_wall_repulsion));
    params.push_back(readArray("packing_fraction", defaults.packing_fraction));
    params.push_back(readArray("chem_strength", defaults.chem_strength));
    params.push_back(readArray("chem_decay", defaults.chem_decay));
    params.push_back(readArray("Dchem", defaults.Dchem));
    params.push_back(readArray("Nchem", defaults.Nchem));
    params.push_back(readArray("chem_force", defaults.chem_force));
    params.push_back(readArray("delta_r", defaults.delta_r));

  } catch (const toml::parse_error& err) {
    std::cerr << "Error parsing TOML file: " << err.what() << std::endl;
  }

  return params;
}

Param
mapCombinationToParam(const std::vector<Real>& combination)
{
  Param params;

  // Ensure the combination has the correct number of elements
  if (combination.size() < 34) {
    std::cerr << "Error: Combination vector has insufficient elements."
              << std::endl;
    return params;
  }

  // Map the values to the Param struct
  params.N = static_cast<int>(combination[0]);
  params.T = static_cast<int>(combination[1]);
  params.blind = static_cast<int>(combination[2]);
  params.dt = combination[3];
  params.dx = combination[4];

  params.gamma = combination[5];
  params.eps2 = combination[6];
  params.B_s = combination[7];
  params.xi = combination[8];
  params.alpha = combination[9];
  params.k_a = combination[10];
  params.K = combination[11];
  params.b = combination[12];
  params.A_t = combination[13];
  params.d1 = combination[14];
  params.d2 = combination[15];
  params.c1 = combination[16];
  params.c2 = combination[17];
  params.tau = combination[18];
  params.D_A = combination[19];
  params.D_R = combination[20];
  params.sigma = combination[21];
  params.A0 = combination[22];
  params.S0 = combination[23];
  params.r0 = combination[24];
  params.m_repulsion = combination[25];
  params.numCells = static_cast<int>(combination[26]);

  params.R_wall = combination[27];
  params.m_adhesion = combination[28];
  params.m_wall_repulsion = combination[29];
  params.packing_fraction = combination[30];
  params.chem_strength = combination[31];
  params.chem_decay = combination[32];
  params.Dchem = combination[33];
  params.Nchem = static_cast<int>(combination[36]);
  params.chem_force = combination[35];
  params.delta_r = combination[36];

  return params;
}

bool
writeParamToTOML(const Param& params,
                 const std::string& filepath,
                 unsigned long long seed)
{
  try {
    // Create a toml::table and populate it with the Param values using an
    // initializer list
    auto tbl = toml::table{
      { "N",
        static_cast<int64_t>(params.N) }, // Cast to int64_t for TOML integer
      { "T", static_cast<int64_t>(params.T) },
      { "blind", static_cast<int64_t>(params.blind) },
      { "dt", params.dt }, // No cast needed for floating-point values
      { "dx", params.dx },

      { "gamma", params.gamma },
      { "eps2", params.eps2 },
      { "B_s", params.B_s },
      { "xi", params.xi },
      { "alpha", params.alpha },
      { "k_a", params.k_a },
      { "K", params.K },
      { "b", params.b },
      { "A_t", params.A_t },
      { "d1", params.d1 },
      { "d2", params.d2 },
      { "c1", params.c1 },
      { "c2", params.c2 },
      { "tau", params.tau },
      { "D_A", params.D_A },
      { "D_R", params.D_R },
      { "sigma", params.sigma },
      { "A0", params.A0 },
      { "S0", params.S0 },
      { "r0", params.r0 },
      { "m_repulsion", params.m_repulsion },
      { "numCells", static_cast<int64_t>(params.numCells) },

      { "R_wall", params.R_wall },
      { "m_adhesion", params.m_adhesion },
      { "m_wall_repulsion", params.m_wall_repulsion },
      { "packing_fraction", params.packing_fraction },
      { "chem_strength", params.chem_strength },
      { "chem_decay", params.chem_decay },
      { "Dchem", params.Dchem },
      { "Nchem", static_cast<int64_t>(params.Nchem) },
      { "chem_force", params.chem_force },
      { "delta_r", params.delta_r },
      { "seed", static_cast<int64_t>(seed) }
    };

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

void
printParam(const Param& param)
{
  std::cout << "N " << param.N << std::endl;
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
  std::cout << "Nchem " << param.Nchem << std::endl;
  std::cout << "chem_force " << param.chem_force << std::endl;
  std::cout << "delta_r " << param.delta_r << std::endl;
}
