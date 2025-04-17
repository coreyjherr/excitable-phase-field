# Stochastic Phase Field Model

This project implements a stochastic phase field model for simulating physical systems with excitable dynamics. 

## Features

- GPU-accelerated simulations using CUDA.
- Configurable parameters via TOML files.
- Multi-threaded execution for running multiple simulations in parallel.
- Outputs include dynamics, condensed fields, and full simulation data.

## Prerequisites

- **CUDA Toolkit** (version 10.0 or later)
- **CMake** (version 3.10 or later)
- **A C++ Compiler** with C++11 support
- **Python** (optional, for post-processing scripts)

## Compilation

To compile the project, follow these steps:

1. Run CMake to configure the project:
   ```bash
   cmake -Bbuild . 
   ```

2. Build the project:
   ```bash
   cmake --build build
   ```

The compiled executable will be located in the base directory.

## Running the Program

The program accepts a configuration file in TOML format and optional arguments for output location and type.

### Usage

```bash
./run_phase_field <config_file> [output_directory] [output_type]
```

- `<config_file>`: Path to the TOML configuration file.
- `[output_directory]`: (Optional) Directory to store results. Defaults to `./results/`.
- `[output_type]`: (Optional) Output type:
  - `0`: Full output
  - `1`: Condensed output
  - `2`: Dynamics output
  - `3`: Parameters output
  - `4`: No output

### Example

```bash
./run_phase_field config.toml ./results/ 1
```

This runs the simulation using `config.toml`, stores results in `./results/`, and generates condensed output.

## Configuration

The simulation parameters are specified in a TOML file. If you want to run multiple simulations you can include multiple values for the parameters. The code will run every combination of all parameters. An example parameter file that runs 4 simulations with different parameters is shown below:

```toml
numCells = [64]
m_adhesion = [15, 30]
alpha = [7.0, 7.5]
xi = [5]
gamma = [2.0]
```

Refer to the `Param` struct in `include/params.hpp` for a full list of configurable parameters.

## Running Tests

To run tests, ensure the project is compiled with testing enabled:

1. Configure the project with testing:
   ```bash
   cmake -DBUILD_TESTS=ON ..
   ```

2. Build the tests:
   ```bash
   cmake --build build --target run_tests
   ```

3. Run the tests:
   ```bash
   ./build/run_tests
   ```

## Post-Processing

The `scripts/` directory contains Python scripts for analyzing and visualizing simulation results. For example:

```bash
python scripts/create_video.py ./results/ -v
```

This generates a video from the simulation output.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
