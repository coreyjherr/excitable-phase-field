set(CMAKE_CXX_COMPILER /usr/bin/g++-11)
set(CMAKE_C_COMPILER /usr/bin/gcc-11)
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-11)

cmake_minimum_required(VERSION 3.20)
project(stochastic_phase_field LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(SOURCES
    main.cu
    model.cu
)

add_executable(run_phase_field ${SOURCES})
set_property(TARGET run_phase_field PROPERTY CUDA_SEPARABLE_COMPILATION ON)


# Add flag to change nvcc ccbin
# set(CMAKE_CUDA_FLAGS "-ccbin /usr/local/
