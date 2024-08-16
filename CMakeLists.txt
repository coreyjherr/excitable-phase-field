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
    kernels.cu
)
# set the executable directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY /home/cherr/workspace/stochastic_phase_field_model)
add_executable(run_phase_field ${SOURCES})
target_link_libraries(run_phase_field PRIVATE -lcublas -lcurand)
set_property(TARGET run_phase_field PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(run_phase_field PUBLIC -O3)


# Add flag to change nvcc ccbin
# set(CMAKE_CUDA_FLAGS "-ccbin /usr/local/
