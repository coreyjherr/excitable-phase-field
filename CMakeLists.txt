set(CMAKE_CXX_COMPILER /usr/bin/g++-11)
set(CMAKE_C_COMPILER /usr/bin/gcc-11)
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-11)

cmake_minimum_required(VERSION 3.20)
project(stochastic_phase_field LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#
set(SOURCES
    main.cu
    model.cu
    kernels.cu
    params.cpp
)

# set the executable directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})
add_executable(run_phase_field ${SOURCES})

target_link_libraries(run_phase_field PRIVATE -lcublas -lcurand)
set_property(TARGET run_phase_field PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET run_phase_field PROPERTY CUDA_ARCHITECTURES 75)
target_compile_options(run_phase_field PUBLIC -O3)

option(CHEMO "Use chemotaxis" OFF)
option(USE_BACKLOG "Use gpu backlog for outputs" OFF)

if(CHEMO)
    target_compile_definitions(run_phase_field PRIVATE USE_CHEMO)
endif()

if(USE_BACKLOG)
    target_compile_definitions(run_phase_field PRIVATE USE_DEVICE_BACKLOG)
endif()
