set(CMAKE_CXX_COMPILER /usr/bin/g++-11)
set(CMAKE_C_COMPILER /usr/bin/gcc-11)
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-11)

cmake_minimum_required(VERSION 3.20)
project(stochastic_phase_field LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})

set(SOURCES
    model.cu
    include/model.h
    kernels.cu
    include/kernels.cuh
    params.cpp
    include/params.hpp
)

# make a library out of the source files
add_library(stochastic_phase_field STATIC ${SOURCES})

set_target_properties(stochastic_phase_field PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 75
)

target_link_libraries(stochastic_phase_field PRIVATE -lcublas -lcurand)
target_compile_options(stochastic_phase_field PRIVATE -O3)

# set the executable 
add_executable(run_phase_field main.cpp)

target_link_libraries(run_phase_field PRIVATE stochastic_phase_field)
target_compile_options(run_phase_field PUBLIC -O3)

# set compiler options for using various features of the code
option(CHEMO "Use chemotaxis" OFF)
option(USE_BACKLOG "Use gpu backlog for outputs" ON)

if(CHEMO)
    target_compile_definitions(stochastic_phase_field PRIVATE USE_CHEMO)
endif()

if(USE_BACKLOG)
    # print USE_BACKLOG to the console
    message(STATUS "Using backlog for outputs")
    target_compile_definitions(stochastic_phase_field PRIVATE USE_DEVICE_BACKLOG)
else()
    # print USE_BACKLOG to the console
    message(STATUS "Not using backlog for outputs")
endif()

option(BUILD_TESTS "Build tests" OFF)
if (BUILD_TESTS)
    Include(FetchContent)

    FetchContent_Declare(
      Catch2
      GIT_REPOSITORY https://github.com/catchorg/Catch2.git
      GIT_TAG        v3.4.0 # or a later release
    )

    FetchContent_MakeAvailable(Catch2)

    set(SOURCES
        ./tests/phase_field_initialization.cpp
    )

    add_executable(run_tests ${SOURCES})
    # make test an optional target
    set_target_properties(run_tests PROPERTIES EXCLUDE_FROM_ALL TRUE)
    # choose the location of the test executable
    set_target_properties(run_tests PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    )
    target_include_directories(run_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(run_tests PRIVATE Catch2::Catch2WithMain stochastic_phase_field)
endif()
