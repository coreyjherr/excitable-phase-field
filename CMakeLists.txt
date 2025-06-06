set(CMAKE_CXX_COMPILER /usr/bin/g++-11)
set(CMAKE_C_COMPILER /usr/bin/gcc-11)
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-11)

cmake_minimum_required(VERSION 3.20)
project(stochastic_phase_field LANGUAGES CXX CUDA)

# this project is only verified for gcc 11.x
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message(STATUS "Using GCC compiler")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 11.0 OR CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0)
        # This must be filled in with the path to your gcc and g++ compilers version 11.x
    endif()
else()
    message(FATAL_ERROR "This project is only verified for GCC 11.x")
endif()

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
target_compile_options(stochastic_phase_field PRIVATE -O0)

# set the executable
add_executable(run_phase_field main.cpp)

target_link_libraries(run_phase_field PRIVATE stochastic_phase_field)
target_compile_options(run_phase_field PUBLIC -O0 )

# set compiler options for using various features of the code
option(CHEMO "Use chemotaxis" OFF)
option(USE_BACKLOG "Use gpu backlog for outputs" OFF)
option(SQUARE_CONFINEMENT "Use square confinement" OFF)

if(CHEMO)
    target_compile_definitions(stochastic_phase_field PRIVATE USE_CHEMO)
endif()

if(USE_BACKLOG)
    # print USE_BACKLOG to the console
    message(STATUS "[CONFIG] Using backlog for outputs")
    target_compile_definitions(stochastic_phase_field PRIVATE USE_DEVICE_BACKLOG)
else()
    # print USE_BACKLOG to the console
    message(STATUS "[CONFIG] Not using backlog for outputs")
endif()

if(SQUARE_CONFINEMENT)
    # print SQUARE_CONFINEMENT to the console
    message(STATUS "[CONFIG] Using square confinement")
    target_compile_definitions(stochastic_phase_field PRIVATE SQUARE_CONFINEMENT)
else()
    message(STATUS "[CONFIG] Using circular confinement")
    target_compile_definitions(stochastic_phase_field PRIVATE CIRCULAR_CONFINEMENT)
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
