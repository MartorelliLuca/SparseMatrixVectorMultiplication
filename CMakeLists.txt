cmake_minimum_required(VERSION 3.10)
project(project)

set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)

set(CMAKE_CUDA_ARCHITECTURES 75)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Iinclude -O3 -march=native -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -fopenmp -lm")

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fopenmp")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --compiler-options=-fopenmp")

add_compile_options(-ffast-math -funroll-loops)

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP: ${OpenMP_CXX_FLAGS}")
endif()

# File sorgenti (C, C++ e CUDA)
set(SRC_FILES
    src/main.c
    src/implementations/csr.c
    src/implementations/hll.c
    src/implementations/operation.c
    src/utils/initialization.c
    src/utils/mmio.c
    src/utils/read.c
    src/utils/utils.c
    src/utils/write.c
    CUDA_src/cudacsr.cu
    CUDA_src/cudahll.cu
)

add_executable(project ${SRC_FILES})

# Linka OpenMP
target_link_libraries(project OpenMP::OpenMP_CXX)

# Abilita la compilazione separabile per CUDA
set_target_properties(project PROPERTIES CUDA_SEPARABLE_COMPILATION ON)