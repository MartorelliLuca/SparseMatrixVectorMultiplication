cmake_minimum_required(VERSION 3.18)
project(SparseMatrixVectorMultiplication)

# Set compiler for C e C++
set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)

# Flag di compilazione per il C
set(CMAKE_C_FLAGS "-Iinclude -fopenmp")

# Linking flags
set(CMAKE_EXE_LINKER_FLAGS "-fopenmp")

# Flag di ottimizzazione
add_compile_options(-O3 -march=native -ffast-math -funroll-loops)

# Abilita il supporto CUDA
enable_language(CUDA)

# Imposta i flag per il compilatore CUDA
set(CMAKE_CUDA_FLAGS "-O3 -arch=sm_60 -Xcompiler -fopenmp -lineinfo")

# File sorgenti C
set(C_SRC_FILES
    src/utils/mmio.c
    src/utils/read.c
    src/utils/write.c
    src/utils/utils.c
    src/main.c
    src/implementations/csr.c
    src/implementations/hll.c
    src/implementations/operation.c
    src/utils/initialization.c
)

# File sorgenti CUDA
file(GLOB CUDA_SRC_FILES CUDA/src/*.cu)

# Creazione dell'eseguibile
add_executable(project ${C_SRC_FILES} ${CUDA_SRC_FILES})

# Link a OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_C_FOUND)
    target_link_libraries(project OpenMP::OpenMP_C)
endif()

# Link a CUDA Toolkit (librerie base)
find_package(CUDAToolkit REQUIRED)
target_link_libraries(project CUDA::cudart)

# Target di esecuzione personalizzato
add_custom_target(run COMMAND ./project DEPENDS project)
