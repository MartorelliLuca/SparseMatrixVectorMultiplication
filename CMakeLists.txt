cmake_minimum_required(VERSION 3.10)
project(progetto LANGUAGES C CXX CUDA)

# Specifica i compilatori per C e C++
set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CUDA_COMPILER nvcc)

# Architetture CUDA (Quadro RTX 5000 è 7.5)
set(CMAKE_CUDA_ARCHITECTURES 75)

# Impostazioni di ottimizzazione e flag per OpenMP (C e C++)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Iinclude -O3 -march=native -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -fopenmp -lm")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xcompiler -fopenmp")

# Linker flags
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fopenmp")

# Opzioni aggiuntive
add_compile_options(-ffast-math -funroll-loops)

# Trova OpenMP
find_package(OpenMP REQUIRED)
if (OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP: ${OpenMP_CXX_FLAGS}")
endif()

# File sorgenti
set(SRC_FILES
    src/implementations/csr.c
    src/implementations/hll.c
    src/implementations/operation.c
    src/utils/initialization.c
    src/utils/mmio.c
    src/utils/read.c
    src/utils/utils.c
    src/utils/write.c
    CUDA/src/cudacsr.cu
    CUDA/src/cudahll.cu
    src/main.c
)

# Creazione dell'eseguibile
add_executable(project ${SRC_FILES})

# Link delle librerie
find_package(CUDAToolkit REQUIRED)
target_link_libraries(project OpenMP::OpenMP_CXX CUDA::cudart)

# Abilita la compilazione separabile per CUDA
set_target_properties(project PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
