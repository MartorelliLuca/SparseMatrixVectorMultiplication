cmake_minimum_required(VERSION 3.7)
project(SparseMatrixVectorMultiplication)

set(CC 75)

find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

if(OpenMP_FOUND)
    message(STATUS "OpenMP trovato")
endif()

if(CUDAToolkit_FOUND)
    message(STATUS "CUDA trovato")
endif()

# File sorgenti del progetto
set(SOURCE_FILES
    src/main.c
    src/utils/initialization.c
    src/utils/mmio.c
    src/utils/read.c
    src/utils/utils.c
    src/utils/write.c
    src/implementations/hll.c
    src/implementations/csr.c
    src/implementations/operation.c
)

# File CUDA
set(CUDA_SOURCES
    CUDA/src/cudacsr.cu
    CUDA/src/cudahll.cu
)

# Kernel CUDA
set(CUDA_KERNELS
    CUDA/kernel/csr/cudakernel1.cuh
    CUDA/kernel/hll/cuda_hll_kernel_v1.cuh
)

add_executable(project ${SOURCE_FILES} ${CUDA_SOURCES} ${CUDA_KERNELS})

# Collegamento delle librerie necessarie
target_link_libraries(project PUBLIC OpenMP::OpenMP_C CUDA::cudart)

# Opzioni di compilazione
set(CMAKE_CXX_FLAGS "-O3")
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -DSM_${CC} -arch=sm_${CC} -lineinfo -Xcompiler=${CMAKE_CXX_FLAGS} -Xptxas=-v")

set_target_properties(project PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    LINKER_LANGUAGE CXX
)
