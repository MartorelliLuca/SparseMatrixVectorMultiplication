cmake_minimum_required(VERSION 3.10)
project(SparseMatrixVectorMultiplication C)

# Set compiler
set(CMAKE_C_COMPILER gcc)

# Set compilation flags
set(CMAKE_C_FLAGS "-Iinclude -fopenmp")

# Set linking flags
set(CMAKE_EXE_LINKER_FLAGS "-fopenmp")

add_compile_options(-O3)
add_compile_options(-march=native)

set(CMAKE_CXX_FLAGC "-O3")

add_compile_options(-ffast-math)
add_compile_options(-funroll-loops)

# Source files
set(SRC_FILES
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

# Create the executable
add_executable(project ${SRC_FILES})

# Link OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_C_FOUND)
    target_link_libraries(project OpenMP::OpenMP_C)
endif()

# Custom run target (clean target removed)
add_custom_target(run COMMAND ./project DEPENDS project)
   
