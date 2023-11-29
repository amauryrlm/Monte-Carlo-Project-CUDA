cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(hello LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)

# Set the CUDA architecture to 75 for the Tesla T4 GPU
set(CMAKE_CUDA_ARCHITECTURES 75)

set(SRC ${CMAKE_SOURCE_DIR})
set(INC ${CMAKE_SOURCE_DIR}/inc)

include_directories(${INC})

# Find CUDA package
find_package(CUDA REQUIRED)

# Add your main CUDA executable
add_executable(main hello.cu)
add_executable(test testing.cu)

# Link the cuRAND library to your main CUDA executable
target_link_libraries(main ${CUDA_LIBRARIES} ${CUDA_curand_LIBRARY})
target_link_libraries(test ${CUDA_LIBRARIES} ${CUDA_curand_LIBRARY})

# Your other executable
add_executable(csv traj_to_csv.cpp)
