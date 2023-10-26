cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(hello LANGUAGES CXX CUDA)
enable_language(CUDA)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CUDA_ARCHITECTURES 89)
set(SRC ${CMAKE_SOURCE_DIR})
set(INC ${CMAKE_SOURCE_DIR}/inc)

include_directories(${INC})

add_executable(main hello.cu)
