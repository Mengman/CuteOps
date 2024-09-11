cmake_minimum_required(VERSION 3.28)
project(CuteOps VERSION 1.0 LANGUAGES CXX CUDA)
set(CMAKE_DEBUG_POSTFIX d)

add_library(cuteops_compiler_flags INTERFACE)
target_compile_features(cuteops_compiler_flags INTERFACE cxx_std_11)


