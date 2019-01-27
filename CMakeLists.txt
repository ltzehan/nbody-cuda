cmake_minimum_required (VERSION 3.11)

project (nbody LANGUAGES CXX CUDA)

enable_language (CUDA)

# Add header file directories
list (APPEND include_dirs
	src
	src/phys
	src/util
	src/tests
	${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Find all source files
file (GLOB_RECURSE src_files
	*.h
	*.cpp
	*.cu
)

# Executable for simulation
add_executable (nbody ${src_files})
target_include_directories (nbody PRIVATE ${include_dirs})
