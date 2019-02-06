cmake_minimum_required (VERSION 3.11)

project (nbody LANGUAGES CXX CUDA)

enable_language (CUDA)

# currently relies on user-supplied definitions for
# GLEW and GLFW path
set (gl_paths GLEW_INCLUDE_DIR GLEW_LIBRARY GLFW_INCLUDE_DIR GLFW_LIBRARY)
foreach (glvar IN LISTS gl_paths)
	if (NOT ${glvar})
		message (FATAL_ERROR "${glvar} not defined; run CMake with -D flag")
	endif ()
endforeach ()

# Add header file directories
list (APPEND include_dirs
	src
	src/phys
	src/util
	src/tests
	${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
	${GLEW_INCLUDE_DIR}
	${GLFW_INCLUDE_DIR}
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
target_link_libraries (nbody PRIVATE ${GLEW_LIBRARY}\\glew32s.lib ${GLFW_LIBRARY}\\glfw3.lib opengl32.lib)
