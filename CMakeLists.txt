#
# /src (top level CMakeLists.txt)
#

# === Set the project name and specify the required CMAKE version.
project(oskar-lib CXX C)
cmake_minimum_required(VERSION 2.6.4 FATAL_ERROR)

# === Project version.
# (todo)

# === Append the src/cmake directory to the module path.
list(INSERT CMAKE_MODULE_PATH 0 ${oskar-lib_SOURCE_DIR}/cmake)

# === Set dependencies.
include(dependencies)

# === Set compiler options.
include(compilerOptions)

# === Recurse into sub-diretories.
#add_subdirectory(data)
#add_subdirectory(utility)
#add_subdirectory(math)
add_subdirectory(cuda)
#add_subdirectory(mpi)
#add_subdirectory(widgets)
#add_subdirectory(apps)
