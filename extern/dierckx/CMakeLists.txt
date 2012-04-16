#
# src/extern/dierckx/CMakeLists.txt
#

project(DIERCKX Fortran C CXX)
cmake_minimum_required(VERSION 2.8)

# ==== Append the src/cmake directory to the module path.
list(INSERT CMAKE_MODULE_PATH 0 ${DIERCKX_SOURCE_DIR}/cmake)

include_directories(${DIERCKX_SOURCE_DIR})

find_package(CppUnit)
find_package(OSKAR2)

set(dierckx_SRC
    src/bispev.f
    src/fpback.f
    src/fpbisp.f
    src/fpbspl.f
    src/fpdisc.f
    src/fpgivs.f
    src/fporde.f
    src/fprank.f
    src/fprati.f
    src/fprota.f
    src/fpsurf.f
    src/surfit.f
)

add_library(dierckx SHARED ${dierckx_SRC})

enable_testing()
add_subdirectory(test)
