#
# dependencies.cmake:
#
# Sets Dependencies:
#   Qt4
#   Qwt5
#   FFTW
#   CFitsio
#   Boost
#   OpenMP
#   MPI
#   CppUnit
#   CBLAS (either ATLAS or MKL)
#   LAPACK (either ATLAS or MKL)
#   CUDA
#

find_package(Qt4 COMPONENTS QtCore QtGui QtOpenGL QtXml QtTest REQUIRED)
find_package(Qwt5 REQUIRED)
find_package(FFTW3 REQUIRED)
find_package(CFitsio REQUIRED)
find_package(Boost REQUIRED) # Which components?
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED) # Maybe not required (likewise for CUDA)
find_package(CppUnit REQUIRED)

# === Find CBLAS and LAPACK from MKL if availiable, otherwise elsewhere.
if(NOT DEFINED oskar_mkl)
    set(pelican_mkl true)
endif(NOT DEFINED oskar_mkl)

if(oskar_mkl)
    find_package(MKL QUIET)
endif(oskar_mkl)

if (MKL_FOUND)
    add_definitions(-DUSING_MKL)
    set(oskar_math_libs ${MKL_LIBRARIES})
    set(oskar_mkl true)
    include_directories(${MKL_INCLUDE_DIR})
    message(STATUS "FoundMKL: ${oskar_math_libs}")
else (MKL_FOUND)
    find_package(CBLAS REQUIRED)
    find_package(LAPACK REQUIRED)
    set(oskar_math_libs ${LAPACK_LIBRARIES} ${CBLAS_LIBRARIES})
endif (MKL_FOUND)


# == Cuda
set(CUDA_SDK_ROOT_DIR /usr/local/cudaSDK/) # <== might need to change this
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_BUILD_EMULATION OFF) # should be default off anyway.
#set(CUDA_NVCC_FLAGS --compiler-options;-Wall;--ptxas-options=-v)
find_package(CUDA 2.1 REQUIRED)
set(CUDA_NVCC_FLAGS --compiler-options;-Wall;--compiler-options;-O2;--compiler-options;-pedantic)

# === Set global project include directories.
include_directories(
    ${oskar-lib_SOURCE_DIR}
    ${QT_INCLUDE_DIR}
    ${CFITSIO_INCLUDE_DIR}
)
