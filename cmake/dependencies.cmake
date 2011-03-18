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
find_package(Qwt5 QUIET)
find_package(FFTW3 REQUIRED)
find_package(CFitsio REQUIRED)
find_package(Boost REQUIRED)
find_package(OpenMP REQUIRED)
find_package(MPI QUIET)
find_package(CppUnit REQUIRED)
find_package(CUDA 2.1 QUIET)
find_package(CasaCore QUIET)

if (NOT CUDA_FOUND)
    message("*****************************************************************")
    message("** WARNING: CUDA not found: "
        "Unable to build OSKAR CUDA library!")
    message("*****************************************************************")
endif()

if (NOT CASACORE_FOUND)
    message("*****************************************************************")
    message("** WARNING: CasaCore not found: "
        "Unable to build OSKAR measurement set library!")
    message("*****************************************************************")
endif()


# === Find CBLAS and LAPACK from MKL if availiable, otherwise elsewhere.
if(NOT DEFINED oskar_mkl)
    set(pelican_mkl true)
endif(NOT DEFINED oskar_mkl)

if(oskar_mkl)
    find_package(MKL QUIET)
endif()

if(MKL_FOUND)
    add_definitions(-DUSING_MKL)
    set(oskar_math_libs ${MKL_LIBRARIES})
    set(oskar_mkl true)
    include_directories(${MKL_INCLUDE_DIR})
    message(STATUS "FoundMKL: ${oskar_math_libs}")
else()
    find_package(CBLAS REQUIRED)
    find_package(LAPACK REQUIRED)
    set(oskar_math_libs ${LAPACK_LIBRARIES} ${CBLAS_LIBRARIES})
endif()


# == Cuda
#set(CUDA_SDK_ROOT_DIR /usr/local/cudaSDK/) # <== might need to change this
if(CUDA_FOUND)
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
    set(CUDA_BUILD_EMULATION OFF) # should be default off anyway.
    if(CMAKE_BUILD_TYPE MATCHES RELEASE|[Rr]elease)
        set(CUDA_NVCC_FLAGS --compiler-options;-Wall;--compiler-options;-O2)
    else()
        set(CUDA_NVCC_FLAGS --compiler-options;-Wall;--compiler-options;-O0;--compiler-options;-g)
    endif()
endif()


# === Set global project include directories.
include_directories(
    ${oskar-lib_SOURCE_DIR}
    ${QT_INCLUDE_DIR}
    ${CFITSIO_INCLUDE_DIR}
)
