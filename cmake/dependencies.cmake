#
# dependencies.cmake:
#
# Dependencies:
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

# note: 4.4.3 is just chosen for illustration at the moment.
find_package(Qt4 4.4.3 COMPONENTS QtCore QtGui QtXml QtOpenGL QtTest QUIET)

if (QT_FOUND)
    if (WIN32)
        set(QWT_INCLUDES ${CMAKE_SOURCE_DIR}/../include/qwt-5.2.2/)
        set(QWT_LIBRARY_DIR ${CMAKE_SOURCE_DIR}/../lib/qwt-5.2.2/)
    endif ()
    find_package(Qwt5 QUIET)
    #message("************* ${Qwt5_Qt4_LIBRARY}")
    #message("************* ${Qwt5_INCLUDE_DIR}")
endif ()

find_package(OpenMP QUIET)
find_package(CUDA 4.0 REQUIRED)
find_package(MPI QUIET)

if (WIN32)
    set(CPPUNIT_INCLUDES ${CMAKE_SOURCE_DIR}/../include/cppunit-1.12.1/)
    set(CPPUNIT_LIB_DIR ${CMAKE_SOURCE_DIR}/../lib/cppunit-1.12.1/)
endif()
find_package(CppUnit QUIET)
#message("============ ${CPPUNIT_INCLUDE_DIR}")

find_package(FFTW3 QUIET)
find_package(Boost QUIET)

find_package(CasaCore QUIET)
find_package(CFitsio QUIET)

set(OSKAR_MATH_LIBS_FOUND false)
if (NOT DEFINED oskar_mkl)
    set(pelican_mkl true)
endif (NOT DEFINED oskar_mkl)
if (oskar_mkl)
    find_package(MKL QUIET)
endif ()
if (MKL_FOUND)
    set(OSKAR_MATH_LIBS_FOUND true)
    add_definitions(-DUSING_MKL)
    set(oskar_math_libs ${MKL_LIBRARIES})
    set(oskar_mkl true)
    include_directories(${MKL_INCLUDE_DIR})
    message(STATUS "FoundMKL: ${oskar_math_libs}")
else ()
    find_package(CBLAS QUIET)
    find_package(LAPACK QUIET)
    set(oskar_math_libs ${LAPACK_LIBRARIES} ${CBLAS_LIBRARIES})
    if (CBLAS_FOUND AND LAPACK_FOUND)
        set(OSKAR_MATH_LIBS_FOUND true)
    endif()
endif ()

find_package(Matlab)

# === Print some warning messages if key library modules will be disabled.
if (NOT QT_QTCORE_FOUND)
    message("*****************************************************************")
    message("** WARNING: QT Core not found. ")
    message("*****************************************************************")
endif()

if (NOT CUDA_FOUND)
    message("*****************************************************************")
    message("** WARNING: CUDA not found: "
        "Unable to build OSKAR CUDA library!")
    message("*****************************************************************")
endif()

if (NOT Qwt5_FOUND)
    message("*****************************************************************")
    message("** WARNING: Qwt5 not found: "
        "Unable to build plotting widgets library!")
    message("*****************************************************************")
endif()

if (NOT CPPUNIT_FOUND)
    message("*****************************************************************")
    message("** WARNING: CppUnit not found: "
        "Unable to build unit testing binaries!")
    message("*****************************************************************")
endif()

if (NOT CASACORE_FOUND)
    message("*****************************************************************")
    message("** WARNING: CasaCore not found: "
        "Unable to build OSKAR measurement set library!")
    message("*****************************************************************")
endif()


if (NOT MATLAB_FOUND)
    message("*****************************************************************")
    message("** WARNING: MATLAB not found: "
        "Unable to compile MATLAB mex functions!")
    message("*****************************************************************")
endif()


set(DEPENDENCIES_FOUND TRUE)
