#
# src/cmake/dependencies.cmake:
#
#
# Dependencies for liboskar:
#------------------------------------------------------------------------------
# [required]
#   CUDA (>= 4.0)
#
# [optional]
#   CppUnit
#   Qt4
#   Qwt5-qt4
#
# Dependencies for liboskar_ms:
#------------------------------------------------------------------------------
#  casacore
#  CppUnit
#
# Dependencies for liboskar_imaging:
#------------------------------------------------------------------------------
#   FFTW
#   CFitsio
#   CppUnit
#
# Dependencies for liboskar_widgets:
#------------------------------------------------------------------------------
#   Qt4
#   Qwt5
#   CppUnit
#   CBLAS (either ATLAS or MKL)
#   LAPACK (either ATLAS or MKL)
#

# ==== OS specific path settings.
if (WIN32)
    # qwt5
    set(QWT_INCLUDES     ${CMAKE_SOURCE_DIR}/../include/qwt-5.2.2/)
    set(QWT_LIBRARY_DIR  ${CMAKE_SOURCE_DIR}/../lib/qwt-5.2.2/)

    # cppunit
    set(CPPUNIT_INCLUDES ${CMAKE_SOURCE_DIR}/../include/cppunit-1.12.1/)
    set(CPPUNIT_LIB_DIR  ${CMAKE_SOURCE_DIR}/../lib/cppunit-1.12.1/)
endif ()

# ==== Find dependencies.
find_package(CUDA 4.0 QUIET)
#find_package(Qt4 4.5 COMPONENTS QtCore QtGui QtXml QtOpenGL QtTest QUIET)
#find_package(Qwt5 QUIET)
find_package(OpenMP QUIET)
find_package(CppUnit QUIET)
find_package(FFTW3 QUIET)
find_package(CasaCore QUIET)
find_package(CFitsio QUIET)
find_package(Matlab QUIET)


# ==== Work out which libraries to build.
if (NOT CUDA_FOUND)
    message("*****************************************************************")
    message("** WARNING: CUDA not found: "
            "Unable to build OSKAR CUDA library!")
    message("*****************************************************************")
    set(BUILD_OSKAR FALSE)
endif ()

if (NOT MATLAB_FOUND)
    message("*****************************************************************")
    message("** WARNING: MATLAB not found: "
            "Unable to compile MATLAB mex functions!")
    message("*****************************************************************")
endif()

if (NOT QT4_FOUND)
    message("*****************************************************************")
    message("** WARNING: QT4 not found. ")
    message("*****************************************************************")
    set(BUILD_OSKAR_WIDGETS FALSE)
    set(BUILD_OSKAR_IMAGING FALSE)
endif()

if (NOT Qwt5_FOUND)
    message("*****************************************************************")
    message("** WARNING: Qwt5 not found: "
        "Unable to build plotting widgets library!")
    message("*****************************************************************")
    set(BUILD_OSKAR_WIDGETS FALSE)
endif()

if (NOT CASACORE_FOUND)
    message("*****************************************************************")
    message("** WARNING: CasaCore not found: "
        "Unable to build OSKAR measurement set library!")
    message("*****************************************************************")
    set(BUILD_OSKAR_MS FALSE)
endif()

if (NOT FFTW3_FOUND)
    message("*****************************************************************")
    message("** WARNING: FFTW3 not found: "
            "Unable to build imaging library!")
    message("*****************************************************************")
    set(BUILD_OSKAR_IMAGING FALSE)
endif ()

if (NOT CPPUNIT_FOUND)
    message("*****************************************************************")
    message("** WARNING: CppUnit not found: "
           "Unable to build unit testing binaries!")
    message("*****************************************************************")
endif()



# ==== Prints a message saying which libraries are being built.
if (BUILD_OSKAR)
    message("==> Building 'liboskar'")
endif ()
if (BUILD_OSKAR_MS)
    message("==> Building 'liboskar_ms'")
endif ()
if (BUILD_OSKAR_IMAGING)
    message("==> Building 'liboskar_imaging'")
endif ()
if (BUILD_OSKAR_WIDGETS)
    message("==> Building 'liboskar_widgets'")
endif ()

# ==== Set a flag to tell cmake that dependencies have been checked.
set(CHECKED_DEPENDENCIES YES)
