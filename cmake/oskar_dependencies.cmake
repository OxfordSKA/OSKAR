#
# src/cmake/oskar_dependencies.cmake:
#
#
# Dependencies for liboskar:
#------------------------------------------------------------------------------
# [required]
#   CUDA (>= 4.0)
#
# [optional]
#   Qt4       (oskar_apps, oskar_widgets)
#   MKL       (oskar, for extended sources)
#   CBLAS     (oskar, for extended sources)
#   LAPACK    (oskar, for extended sources) 
#   casacore  (oskar_ms)
#   cfitsio   (oskar_fits)
#   MATLAB    (for MATLAB interface fuctions)
#   CppUnit   (unit test binaries)
#
# =============================================================================
#

# OS specific path settings.
if (WIN32)
    # qwt5
    set(QWT_INCLUDES     ${CMAKE_SOURCE_DIR}/../include/qwt-5.2.2/)
    set(QWT_LIBRARY_DIR  ${CMAKE_SOURCE_DIR}/../lib/qwt-5.2.2/)

    # cppunit
    set(CPPUNIT_INCLUDES ${CMAKE_SOURCE_DIR}/../include/cppunit-1.12.1/)
    set(CPPUNIT_LIB_DIR  ${CMAKE_SOURCE_DIR}/../lib/cppunit-1.12.1/)
endif ()

# ==== Find dependencies.
find_package(CUDA 4.0 QUIET) # liboskar
find_package(OpenMP QUIET)   # liboskar
#find_package(MKL QUIET)     # liboskar
find_package(CBLAS QUIET)    # liboskar
find_package(LAPACK QUIET)   # liboskar
find_package(Qt4 4.5 QUIET)  # liboskar_apps, liboskar_widgets, apps 
find_package(CasaCore QUIET) # liboskar_ms
find_package(CFitsio QUIET)  # liboskar_fits
find_package(Matlab QUIET)   # mex functions
find_package(CppUnit QUIET)  # unit tests

# ==== Work out which libraries to build.
if (NOT CUDA_FOUND)
    message("================================================================================")
    message("-- WARNING: CUDA not found: "
            "Unable to build main OSKAR library.")
    message("================================================================================")
endif ()

if (MKL_FOUND)
    message("================================================================================")
    message("INFO: Using MKL for LAPACK AND BLAS.")
    message("================================================================================")
    set(OSKAR_LAPACK ${MKL_LIBRARIES})
    set(OSKAR_BLAS ${MKL_LIBRARIES})
    include_directories(${MKL_INCLUDE_DIR})
    set(OSKAR_USE_LAPACK YES)
    set(OSKAR_USE_CBLAS YES)
    add_definitions(-DOSKAR_MKL_FOUND)
else ()
    if (LAPACK_FOUND)
        set(OSKAR_LAPACK ${LAPACK_LIBRARIES})
        set(OSKAR_USE_LAPACK YES)
    endif ()
    if (CBLAS_FOUND)
        set(OSKAR_USE_CBLAS YES)
        set(OSKAR_CBLAS ${CBLAS_LIBRARIES})
    endif ()
endif ()

if (NOT OSKAR_USE_CBLAS)
    message("================================================================================")
    message("-- WARNING: CBLAS not found.")
    message("================================================================================")
    add_definitions(-DOSKAR_NO_CBLAS)
endif()

if (NOT OSKAR_USE_LAPACK)
    message("================================================================================")
    message("-- WARNING: LAPACK not found.")
    message("================================================================================")
    add_definitions(-DOSKAR_NO_LAPACK)
endif()

if (NOT QT4_FOUND)
    message("================================================================================")
    message("-- WARNING: Qt4 not found: "
            "Unable to build OSKAR widgets and applications.")
    message("================================================================================")
endif()

if (NOT CASACORE_FOUND)
    message("================================================================================")
    message("-- WARNING: CasaCore not found: "
        "Unable to build OSKAR Measurement Set library.")
    message("================================================================================")
    add_definitions(-DOSKAR_NO_MS)
endif()

if (NOT CFITSIO_FOUND)
    message("================================================================================")
    message("-- WARNING: CFITSIO not found: "
           "Unable to build OSKAR FITS library.")
    message("================================================================================")
    add_definitions(-DOSKAR_NO_FITS)
endif ()

if (NOT MATLAB_FOUND)
    message("================================================================================")
    message("-- WARNING: MATLAB not found: "
            "Unable to build the OSKAR MATLAB interface.")
    message("================================================================================")
endif()

if (NOT CPPUNIT_FOUND)
    message("================================================================================")
    message("-- WARNING: CppUnit not found: "
           "Unable to build unit testing binaries.")
    message("================================================================================")
endif()

# Prints a message saying which libraries are being built.
message("================================================================================")
message("-- INFO: The following OSKAR components will be built:")
if (CUDA_FOUND)
    message("        - liboskar")
endif ()
if (CASACORE_FOUND)
    message("        - liboskar_ms")
endif ()
if (CFITSIO_FOUND)
    message("        - liboskar_fits")
endif ()
if (QT4_FOUND AND CUDA_FOUND)
    message("        - liboskar_widgets")
endif ()
if (QT4_FOUND AND CUDA_FOUND)
    message("        - liboskar_apps")
    message("        - OSKAR applications")
endif ()
if (MATLAB_FOUND AND CUDA_FOUND)
    message("        - OSKAR MATLAB interface functions")
endif ()
message("================================================================================")

# Set a flag to tell cmake that dependencies have been checked.
set(CHECKED_DEPENDENCIES YES)
