#
# src/cmake/oskar_dependencies.cmake:
#
#
# Dependencies:
#------------------------------------------------------------------------------
#
#   CUDA (>= 4.0)   (oskar, oskar_apps, oskar_widgets, OSKAR applications)
#   Qt4 (>=4.6)     (oskar_apps, oskar_widgets, OSKAR applications)
#   MKL             (oskar -> to enable extended sources)
#   CBLAS           (oskar -> to enable extended sources)
#   LAPACK          (oskar -> to enable extended sources)
#   casacore        (oskar_ms)
#   cfitsio         (oskar_fits)
#   MATLAB          (for MATLAB interface fuctions)
#
# =============================================================================
#


# === Append the src/cmake directory to the module path.
list(INSERT CMAKE_MODULE_PATH 0 ${OSKAR_SOURCE_DIR}/cmake/modules)

# OS specific path settings.
if (WIN32)
    # qwt5
    set(QWT_INCLUDES     ${CMAKE_SOURCE_DIR}/../include/qwt-5.2.2/)
    set(QWT_LIBRARY_DIR  ${CMAKE_SOURCE_DIR}/../lib/qwt-5.2.2/)
endif ()

# ==== Find dependencies.
find_package(CUDA 4.0 QUIET) # liboskar
find_package(OpenMP QUIET)   # liboskar
#find_package(MKL QUIET)     # liboskar
find_package(CBLAS QUIET)    # liboskar
find_package(LAPACK QUIET)   # liboskar
find_package(Qt4 4.6 QUIET) # liboskar_apps, liboskar_widgets, apps
find_package(CasaCore QUIET) # liboskar_ms
find_package(CFitsio QUIET)  # liboskar_fits
find_package(Matlab QUIET)   # mex functions
#find_package(PNG QUIET)     # For writing PNG images.

# ==== Work out which libraries to build.
if (NOT CUDA_FOUND)
    message("===============================================================================")
    message("-- WARNING: CUDA toolkit not found: Unable to build main OSKAR library.")
    message("===============================================================================")
elseif (NOT CUDA_CUDA_LIBRARY)
    message("===============================================================================")
    message("-- WARNING: CUDA driver library not found: Unable to build main OSKAR library.")
    message("===============================================================================")
    set(CUDA_FOUND FALSE)
endif ()

if (CUDA_FOUND)
    add_definitions(-DOSKAR_HAVE_CUDA)
else ()
    add_definitions(-DOSKAR_NO_CUDA)
endif ()

if (MKL_FOUND)
    message("===============================================================================")
    message("INFO: Using MKL for LAPACK AND BLAS.")
    message("===============================================================================")
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
    message("===============================================================================")
    message("-- WARNING: CBLAS not found.")
    message("===============================================================================")
    add_definitions(-DOSKAR_NO_CBLAS)
endif()

if (NOT OSKAR_USE_LAPACK)
    message("===============================================================================")
    message("-- WARNING: LAPACK not found.")
    message("===============================================================================")
    add_definitions(-DOSKAR_NO_LAPACK)
endif()

if (NOT QT4_FOUND)
    message("===============================================================================")
    message("-- WARNING: Qt4 not found: "
            "Unable to build OSKAR widgets and applications.")
    message("===============================================================================")
endif()

if (NOT CASACORE_FOUND)
    message("===============================================================================")
    message("-- WARNING: CasaCore not found: "
        "Unable to build OSKAR Measurement Set library.")
    message("===============================================================================")
    add_definitions(-DOSKAR_NO_MS)
endif()

if (NOT CFITSIO_FOUND)
    message("===============================================================================")
    message("-- WARNING: CFITSIO not found: "
           "Unable to build OSKAR FITS library.")
    message("===============================================================================")
    add_definitions(-DOSKAR_NO_FITS)
endif ()

if (NOT MATLAB_FOUND)
    message("===============================================================================")
    message("-- WARNING: MATLAB not found: "
            "Unable to build the OSKAR MATLAB interface.")
    message("===============================================================================")
endif()

# Prints a message saying which libraries are being built.
message("===============================================================================")
message("-- INFO: The following OSKAR components will be built:")
set(component_count 0)
if (CUDA_FOUND)
    message("-- INFO:   - liboskar")
    math(EXPR component_count '${component_count}+1')
endif ()
if (CASACORE_FOUND)
    message("-- INFO:   - liboskar_ms")
    math(EXPR component_count '${component_count}+1')
endif ()
if (CFITSIO_FOUND AND CUDA_FOUND)
    message("-- INFO:   - liboskar_fits")
    math(EXPR component_count '${component_count}+1')
endif ()
if (QT4_FOUND)
    message("-- INFO:   - liboskar_widgets")
    math(EXPR component_count '${component_count}+1')
endif ()
if (QT4_FOUND AND CUDA_FOUND)
    message("-- INFO:   - liboskar_apps")
    message("-- INFO:   - OSKAR applications")
    math(EXPR component_count '${component_count}+1')
endif ()
if ("${component_count}" EQUAL 0)
    message("===============================================================================")
    message("== ERROR: Unable to build any OSKAR components, check your dependencies!")
    message("===============================================================================")
    message(FATAL_ERROR "")
endif()
if (MATLAB_FOUND AND CUDA_FOUND)
    message("-- INFO:   - OSKAR MATLAB interface functions")
endif ()
message("===============================================================================")


message("===============================================================================")
message("-- INFO: 'make install' will install OSKAR to:")
message("-- INFO:   - Libraries         ${CMAKE_INSTALL_PREFIX}/${OSKAR_LIB_INSTALL_DIR}/")
message("-- INFO:   - Headers           ${CMAKE_INSTALL_PREFIX}/${OSKAR_INCLUDE_INSTALL_DIR}/")
if (QT4_FOUND AND CUDA_FOUND)
message("-- INFO:   - Applications      ${CMAKE_INSTALL_PREFIX}/${OSKAR_BIN_INSTALL_DIR}/")
endif()
if (MATLAB_FOUND AND CUDA_FOUND)
message("-- INFO:   - MATLAB interface  ${CMAKE_INSTALL_PREFIX}/${OSKAR_MATLAB_INSTALL_DIR}/")
endif()
#message("-- NOTE: These paths can be changed using: '-DCMAKE_INSTALL_PREFIX=<path>'")
message("===============================================================================")

# Set a flag to tell cmake that dependencies have been checked.
set(CHECKED_DEPENDENCIES YES)
