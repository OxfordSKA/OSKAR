#
# src/cmake/oskar_dependencies.cmake:
#
#
# Dependencies:
#------------------------------------------------------------------------------
#
#   CUDA (>= 4.0)   (oskar, oskar_apps, OSKAR applications)
#   OpenMP          (for multi-GPU support)
#   Qt4 (>=4.6)     (oskar_apps, GUI, OSKAR applications)
#   MKL             (oskar -> to enable extended sources)
#   LAPACK          (oskar -> to enable extended sources)
#   casacore        (oskar_ms)
#   MATLAB          (for MATLAB interface fuctions)
#
# =============================================================================

if (DEFINED LAPACK_LIB_DIR)
    list(INSERT CMAKE_LIBRARY_PATH 0 ${LAPACK_LIB_DIR})
endif()

# ==== Find dependencies.
find_package(CUDA 5.5)              # liboskar
find_package(OpenMP)                # liboskar
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    find_package(MKL)               # liboskar
endif()
if (NOT MKL_FOUND)
    find_package(LAPACK)            # liboskar
endif ()
find_package(Qt4 4.6 COMPONENTS QtCore QtGui QtNetwork) # liboskar_apps, apps
# HACK for using Qt4 frameworks on OS X.
# Avoids having to symlink headers and libraries from the Qt binary installer
# into the system paths.
if (APPLE AND QT_USE_FRAMEWORKS)
    set(QT_QTCORE_LIBRARY ${QT_QTCORE_LIBRARY}/QtCore)
    set(QT_QTGUI_LIBRARY ${QT_QTGUI_LIBRARY}/QtGui)
    set(QT_QTNETWORK_LIBRARY ${QT_QTNETWORK_LIBRARY}/QtNetwork)
endif()
#if (NOT QT4_FOUND)
#    find_package(Qt5Core)
#endif()
find_package(CasaCore)               # liboskar_ms
#find_package(Matlab)                 # mex functions
#find_package(PNG QUIET)             # For writing PNG images
#find_package(PythonLibs 2.7)         # For python interface
#find_package(NumPy 1.8)              # For python interface


if (PYTHONLIBS_FOUND AND NUMPY_FOUND AND PYTHONINTERP_FOUND AND PYTHON_VERSION_MAJOR EQUAL 2)
    set(PYTHON_FOUND TRUE)
endif()

# ==== Work out which libraries to build.
if (NOT CUDA_FOUND)
    message("===============================================================================")
    message("-- WARNING: CUDA toolkit not found: Unable to build main OSKAR library.")
    message("===============================================================================")
elseif (NOT CUDA_CUDA_LIBRARY)
    # Leave this as a warning only, as drivers may not be installed
    # on cluster head nodes.
    message("-- WARNING: CUDA driver library not found.")
endif ()

if (CUDA_FOUND)
    add_definitions(-DOSKAR_HAVE_CUDA)
else ()
    add_definitions(-DOSKAR_NO_CUDA)
endif ()

if (MKL_FOUND)
    message("===============================================================================")
    message("-- INFO: Using MKL for LAPACK.")
    message("===============================================================================")
    set(OSKAR_LAPACK ${MKL_LIBRARIES})
    include_directories(${MKL_INCLUDE_DIR})
    set(OSKAR_USE_LAPACK YES)
    add_definitions(-DOSKAR_MKL_FOUND)
else ()
    if (LAPACK_FOUND)
        set(OSKAR_LAPACK ${LAPACK_LIBRARIES})
        set(OSKAR_USE_LAPACK YES)
    endif ()
endif ()

if (NOT OSKAR_USE_LAPACK)
    message("===============================================================================")
    message("-- WARNING: LAPACK not found.")
    message("===============================================================================")
    add_definitions(-DOSKAR_NO_LAPACK)
endif()

if (NOT QT4_FOUND)
    message("===============================================================================")
    message("-- WARNING: Qt4 not found: "
            "Unable to build OSKAR applications or GUI.")
    message("===============================================================================")
endif()

if (NOT CASACORE_FOUND)
    message("===============================================================================")
    message("-- WARNING: CasaCore not found: "
        "Unable to build OSKAR Measurement Set library.")
    message("===============================================================================")
    add_definitions(-DOSKAR_NO_MS)
endif()

if (NOT OPENMP_FOUND)
    message("===============================================================================")
    message("-- WARNING: OpenMP not found: Unable to use multiple GPUs.")
    message("===============================================================================")
endif ()

#if (NOT MATLAB_FOUND)
#    message("===============================================================================")
#    message("-- WARNING: MATLAB not found: "
#            "Unable to build the OSKAR MATLAB interface.")
#    message("===============================================================================")
#endif()

#if (NOT PYTHON_FOUND)
#    message("===============================================================================")
#    message("-- WARNING: Python 2.7 not found: "
#            "Unable to build the OSKAR Python interface.")
#    message("===============================================================================")
#endif()

# Prints a message saying which components are being built.
message("===============================================================================")
message("-- INFO: The following OSKAR components will be built:")
set(component_count 0)
if (CUDA_FOUND)
    message("-- INFO:   - liboskar")
    message("-- INFO:   - liboskar_fits")
    math(EXPR component_count '${component_count}+2')
endif ()
if (CASACORE_FOUND)
    message("-- INFO:   - liboskar_ms")
    math(EXPR component_count '${component_count}+1')
endif ()
if (QT4_FOUND AND CUDA_FOUND)
    message("-- INFO:   - liboskar_apps")
    message("-- INFO:   - OSKAR command line applications")
    math(EXPR component_count '${component_count}+1')
endif ()
if (QT4_FOUND)
    message("-- INFO:   - OSKAR GUI")
    math(EXPR component_count '${component_count}+1')
endif ()
if ("${component_count}" EQUAL 0)
    message("===============================================================================")
    message("== ERROR: Unable to build any OSKAR components, check your dependencies!")
    message("===============================================================================")
    message(FATAL_ERROR "")
endif()
#if (MATLAB_FOUND AND CUDA_FOUND)
#    message("-- INFO:   - OSKAR MATLAB interface functions")
#endif()
#if (PYTHON_FOUND AND CUDA_FOUND)
#    message("-- INFO:   - OSKAR Python interface functions (experimental)")
#endif()
message("===============================================================================")

message("===============================================================================")
message("-- INFO: 'make install' will install OSKAR to:")
message("-- INFO:   - Libraries         ${CMAKE_INSTALL_PREFIX}/${OSKAR_LIB_INSTALL_DIR}")
message("-- INFO:   - Headers           ${CMAKE_INSTALL_PREFIX}/${OSKAR_INCLUDE_INSTALL_DIR}")
if (QT4_FOUND AND CUDA_FOUND)
message("-- INFO:   - Applications      ${CMAKE_INSTALL_PREFIX}/${OSKAR_BIN_INSTALL_DIR}")
endif()
#if (MATLAB_FOUND AND CUDA_FOUND)
#message("-- INFO:   - MATLAB interface  ${CMAKE_INSTALL_PREFIX}/${OSKAR_MATLAB_INSTALL_DIR}")
#endif()
#if (PYTHON_FOUND AND CUDA_FOUND)
#message("-- INFO:   - Python interface  ${CMAKE_INSTALL_PREFIX}/${OSKAR_PYTHON_INSTALL_DIR}")
#endif()
#message("-- NOTE: These paths can be changed using: '-DCMAKE_INSTALL_PREFIX=<path>'")
message("===============================================================================")

# Optional verbose printing.
if (BUILD_INFO)
message("===============================================================================")
if (LAPACK_FOUND)
    message("-- INFO: LAPACK   : ${LAPACK_LIBRARIES}")
endif ()
if (CASACORE_FOUND)
    message("-- INFO: CASACORE : ${CASACORE_LIBRARIES}")
endif()
if (QT4_FOUND)
    message("-- INFO: QT4      : ${QT_QTCORE_LIBRARY}")
    message("--                : ${QT_QTGUI_LIBRARY}")
    message("--                : ${QT_QTNETWORK_LIBRARY}")
endif()
message("===============================================================================")
endif()

# Set a flag to tell cmake that dependencies have been checked.
set(CHECKED_DEPENDENCIES YES)
