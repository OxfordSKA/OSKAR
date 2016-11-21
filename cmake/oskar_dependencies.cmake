#
# cmake/oskar_dependencies.cmake:
#

# ==== Find dependencies.
find_package(OpenCL)
find_package(CUDA 5.5)
find_package(OpenMP)
find_package(CasaCore)
find_package(Qt4 4.6 COMPONENTS QtCore QtGui QtNetwork)
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
#find_package(PNG QUIET)             # For writing PNG images

find_package(PythonLibs 2.7)
find_package(NumPy 1.8)
if (PYTHONLIBS_FOUND AND NUMPY_FOUND AND PYTHONINTERP_FOUND AND PYTHON_VERSION_MAJOR EQUAL 2)
    set(PYTHON_FOUND TRUE)
endif()

# ==== Work out what we can build.
if (NOT CUDA_FOUND)
    message("===============================================================================")
    message("-- WARNING: CUDA toolkit not found: Unable to use any GPUs.")
    message("===============================================================================")
elseif (NOT CUDA_CUDA_LIBRARY)
    # Leave this as a note only, as drivers may not be installed
    # on cluster head nodes.
    message("-- NOTE: CUDA driver library not found.")
endif ()

if (CUDA_FOUND)
    add_definitions(-DOSKAR_HAVE_CUDA)
endif ()

if (NOT QT4_FOUND)
    message("===============================================================================")
    message("-- WARNING: Qt4 not found: "
            "Unable to build OSKAR applications or GUI.")
    message("===============================================================================")
endif()

if (NOT CASACORE_FOUND)
    message("===============================================================================")
    message("-- WARNING: CASACORE not found: Unable to write Measurement Sets.")
    message("===============================================================================")
    add_definitions(-DOSKAR_NO_MS)
endif()

if (NOT OPENMP_FOUND)
    message("===============================================================================")
    message("-- WARNING: OpenMP not found: Unable to use multiple GPUs.")
    message("===============================================================================")
endif ()

# Prints a message saying which components are being built.
message("===============================================================================")
message("-- INFO: The following OSKAR components will be built:")
message("-- INFO:   - liboskar")
if (CASACORE_FOUND)
    message("-- INFO:   - liboskar_ms")
endif ()
if (QT4_FOUND)
    message("-- INFO:   - OSKAR command line applications")
    message("-- INFO:   - OSKAR GUI")
endif ()
if (PYTHON_FOUND)
    message("-- INFO:   - OSKAR Python interface functions (experimental)")
endif()
message("===============================================================================")

message("===============================================================================")
message("-- INFO: 'make install' will install OSKAR to:")
message("-- INFO:   - Libraries         ${CMAKE_INSTALL_PREFIX}/${OSKAR_LIB_INSTALL_DIR}")
message("-- INFO:   - Headers           ${CMAKE_INSTALL_PREFIX}/${OSKAR_INCLUDE_INSTALL_DIR}")
if (QT4_FOUND)
message("-- INFO:   - Applications      ${CMAKE_INSTALL_PREFIX}/${OSKAR_BIN_INSTALL_DIR}")
endif()
if (PYTHON_FOUND)
message("-- INFO:   - Python interface  ${CMAKE_INSTALL_PREFIX}/${OSKAR_PYTHON_INSTALL_DIR}")
endif()
#message("-- NOTE: These paths can be changed using: '-DCMAKE_INSTALL_PREFIX=<path>'")
message("===============================================================================")

# Optional verbose printing.
if (BUILD_INFO)
message("===============================================================================")
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
