#
# cmake/oskar_dependencies.cmake:
#

# ==== Find dependencies.
find_package(OpenCL)
find_package(CUDA 5.5)
find_package(OpenMP)
find_package(CasaCore)
#find_package(PNG QUIET)

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

if (NOT CASACORE_FOUND)
    message("===============================================================================")
    message("-- WARNING: CASACORE not found: Unable to use Measurement Sets.")
    message("===============================================================================")
    add_definitions(-DOSKAR_NO_MS)
endif()

if (NOT OPENMP_FOUND)
    message("===============================================================================")
    message("-- WARNING: OpenMP not found: Unable to use multiple GPUs.")
    message("===============================================================================")
endif ()

message("===============================================================================")
message("-- INFO: 'make install' will install OSKAR to:")
message("-- INFO:   - Libraries         ${CMAKE_INSTALL_PREFIX}/${OSKAR_LIB_INSTALL_DIR}")
message("-- INFO:   - Headers           ${CMAKE_INSTALL_PREFIX}/${OSKAR_INCLUDE_INSTALL_DIR}")
message("-- INFO:   - Applications      ${CMAKE_INSTALL_PREFIX}/${OSKAR_BIN_INSTALL_DIR}")
message("-- NOTE: These paths can be changed using: '-DCMAKE_INSTALL_PREFIX=<path>'")
message("===============================================================================")

# Optional verbose printing.
if (BUILD_INFO)
message("===============================================================================")
if (CASACORE_FOUND)
    message("-- INFO: CASACORE : ${CASACORE_LIBRARIES}")
endif()
message("===============================================================================")
endif()

# Set a flag to tell cmake that dependencies have been checked.
set(CHECKED_DEPENDENCIES YES)
