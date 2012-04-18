#
# src/matlab/visibilities/CMakeLists.txt
#

# Create mex utility library for handling visibilities
add_library(mex_visibilities STATIC
    lib/src/oskar_mex_vis_from_matlab_struct.cpp
    lib/src/oskar_mex_vis_to_matlab_struct.cpp)

# Build mex functions.
set(OSKAR_MEX_INSTALL_DIR "${OSKAR_MATLAB_INSTALL_DIR}/+oskar/+visibilities")

oskar_mex(read oskar_mex_visibilities_read.cpp EXTRA_LIBS mex_visibilities)

# Install scripts
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/scripts/"
    DESTINATION "${OSKAR_MEX_INSTALL_DIR}"
    FILES_MATCHING PATTERN "*.m"
    PATTERN ".svn" EXCLUDE)
