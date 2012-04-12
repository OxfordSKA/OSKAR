#
# src/matlab/visibilities/CMakeLists.txt
#

# Create mex utility library for handling visibilities
add_library(mex_visibilties STATIC
    lib/src/oskar_mex_vis_from_matlab_struct.cpp
    lib/src/oskar_mex_vis_to_matlab_struct.cpp)

# Build mex functions.
set(OSKAR_MEX_INSTALL_DIR "${OSKAR_MATLAB_INSTALL_DIR}/+oskar/+visibilities")

oskar_mex(read oskar_mex_visibilities_read.cpp mex_visibilties)
