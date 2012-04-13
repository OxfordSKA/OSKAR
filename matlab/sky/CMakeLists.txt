#
# src/matlab/sky/CMakeLists.txt
#

set(OSKAR_MEX_INSTALL_DIR "${OSKAR_MATLAB_INSTALL_DIR}/+oskar/+sky")

oskar_mex(load oskar_mex_sky_model_load.cpp)
