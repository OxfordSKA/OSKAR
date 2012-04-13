#
# src/matlab/imager/CMakeLists.txt
#

set(OSKAR_MEX_INSTALL_DIR "${OSKAR_MATLAB_INSTALL_DIR}/+oskar/+imager")

oskar_mex(run oskar_mex_make_image)
