#
# src/matlab/settings/CMakeLists.txt
#

set(OSKAR_MEX_INSTALL_DIR "${OSKAR_MATLAB_INSTALL_DIR}/+oskar/+settings")
oskar_qt_mex(set oskar_mex_set_setting.cpp)
