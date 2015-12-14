#
# src/matlab/cuda/CMakeLists.txt
#

set(OSKAR_MEX_INSTALL_DIR "${OSKAR_MATLAB_INSTALL_DIR}/+oskar/+cuda")

oskar_mex(device_info oskar_mex_device_info.cpp LIBS ${CUDA_LIBRARIES})
