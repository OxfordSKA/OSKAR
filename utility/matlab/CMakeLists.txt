#
# src/utility/matlab/CMakeLists.txt
#

if (MATLAB_FOUND)
    include(oskar_build_macros)
    oskar_mex(oskar_cuda_check_error oskar_mex_cuda_check_error.cu)
    oskar_mex(oskar_cuda_get_device_memory oskar_mex_cuda_get_device_memory.cu)
endif (MATLAB_FOUND)

