#
# src/utility/matlab/CMakeLists.txt
#

if (MATLAB_FOUND)
    include(oskar_build_macros)
    oskar_mex(oskar_check_cuda_error oskar_check_cuda_error_mex.cu)
    oskar_mex(oskar_check_cuda_device_memory oskar_check_cuda_device_memory_mex.cu)
endif (MATLAB_FOUND)

