#
# src/interferometry/matlab/CMakeLists.txt
#

# ==== Matlab mex funtions
if (MATLAB_FOUND)
    include(oskar_build_macros)
    
    include_directories(${CUDA_INCLUDE_DIRS})
    
    oskar_mex(oskar_interferometer1_scalar_d oskar_mex_interferometer1_scalar_d.cpp)
    
    add_subdirectory(@Visibilities)
    
endif (MATLAB_FOUND)

