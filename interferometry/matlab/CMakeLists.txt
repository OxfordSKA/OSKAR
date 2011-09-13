#
# src/interferometry/matlab/CMakeLists.txt
#

# ==== Matlab mex funtions
if (MATLAB_FOUND)
    include(oskar_build_macros)
    oskar_mex(interferometer1_scalar_d oskar_interferometer1_scalar_d_mex.cpp)   
endif (MATLAB_FOUND)

