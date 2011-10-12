#
# src/interferometry/matlab/CMakeLists.txt
#

# ==== Matlab mex funtions
if (MATLAB_FOUND)
    include(oskar_build_macros)

    oskar_mex(oskar_interferometer1_scalar_d oskar_mex_interferometer1_scalar_d.cpp)
endif (MATLAB_FOUND)

