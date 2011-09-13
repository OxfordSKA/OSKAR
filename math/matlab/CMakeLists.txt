#
# src/math/matlab/CMakeLists.txt
#

# ==== Matlab mex funtions
if (MATLAB_FOUND)
    include(oskar_build_macros)
    oskar_mex(oskar_random_bpl oskar_random_bpl.cpp)   
    oskar_mex(oskar_spherical_positions oskar_spherical_positions.cpp)
endif ()

