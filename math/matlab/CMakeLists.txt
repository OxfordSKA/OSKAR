#
# src/math/matlab/CMakeLists.txt
#

# ==== Matlab mex funtions
if (MATLAB_FOUND)
    include(oskar_build_macros)
    oskar_mex(oskar_random_broken_power_law oskar_mex_random_broken_power_law.cpp)
    oskar_mex(oskar_spherical_positions oskar_mex_spherical_positions.cpp)

    add_subdirectory("@Jones")

endif ()

