#
# src/math/matlab/CMakeLists.txt
#

# ==== Matlab mex funtions
if (MATLAB_FOUND)
    include(oskar_build_macros)
    oskar_mex(oskar_random_broken_power_law oskar_mex_random_broken_power_law.cpp)
    oskar_mex(oskar_spherical_positions oskar_mex_spherical_positions.cpp)

    oskar_mex(oskar_Jones_constructor oskar_mex_Jones_constructor.cpp)
    oskar_mex(oskar_Jones_destructor oskar_mex_Jones_destructor.cpp)
    oskar_mex(oskar_Jones_copy oskar_mex_Jones_copy.cpp)
    oskar_mex(oskar_Jones_get_values oskar_mex_Jones_get_values.cpp)
    oskar_mex(oskar_Jones_set_real_scalar oskar_mex_Jones_set_real_scalar.cpp)
    oskar_mex(oskar_Jones_get_parameter oskar_mex_Jones_get_parameter.cpp)
    oskar_mex(oskar_Jones_set_values oskar_mex_Jones_set_values.cpp)
endif ()

