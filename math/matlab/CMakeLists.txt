#
# src/math/matlab/CMakeLists.txt
#

if (MATLAB_FOUND)
    include(oskar_build_macros)
    oskar_mex(oskar_random_broken_power_law oskar_mex_random_broken_power_law.cpp)
    oskar_mex(oskar_spherical_positions oskar_mex_spherical_positions.cpp)
    oskar_mex(oskar_sph_to_lm oskar_mex_sph_to_lm.cpp)
    oskar_mex(oskar_sph_rotate_to_position oskar_mex_sph_rotate_to_position.cpp)
    oskar_mex(oskar_sph_rotate_points oskar_mex_sph_rotate_points.cpp)
    oskar_mex(oskar_sph_from_lm oskar_mex_sph_from_lm.cpp)
    oskar_mex(oskar_fit_ellipse oskar_mex_fit_ellipse.cpp)
    target_link_libraries(oskar_fit_ellipse
         /data/MATLAB/R2011b/bin/glnxa64/libmwlapack.so
         /data/MATLAB/R2011b/bin/glnxa64/libmwblas.so
    )

    add_subdirectory("@Jones")

endif (MATLAB_FOUND)

