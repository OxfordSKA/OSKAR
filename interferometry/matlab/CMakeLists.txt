#
# src/interferometry/matlab/CMakeLists.txt
#

# ==== Matlab mex funtions
if (MATLAB_FOUND)
    include(oskar_build_macros)

    add_subdirectory(@Visibilities)

endif (MATLAB_FOUND)

