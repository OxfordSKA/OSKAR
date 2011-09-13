#
# src/sky/matlab/CMakeLists.txt
#

if (MATLAB_FOUND)

    include(oskar_build_macros)
    oskar_mex(oskar_ra_dec_to_az_el_d oskar_mex_ra_dec_to_az_el_d.cu)

endif (MATLAB_FOUND)

