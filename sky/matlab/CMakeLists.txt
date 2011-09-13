#
# src/sky/matlab/CMakeLists.txt
#

if (MATLAB_FOUND)
    include(oskar_build_macros)
    oskar_mex(oskar_ra_dec_to_az_el_d oskar_mex_ra_dec_to_az_el_d.cu)
    oskar_mex(oskar_ra_dec_to_az_el_f oskar_mex_ra_dec_to_az_el_f.cu)
endif (MATLAB_FOUND)

