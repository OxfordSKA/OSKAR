#
# src/sky/matlab/CMakeLists.txt
#

if (MATLAB_FOUND)
    include(oskar_build_macros)
    oskar_mex(oskar_ra_dec_to_az_el_d oskar_mex_ra_dec_to_az_el_d.cu)
    oskar_mex(oskar_ra_dec_to_az_el_f oskar_mex_ra_dec_to_az_el_f.cu)
    oskar_mex(oskar_mjd_to_last_fast  oskar_mex_mjd_to_last_fast.cpp)
    oskar_mex(oskar_date_time_to_mjd  oskar_mex_date_time_to_mjd.cpp)
    oskar_mex(oskar_mjd_to_gmst       oskar_mex_mjd_to_gmst.cpp)
endif (MATLAB_FOUND)

