#
# src/sky/matlab/CMakeLists.txt
#

if (MATLAB_FOUND)

    include(oskar_build_macros)
    
    oskar_mex(oskar_date_time_to_mjd  oskar_mex_date_time_to_mjd.cpp)
    oskar_mex(oskar_mjd_to_gast       oskar_mex_mjd_to_gast.cpp)
    oskar_mex(oskar_mjd_to_gmst       oskar_mex_mjd_to_gmst.cpp)
    oskar_mex(oskar_mjd_to_last_fast  oskar_mex_mjd_to_last_fast.cpp)
    oskar_mex(oskar_ra_dec_to_az_el   oskar_mex_ra_dec_to_az_el.cpp)
    oskar_mex(oskar_ra_dec_to_hor_lmn oskar_mex_ra_dec_to_hor_lmn.cpp)
    
    add_subdirectory("@SkyModel")
    
endif (MATLAB_FOUND)

