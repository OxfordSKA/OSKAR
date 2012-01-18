#
# src/sky/matlab/CMakeLists.txt
#

if (MATLAB_FOUND)

    include(oskar_build_macros)
    
    set(OSKAR_MEX_INSTALL_DIR "${OSKAR_MATLAB_INSTALL_DIR}/+sky")
    oskar_mex(date_time_to_mjd  oskar_mex_date_time_to_mjd.cpp)
    oskar_mex(mjd_to_gast       oskar_mex_mjd_to_gast.cpp)
    oskar_mex(mjd_to_gmst       oskar_mex_mjd_to_gmst.cpp)
    oskar_mex(mjd_to_last_fast  oskar_mex_mjd_to_last_fast.cpp)
    oskar_mex(ra_dec_to_az_el   oskar_mex_ra_dec_to_az_el.cpp)
    oskar_mex(ra_dec_to_hor_lmn oskar_mex_ra_dec_to_hor_lmn.cpp)
    oskar_mex(load_source_file  oskar_mex_sky_model_load.cpp)
    
    add_subdirectory("@SkyModel")
    
endif (MATLAB_FOUND)

