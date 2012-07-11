#
# src/matlab/sky/CMakeLists.txt
#

set(OSKAR_MEX_INSTALL_DIR "${OSKAR_MATLAB_INSTALL_DIR}/+oskar/+sky")

oskar_mex(load oskar_mex_sky_model_load.cpp)
oskar_mex(mjd_to_last_fast oskar_mex_mjd_to_last_fast.cpp)
oskar_mex(ra_dec_to_az_el oskar_mex_ra_dec_to_az_el.cpp)
oskar_mex(date_time_to_mjd oskar_mex_date_time_to_mjd.cpp)
