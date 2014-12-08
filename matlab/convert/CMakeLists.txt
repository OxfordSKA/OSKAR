#
# src/matlab/convert/CMakeLists.txt
#

set(OSKAR_MEX_INSTALL_DIR "${OSKAR_MATLAB_INSTALL_DIR}/+oskar/+convert")

oskar_mex(mjd_ut1_to_era oskar_mex_convert_mjd_ut1_to_era.cpp LIBS oskar)
oskar_mex(mjd_to_gast_fast oskar_mex_convert_mjd_to_gast_fast.cpp LIBS oskar) # Deprecated
#oskar_mex(ra_dec_to_az_el  oskar_mex_convert_ra_dec_to_az_el.cpp LIBS oskar)
oskar_mex(date_time_to_mjd oskar_mex_convert_date_time_to_mjd.cpp LIBS oskar)
