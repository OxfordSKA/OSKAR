#
# src/matlab/measures/CMakeLists.txt
#

set(OSKAR_MEX_INSTALL_DIR "${OSKAR_MATLAB_INSTALL_DIR}/+oskar/+measures")

oskar_mex(convert_mjd_to_gast_fast oskar_mex_convert_mjd_to_gast_fast.cpp
    LIBS oskar)
#oskar_mex(convert_ra_dec_to_az_el  oskar_mex_convert_ra_dec_to_az_el.cpp
#    LIBS oskar)
oskar_mex(convert_date_time_to_mjd oskar_mex_convert_date_time_to_mjd.cpp
    LIBS oskar)
