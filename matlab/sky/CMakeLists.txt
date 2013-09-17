#
# src/matlab/sky/CMakeLists.txt
#

# Create mex utility library for handling the sky model structure
add_library(mex_sky_model STATIC
    lib/oskar_mex_sky_from_matlab_struct.cpp
    lib/oskar_mex_sky_to_matlab_struct.cpp
)

set(OSKAR_MEX_INSTALL_DIR "${OSKAR_MATLAB_INSTALL_DIR}/+oskar/+sky")

oskar_mex(load oskar_mex_sky_model_load.cpp
    LIBS oskar mex_sky_model)
oskar_mex(save oskar_mex_sky_model_save.cpp
    LIBS oskar mex_sky_model)
oskar_mex(read oskar_mex_sky_model_read.cpp
    LIBS oskar mex_sky_model)
oskar_mex(write oskar_mex_sky_model_write.cpp
    LIBS oskar mex_sky_model)
oskar_mex(mjd_to_last_fast oskar_mex_mjd_to_last_fast.cpp
    LIBS oskar)
oskar_mex(ra_dec_to_az_el  oskar_mex_ra_dec_to_az_el.cpp
    LIBS oskar)
oskar_mex(date_time_to_mjd oskar_mex_date_time_to_mjd.cpp
    LIBS oskar)
