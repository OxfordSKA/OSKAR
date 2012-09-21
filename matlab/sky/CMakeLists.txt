#
# src/matlab/sky/CMakeLists.txt
#

# Create mex utility library for handling the sky model structure
add_library(mex_sky_model STATIC
    lib/oskar_mex_sky_from_matlab_struct.cpp
    lib/oskar_mex_sky_to_matlab_struct.cpp
    )

set(OSKAR_MEX_INSTALL_DIR "${OSKAR_MATLAB_INSTALL_DIR}/+oskar/+sky")

oskar_mex(load oskar_mex_sky_model_load.cpp EXTRA_LIBS mex_sky_model)    # ASCII
oskar_mex(save oskar_mex_sky_model_save.cpp EXTRA_LIBS mex_sky_model)    # ASCII
oskar_mex(read oskar_mex_sky_model_read.cpp EXTRA_LIBS mex_sky_model)    # BINARY
oskar_mex(write oskar_mex_sky_model_write.cpp EXTRA_LIBS mex_sky_model)  # BINARY

#oskar_mex(open oskar_mex_sky_model_write.cpp EXTRA_LIBS mex_sky_model)  # SMART ASCII/BINARY read/load


oskar_mex(mjd_to_last_fast oskar_mex_mjd_to_last_fast.cpp)
oskar_mex(ra_dec_to_az_el  oskar_mex_ra_dec_to_az_el.cpp)
oskar_mex(date_time_to_mjd oskar_mex_date_time_to_mjd.cpp)
