#
# src/matlab/image/CMakeLists.txt
#

# Create mex utility library for handling images
add_library(mex_image STATIC
    lib/src/oskar_mex_image_settings_from_matlab.cpp
    lib/src/oskar_mex_image_to_matlab_struct.cpp)

# Build mex functions
set(OSKAR_MEX_INSTALL_DIR "${OSKAR_MATLAB_INSTALL_DIR}/+oskar/+image")

#oskar_mex(read oskar_mex_image_read.cpp LIBS oskar mex_image)
oskar_mex(make oskar_mex_make_image.cpp LIBS oskar mex_image mex_vis)

# Install scripts
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/scripts/"
    DESTINATION "${OSKAR_MEX_INSTALL_DIR}"
    FILES_MATCHING PATTERN "*.m"
    PATTERN ".svn" EXCLUDE)
