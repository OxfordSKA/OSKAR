#
# src/matlab/image/CMakeLists.txt
#

# Build mex functions
set(OSKAR_MEX_INSTALL_DIR "${OSKAR_MATLAB_INSTALL_DIR}/+oskar/+image")

oskar_mex(make oskar_mex_make_image.cpp LIBS oskar oskar_imager)

# Install scripts
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/scripts/"
    DESTINATION "${OSKAR_MEX_INSTALL_DIR}"
    FILES_MATCHING PATTERN "*.m"
    PATTERN ".svn" EXCLUDE)
