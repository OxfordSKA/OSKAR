#
# src/matlab/time_freq_data/CMakeLists.txt
#

set(OSKAR_MEX_INSTALL_DIR "${OSKAR_MATLAB_INSTALL_DIR}/+oskar/+time_freq_data")

# Install scripts
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/"
    DESTINATION "${OSKAR_MEX_INSTALL_DIR}"
    FILES_MATCHING PATTERN "*.m"
    PATTERN ".svn" EXCLUDE)
