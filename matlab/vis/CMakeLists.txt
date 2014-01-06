#
# src/matlab/vis/CMakeLists.txt
#

# Create mex utility library for handling visibilities
add_library(mex_vis STATIC
    lib/src/oskar_mex_vis_from_matlab_struct.cpp
    lib/src/oskar_mex_vis_to_matlab_struct.cpp)

# Build mex functions.
set(OSKAR_MEX_INSTALL_DIR "${OSKAR_MATLAB_INSTALL_DIR}/+oskar/+vis")

oskar_mex(read oskar_mex_vis_read.cpp LIBS oskar mex_vis)
oskar_mex(write oskar_mex_vis_write.cpp LIBS oskar mex_vis)
oskar_mex(evaluate_baseline_uvw oskar_mex_evaluate_baseline_uvw.cpp
    LIBS oskar mex_vis)

# Install scripts
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/scripts/"
    DESTINATION "${OSKAR_MEX_INSTALL_DIR}"
    FILES_MATCHING PATTERN "*.m"
    PATTERN ".svn" EXCLUDE)
