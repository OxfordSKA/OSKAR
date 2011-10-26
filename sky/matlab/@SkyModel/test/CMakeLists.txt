#
# src/sky/matlab/@SkyModel/test/CMakeLists.txt
#

if (MATLAB_FOUND)
    # Copy *.m files from current directory to destination.
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/"
        DESTINATION "${OSKAR_MATLAB_INSTALL_DIR}/test"
        FILES_MATCHING PATTERN "*.m"
        PATTERN ".svn" EXCLUDE)

endif ()